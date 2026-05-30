# Relay CSV/JSON Optimization Plan
## Based on Deep Research (Papers, Production Systems, SIMD Techniques)

**Current State:**
- CSV boundary detection: 6.46 GiB/s (SWAR, 8 bytes/word)
- CSV full read: 239 MiB/s (1M rows, 10 cols)
- NDJSON full read: 566 MiB/s (1M rows, 10 cols)
- Gap vs Polars CSV: ~15x slower

---

## Root Cause Analysis

The gap between boundary detection (6.46 GiB/s) and full read (239 MiB/s = ~250 MiB/s) is **26x**. The bottleneck is NOT in finding newlines/quotes. It's in:

1. **Field extraction**: Scalar byte-by-byte parsing after boundary detection
2. **Type conversion**: String → i64/f64/bool via `str::parse()` (allocates, branch-heavy)
3. **Arrow builder overhead**: Per-field `append_value()` with dynamic dispatch
4. **No column-first decode**: All columns parsed row-by-row
5. **No SIMD field parsing**: 8-byte SWAR for boundaries, but scalar for fields

---

## Optimization Roadmap (6 Phases)

### Phase 6A: SIMD Structural Scanning (PCLMULQDQ)
**Target: 10+ GiB/s boundary detection**

**Paper**: simdjson (Langdale & Lemire, VLDB Journal 2019, arXiv:1902.08318)
**Technique**: `PCLMULQDQ` carryless multiplication for O(1) quote parity per 64-byte chunk

Current SWAR processes 8 bytes at a time with `eq_bytes()`. simdjson's approach:
1. Load 64 bytes into AVX2 register
2. Compare against structural characters (comma, quote, newline) via `vpcmpeqb`
3. `vpmovmskb` → 64-bit bitmask (1 bit per byte)
4. Quote parity via `PCLMULQDQ`: carryless multiply of quote bitmask against `0xFFFF...`
5. Result: each bit = "this structural char is outside quotes"

**Key insight**: PCLMULQDQ computes prefix XOR in O(1) per 64 bytes. No loops, no state tracking.

**Implementation**:
```rust
// Stage 1: SIMD structural index (64 bytes at a time)
// - AVX2: 32 bytes/iteration with vpcmpeqb + vpmovmskb
// - AVX-512: 64 bytes/iteration with vptestmb
// - NEON: 16 bytes/iteration with vceqq_u8

// Stage 2: PCLMULQDQ quote masking
// - Carryless multiply quote_mask * 0xFFFF... → prefix XOR
// - O(1) per 64-byte chunk
```

**References**:
- csimdv-rs (Rust): 61% faster than state-machine SIMD on AVX-512
- chunkofcoal.com/posts/simd-csv/: Full ARM NEON walkthrough, ~4 GiB/s
- simdcsv (Go/Minio): 1 GB/s parsing, AVX2

**Expected**: 10-15 GiB/s boundary detection (vs current 6.46 GiB/s)

---

### Phase 6B: Two-Stage Architecture (Structural Index → Field Extraction)
**Target: 500+ MiB/s full read**

**Paper**: Speculative Distributed CSV Parsing (Ge et al., SIGMOD 2019)
**Key insight**: Separate "find structure" (SIMD, fast) from "extract fields" (scalar with bitmasks)

Current: Single-pass, interleaved boundary detection + field extraction.

New architecture:
```
Stage 1: SIMD structural scan → Vec<StructuralBitmask>
  - 64-byte chunks, AVX2/NEON
  - Output: bitmask per chunk (comma positions, newline positions, quote positions)
  - PCLMULQDQ for quote parity
  - Pure SIMD, no branches

Stage 2: Extract fields from bitmasks
  - Use bitmasks to find field boundaries
  - Extract byte slices for each field
  - Parse types from byte slices (not strings)
  - Can parallelize per-column
```

**Benefits**:
- Stage 1 is branch-free, predictable
- Stage 2 can be parallelized per-column (column-first decode)
- No string allocation for intermediate values

---

### Phase 6C: Column-First Decode (Type-Specialized Parsing)
**Target: 1+ GiB/s full read**

**Papers**:
- CUDAFastCSV (Kumaigorodski et al., BTW 2021, Best Paper): GPU column-first via "tapes"
- Deephaven CSV Reader (2022): Two-phase parsing, 35% faster than FastCSV
- libvroom (2025): SIMD type inference, 4.7 GB/s single-threaded

**Technique**: Parse each column independently with type-specialized code

```
Row-oriented (current):
  for each row:
    for each column:
      extract field string → parse type → append to builder

Column-first (new):
  1. Find all delimiter positions (SIMD)
  2. Transpose to columnar layout (field positions per column)
  3. For each column (parallel):
     - Int64: SIMD parse integers from byte slices
     - Float64: SIMD parse floats
     - Utf8: extract string slices (zero-copy if possible)
     - Boolean: SIMD match "true"/"false"/"1"/"0"
```

**Key techniques**:
1. **Integer parsing without string intermediate**: Parse `b"12345"` directly to i64
   - `b"12345"` → 1*10000 + 2*1000 + 3*100 + 4*10 + 5
   - Can vectorize with SIMD: process 4-8 integers at once
   - Reference: fast_float (Lemire), Eisel-Lemire algorithm

2. **Boolean parsing via SIMD**: `vpcmpeqb` against "true"/"false" masks

3. **Zero-copy string extraction**: Just store (offset, length) pointers into mmap

**Expected**: 1-2 GiB/s full read (vs current 239 MiB/s)

---

### Phase 6D: State Machine with Cache-Friendly Layout
**Target: 30% speedup on quoted fields**

**Reference**: DuckDB CSV Parser 2.0 (PR #10209, #14260, 2023-2024)

**Technique**: Flip state machine dimensions for cache locality
```
Current: state_machine[state][char] → next_state
New:     state_machine[char][state] → next_state

Why: When processing a character, we access all possible current states
for that character. [char][state] layout keeps those in the same cache line.
```

Also: Pre-computed skip lists for STANDARD state (most common)
- In STANDARD state, most characters don't change state
- Pre-compute "how many bytes until next structural character"
- Use `memchr`-like SIMD search for the fast path

---

### Phase 6E: SIMD Field Parsing (Vectorized Type Conversion)
**Target: 2+ GiB/s full read**

**Reference**: fast_float (Lemire), simdutf

**Techniques**:
1. **SIMD integer parsing**: Parse 8 integers simultaneously
   - Load 8 byte slices into SIMD registers
   - Subtract ASCII '0' → digit values
   - Horner's method with SIMD multiply-add
   - Reference: Wojciech Muła's SIMD integer parsing

2. **SIMD float parsing**: Use fast_float algorithm
   - Parse mantissa as integer (SIMD)
   - Parse exponent (SIMD)
   - Combine with pre-computed powers of 10

3. **SIMD boolean parsing**: 
   - Compare against "true"/"false"/"1"/"0" masks
   - `vpcmpeqb` + `vpmovmskb` → boolean array

---

### Phase 6F: Lazy/On-Demand Materialization
**Target: 3-5x speedup for projection queries**

**Papers**:
- simdjson On-Demand (Keiser & Lemire, SPE 2024): 70% faster than DOM
- JSON Tiles (Durner et al., SIGMOD 2021): 4x over binary JSONB
- Selective Late Materialization (Liu et al., PVLDB 2025): 14.7% avg speedup

**Technique**: Only decode columns requested by user
```
Current: parse all columns → return requested columns
New:     scan structural index → extract only requested columns
```

For CSV: Use structural index to find field positions for requested columns only.
For JSON: simdjson On-Demand approach — parse values lazily on access.

---

## Implementation Priority

| Phase | Technique | Expected Speedup | Complexity | Dependencies |
|-------|-----------|-----------------|------------|--------------|
| 6A | PCLMULQDQ SIMD | 1.5-2x (boundaries) | Medium | `std::arch` |
| 6B | Two-stage architecture | 2-3x (full read) | High | 6A |
| 6C | Column-first decode | 3-5x (full read) | Very High | 6B |
| 6D | Cache-friendly state machine | 1.3x (quoted) | Low | None |
| 6E | SIMD field parsing | 2-3x (type conversion) | High | 6C |
| 6F | Lazy materialization | 3-5x (projection) | High | 6B |

**Recommended order**: 6D → 6A → 6B → 6C → 6E → 6F

**Total expected improvement**: 10-20x over current (from 239 MiB/s to 2-5 GiB/s)

---

## Key Papers & References

1. **simdjson** — Langdale & Lemire, VLDB Journal 2019 (arXiv:1902.08318)
2. **simdjson On-Demand** — Keiser & Lemire, SPE 2024 (DOI:10.1002/spe.3313)
3. **Speculative Distributed CSV Parsing** — Ge et al., SIGMOD 2019
4. **JSONSki** — Jiang & Zhao, ASPLOS 2022 (par.nsf.gov/servlets/purl/10323318)
5. **Pison** — Jiang et al., PVLDB 2021 (DOI:10.14778/3436905.3436926)
6. **CUDAFastCSV** — Kumaigorodski et al., BTW 2021 (DOI:10.18420/btw2021-01)
7. **JSON Tiles** — Durner et al., SIGMOD 2021 (DOI:10.1145/3448016.3452809)
8. **Selective Late Materialization** — Liu et al., PVLDB 2025
9. **DuckDB CSV Parser 2.0** — PR #10209, #14260 (2023-2024)
10. **arrow-csv2** — friendlymatthew (2025, 5.7x over DataFusion fallback)
11. **fast_float** — Lemire, number parsing library
12. **chunkofcoal SIMD CSV** — Blog post with NEON implementation

---

## Benchmark Targets

| Metric | Current | Phase 6D | Phase 6A | Phase 6B | Phase 6C | Final |
|--------|---------|----------|----------|----------|----------|-------|
| Boundary Detection | 6.46 GiB/s | 6.46 | 10+ | 10+ | 10+ | 10+ |
| CSV Full Read | 239 MiB/s | 310 | 400 | 800 | 2000+ | 3000+ |
| CSV Projection | 236 MiB/s | 300 | 380 | 750 | 1500+ | 5000+ |
| NDJSON Full Read | 566 MiB/s | 566 | 700 | 1200 | 2000+ | 3000+ |
| vs Polars | 15x slower | 12x | 8x | 4x | 1.5x | ~1x |
