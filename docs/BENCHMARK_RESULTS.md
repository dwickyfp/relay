# Comprehensive Benchmark Results: Relay vs All Engines

**Date:** 2026-05-30  
**Test Dataset:** 2M rows, 10 columns, int64 (160 MB)  
**Engines:** Relay, PyArrow, Polars, DuckDB, Pandas, NumPy, Daft

---

## 🏆 Summary Rankings (2M rows)

| Workload | 🥇 1st | 🥈 2nd | 🥉 3rd | Relay Rank |
|----------|--------|--------|--------|------------|
| **File Open** | PyArrow (105µs) | DuckDB (668µs) | Relay (13.76ms) | 3rd |
| **Full Scan** | PyArrow (16.72ms) | Polars (18.66ms) | Relay (19.50ms) | 3rd |
| **Read-Only Scan** | **Relay (5.48ms)** | PyArrow (17.07ms) | Polars (18.26ms) | **🏆 1st** |
| **Projection (2/10 cols)** | Polars (4.59ms) | Pandas (6.49ms) | Daft (8.63ms) | 5th |
| **Filter** | Polars (9.34ms) | Daft (21.25ms) | PyArrow (21.66ms) | 4th |
| **Aggregate (SUM)** | Polars (2.71ms) | DuckDB (3.46ms) | Daft (5.76ms) | 5th |
| **Memory Usage** | **Relay (64B)** | Polars (1.6KB) | PyArrow (1.9KB) | **🏆 1st** |

---

## 📊 Detailed Results

### 1. File Open / Scan Init (2M rows)

```
PyArrow:    105µs    ████████████████████████████████████████████████████████
DuckDB:     668µs    ███████████████████████████████████
Relay:    13.76ms    █
Polars:   18.19ms    
Daft:     26.07ms    
Pandas:   34.63ms    
```

**Analysis:** PyArrow and DuckDB are extremely fast because they only read the IPC footer/metadata. Relay is slower because it parses all batch metadata upfront.

---

### 2. Full Table Scan (2M rows)

```
NumPy mmap:  102µs    ████████████████████████████████████████████████████████
PyArrow:   16.72ms    █████████████████████████████
Polars:    18.66ms    ██████████████████████████
Relay:     19.50ms    █████████████████████████
Daft:      26.91ms    ██████████████████
Pandas:    35.20ms    ██████████████
DuckDB:   875.43ms    ▏
```

**Analysis:** NumPy mmap is fastest (no parsing). Among proper columnar engines, PyArrow leads. Relay is competitive with Polars. DuckDB is surprisingly slow for full scan.

---

### 3. Read-Only Scan (Pre-opened, 2M rows) 🏆

```
Relay:     5.48ms    ████████████████████████████████████████████████████████
PyArrow:  17.07ms    ███████████████████
Polars:   18.26ms    ██████████████████
Pandas:   34.73ms    █████████
```

**Analysis:** When the scan is pre-opened (no metadata parsing overhead), **Relay is 3x faster than PyArrow and 3.3x faster than Polars**. This demonstrates the power of zero-copy mmap access.

**This is Relay's killer feature!** 🚀

---

### 4. Column Projection - 2 of 10 columns (2M rows)

```
Polars:    4.59ms    ████████████████████████████████████████████████████████
Pandas:    6.49ms    ██████████████████████████████████████████████
Daft:      8.63ms    ██████████████████████████████████
PyArrow:  16.60ms    ██████████████████
Relay:    19.77ms    ███████████████
DuckDB:  199.92ms    █
```

**Analysis:** Polars has excellent projection pushdown. Relay is slower because it reads all columns first, then projects in Python. This is an optimization opportunity.

---

### 5. Filter - col_0 < N/2 (2M rows)

```
Polars:    9.34ms    ████████████████████████████████████████████████████████
Daft:     21.25ms    ████████████████████████████
PyArrow:  21.66ms    ██████████████████████████
Relay:    21.99ms    █████████████████████████
Pandas:   44.37ms    ████████████
DuckDB:  442.02ms    █
```

**Analysis:** Polars has optimized filter operations. Relay is competitive with PyArrow and Daft but lacks predicate pushdown.

---

### 6. Aggregate - SUM(col_0) (2M rows)

```
Polars:    2.71ms    ████████████████████████████████████████████████████████
DuckDB:    3.46ms    ████████████████████████████████████████████████
Daft:      5.76ms    █████████████████████████████
PyArrow:  17.36ms    █████████
Relay:    20.54ms    ████████
NumPy:    31.01ms    █████
Pandas:   34.98ms    ████
```

**Analysis:** Polars and DuckDB have highly optimized aggregation kernels. Relay currently reads all data then aggregates in Python, which is slower.

---

### 7. Memory Usage (2M rows) 🏆

```
Relay:        64B    ████████████████████████████████████████████████████████
Polars:      1.6KB   ████████████████████████████████████████████████████
PyArrow:     1.9KB   ██████████████████████████████████████████████████
DuckDB:      4.7KB   ██████████████████████████████████████████
Daft:       24.4KB   ████████████████████████████
Pandas:   120.68MB   ▏
```

**Analysis:** **Relay uses virtually zero heap memory** because data stays in the mmap region (kernel-managed). Pandas loads everything into Python heap (120MB!). This is Relay's second killer feature.

---

## 🎯 Key Findings

### Relay's Strengths 🏆

1. **Read-Only Scan Performance** - 3x faster than PyArrow, 3.3x faster than Polars
2. **Memory Efficiency** - Zero-copy mmap, virtually no heap usage
3. **Full Table Scan** - Competitive with top engines (PyArrow, Polars)

### Relay's Weaknesses ⚠️

1. **File Open Time** - 130x slower than PyArrow (13ms vs 105µs)
2. **Column Projection** - 4x slower than Polars (no projection pushdown)
3. **Aggregation** - 7x slower than Polars (Python-side aggregation)

---

## 🔧 Optimization Opportunities

### Phase 3: Expression Engine (High Priority)

**Goal:** Add pushdown for filter, projection, and aggregation

**Expected Improvements:**
- Projection: 19ms → 5ms (4x faster, match Polars)
- Filter: 22ms → 10ms (2x faster)
- Aggregate: 20ms → 5ms (4x faster)

**Implementation:**
- Build AST for expressions
- Vectorized execution in Rust
- Pushdown to mmap reader

### Phase 4: Metadata Caching (Medium Priority)

**Goal:** Reduce file open time from 13ms to <1ms

**Implementation:**
- Cache `FileReader` instance in `MmapIPCReader`
- Lazy evaluation of batch metadata
- Pre-compute row counts at write time

### Phase 5: Query Planner (Future)

**Goal:** Optimize multi-operation queries

**Implementation:**
- Cost-based optimizer
- Predicate pushdown
- Column pruning

---

## 📈 Performance Comparison Chart

```
Read-Only Scan (lower is better):
Relay    ████████████████████████████████████████████████████████ 5.48ms 🏆
PyArrow  ███████████████████ 17.07ms
Polars   ██████████████████ 18.26ms
Pandas   █████████ 34.73ms

Memory Usage (lower is better):
Relay    ████████████████████████████████████████████████████████ 64B 🏆
Polars   ████████████████████████████████████████████████████ 1.6KB
PyArrow  ██████████████████████████████████████████████████ 1.9KB
Pandas   ▏ 120MB

Full Scan (lower is better):
PyArrow  ████████████████████████████████████████████████████████ 16.72ms
Polars   ██████████████████████████████████████████████████ 18.66ms
Relay    █████████████████████████████████████████████████ 19.50ms
Daft     ███████████████████████████████████ 26.91ms
```

---

## 🎬 Conclusion

**Relay excels at:**
- ✅ Zero-copy data access (3x faster read-only scan)
- ✅ Memory efficiency (zero heap allocation)
- ✅ Full table scans (competitive with PyArrow/Polars)

**Relay needs:**
- ❌ Metadata caching (fix 13ms open time)
- ❌ Expression pushdown (fix projection/filter/aggregate)
- ❌ Query planner (optimize multi-step queries)

**Recommendation:** Relay is production-ready for read-heavy workloads where memory efficiency is critical. For analytical queries with filters/aggregations, implement Phase 3 (Expression Engine) first.

---

## 📋 Raw Data (2M rows)

| Engine | Open | Full Scan | Read-Only | Projection | Filter | Aggregate | Memory |
|--------|------|-----------|-----------|------------|--------|-----------|--------|
| **Relay** | 13.76ms | 19.50ms | **5.48ms** | 19.77ms | 21.99ms | 20.54ms | **64B** |
| PyArrow | **105µs** | **16.72ms** | 17.07ms | 16.60ms | 21.66ms | 17.36ms | 1.9KB |
| Polars | 18.19ms | 18.66ms | 18.26ms | **4.59ms** | **9.34ms** | **2.71ms** | 1.6KB |
| DuckDB | 668µs | 875ms | - | 200ms | 442ms | 3.46ms | 4.7KB |
| Pandas | 34.63ms | 35.20ms | 34.73ms | 6.49ms | 44.37ms | 34.98ms | 120MB |
| Daft | 26.07ms | 26.91ms | - | 8.63ms | 21.25ms | 5.76ms | 24KB |
| NumPy | - | **102µs** | - | - | - | 31.01ms | - |

---

**Benchmark Code:** `benchmarks/comprehensive_bench.py`  
**Run:** `python benchmarks/comprehensive_bench.py`
