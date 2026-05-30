# Relay Architecture

> Zero-Copy Data Engine for Python, Powered by Rust & Apache Arrow

---

## 1. Executive Summary

**Relay** is a Rust-based data processing engine designed from the ground up around one principle: **zero-copy data flow**. It bridges Rust's memory safety guarantees with Python's data science ecosystem through Apache Arrow's columnar memory format, creating a data engine where data is never unnecessarily copied — from disk to memory to computation to Python objects and back.

### The 7 Gaps Relay Solves

| # | Gap | Problem | Relay Solution |
|---|-----|---------|---------------|
| G1 | Zero-Copy Python Objects | All engines copy at Rust→Python boundary | Arrow PyCapsule Interface + Buffer Protocol |
| G2 | Zero-Copy UDFs | Python UDFs kill performance (GIL) | Tiered: Rust plugins → Free-threaded Python → WASM |
| G3 | Unified Execution | Polars has 3 separate code paths | Single push-based engine, configurable sinks |
| G4 | mmap + Arrow | No modern Rust engine deeply integrates mmap | mmap-first storage with Arrow zero-copy pointers |
| G5 | ML Integration | `to_pandas()`/`to_numpy()` always copies | `__array__`, `__dataframe__`, `__array_namespace__` |
| G6 | Lightweight | DataFusion=heavy, Polars=monolithic | Modular crates, feature flags, <10MB package |
| G7 | Real-Time Streaming | All engines are batch-oriented | Continuous streaming with Arrow Flight + Kafka |

---

## 2. Design Philosophy

### 2.1 Zero-Copy as Foundation, Not Feature

Most data engines treat zero-copy as an optimization. Relay treats it as an **invariant**. Every layer is designed so that data movement between components uses pointer passing, not memory allocation.

```
Traditional Engine:                    Relay:
  Disk → Copy → Memory → Copy → Rust → Copy → Python
  3 copies, 3x memory overhead        Disk → mmap → Rust → PyCapsule → Python
                                       0 copies, 1x memory overhead
```

### 2.2 Arrow-Native Memory Model

All data in Relay lives as Apache Arrow columnar arrays:
- **Contiguous same-type columns** → SIMD-friendly sequential access
- **64-byte alignment** → aligned SIMD loads, cache-line friendly
- **Validity bitmaps** → branchless null checks via bitmask operations
- **Offset buffers** → O(1) slicing without copying
- **Immutable by design** → safe concurrent reads, no data races

### 2.3 Rust Safety Guarantees

| Bug Class | C/C++ | Rust |
|-----------|-------|------|
| Use-after-free | Common (CVEs in Arrow C++) | Compile-time: ownership + drop |
| Double-free | Common in ref counting | Compile-time: single owner |
| Buffer overflow | Common with raw pointers | Runtime: bounds checking |
| Data races | Common in multi-threaded | Compile-time: borrow checker |
| Memory leaks | Common with complex ownership | Compile-time: RAII + drop |

### 2.4 Python Ergonomics Without Python Overhead

Relay's Python bindings use PyO3 with full type hints and docstrings, but execution happens entirely in Rust with the GIL released. Python is the interface, Rust is the engine.

---

## 3. System Architecture

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: PYTHON INTERFACE                                           │
│ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌───────────────────────────┐│
│ │ pandas   │ │ polars   │ │ pyarrow  │ │ native relay API          ││
│ │ DataFrame│ │ DataFrame│ │ Table    │ │ DataFrame + SQL + stream  ││
│ └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────────────┬───────────┘│
│      └─────────┬───┴─────────┬─┘                     │            │
│     PyCapsule Interface    Buffer Protocol     PyO3 bindings      │
│     (zero-copy FFI)        (numpy compat)     (API layer)         │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2: ARROW INTERCHANGE                                          │
│ ┌──────────┐ ┌──────────────┐ ┌────────────────┐                  │
│ │ arrow-rs │ │ C Data I/F   │ │ PyCapsule I/F  │                  │
│ │ (core)   │ │ FFI_Arrow*   │ │ (export/import)│                  │
│ └──────────┘ └──────────────┘ └────────────────┘                  │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3: UDF RUNTIME                                                │
│ ┌───────────────┐ ┌──────────────────┐ ┌─────────────────────────┐│
│ │ Rust Plugins  │ │ Free-threaded Py │ │ WASM Sandbox            ││
│ │ (zero-copy)   │ │ (PEP 703)        │ │ (extism/wasmtime)       ││
│ │ Default tier  │ │ User convenience │ │ Security-required       ││
│ └───────────────┘ └──────────────────┘ └─────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4: EXECUTION ENGINE                                           │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ Query Planner & Optimizer                                      │ │
│ │ Predicate Pushdown │ Projection Pruning │ CSE │ Join Reorder  │ │
│ └────────────────────────────┬───────────────────────────────────┘ │
│ ┌────────────────────────────┼───────────────────────────────────┐ │
│ │ Streaming Execution Engine │ (Push-based morsel-driven)        │ │
│ │ ┌──────────┐ ┌────────────┴────────┐ ┌─────────────────────┐  │ │
│ │ │ Source   │→│ Pipeline             │→│ Sink                │  │ │
│ │ │(scan/IPC)│ │ filter→project→join  │ │ Memory|File|Iterator│  │ │
│ │ └──────────┘ └─────────────────────┘ └─────────────────────┘  │ │
│ └────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│ Eager:  Plan → Execute → Memory sink                               │
│ Lazy:   Plan → Optimize → Execute → Memory sink                    │
│ Stream: Plan → Optimize → Execute → File/Iterator sink             │
│         ──────────────────────────────────────────────             │
│         SAME ENGINE, different sink configuration                   │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 5: MEMORY MANAGEMENT                                          │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────┐│
│ │ mmap Manager │ │ Buffer Pool  │ │ Spill Manager               ││
│ │ (source data)│ │ (intermediates)│ │ (external sort/agg)        ││
│ │ Arc<Mmap>    │ │ 256KB blocks │ │ Temp files                  ││
│ │ madvise      │ │ LRU eviction │ │ Partitioned hash joins      ││
│ └──────────────┘ └──────────────┘ └──────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6: STORAGE ADAPTERS                                           │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│ │ Arrow IPC│ │ Vortex   │ │ Parquet  │ │ Lance    │ │ Arrow    ││
│ │ (hot)    │ │ (warm)   │ │ (compat) │ │ (ML/     │ │ Flight   ││
│ │ mmap ZC  │ │ compress │ │ decode   │ │ vectors) │ │(stream)  ││
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
├─────────────────────────────────────────────────────────────────────┤
│ Layer 7: SAFETY LAYER                                               │
│ ┌───────────────┐ ┌──────────────┐ ┌─────────────────────────────┐│
│ │ Ownership &   │ │ Borrow       │ │ Arc<Buffer> for shared      ││
│ │ Drop guards   │ │ checker      │ │ column ownership            ││
│ └───────────────┘ └──────────────┘ └─────────────────────────────┘│
│ ┌───────────────┐ ┌──────────────┐ ┌─────────────────────────────┐│
│ │ Bounds-checked│ │ Send+Sync    │ │ Type-safe schema at         ││
│ │ access        │ │ for threads  │ │ compile time                ││
│ └───────────────┘ └──────────────┘ └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

### 4.1 Read Path (Disk → Python)

```
1. Python:  relay.scan("data.ipc")
   ────────
   │ PyO3 binding (no data copy)
   ▼
2. Query Planner:  predicate + projection pushdown
   ────────────────
   │ optimized plan
   ▼
3. Storage Adapter:  mmap file → &[u8]
   ────────────────
   │ madvise(MADV_SEQUENTIAL) for scan, MADV_RANDOM for point lookups
   ▼
4. Decode Column:  only needed cols, zero-copy for uncompressed IPC
   ────────────────
   │ Arrow ArrayData (pointers into mmap'd region)
   ▼
5. Compute Kernel:  SIMD filter/project/aggregate
   ────────────────
   │ result Arrow arrays (new allocations only for computed columns)
   ▼
6. Export to Python:  via C Data Interface / PyCapsule
   ────────────────
   │ zero-copy FFI: buffer pointers transferred, not data
   ▼
7. Python DataFrame:  views into Rust memory
   ────────────────
   │ Arc<Mmap> keeps mmap alive via PyCapsule destructor
   ▼
8. User:  arr = df.to_numpy()  → zero-copy for primitive arrays
```

**Memory at each stage:**
```
Stage 1-3: mmap region only (OS manages paging)
Stage 4:   + computed column buffers (e.g., filter result = bitmask)
Stage 5:   + intermediate hash tables for joins (spillable)
Stage 6-8: Python holds references, mmap stays alive
```

### 4.2 Write Path (Python → Disk)

```
1. Python:  df.to_ipc("output.ipc")
   ────────
   │ Arrow PyCapsule import (zero-copy from Python)
   ▼
2. Validate Schema:  type check at Rust level
   ▼
3. Encode:  Arrow IPC serialization (flatbuffers metadata + raw buffers)
   ▼
4. Write:  writev() for multi-buffer, O_DIRECT for large files
   ▼
5. Sync:  fsync() for durability guarantee
```

---

## 5. Zero-Copy Guarantee Matrix

| Operation | Zero-Copy? | Mechanism |
|-----------|:---:|-----------|
| Scan IPC (uncompressed) | ✅ | mmap → ArrowArray pointers |
| Scan Parquet | ❌ | Must decode (RLE, dictionary, compression) |
| Scan Vortex | ⚠️ | Zero-copy for Arrow-compatible encodings |
| Filter | ✅ | Returns boolean mask, original data untouched |
| Project (select columns) | ✅ | Returns slice of columns, no copy |
| Sort | ❌ | Requires materializing sorted arrays |
| Hash Join | ❌ | Build hash table, probe with copy |
| Aggregate (sum, mean) | ✅ | Single-pass over columns |
| Export to numpy | ✅* | Buffer protocol for contiguous primitives without nulls |
| Export to pandas | ✅* | ArrowDtype backend keeps Arrow memory |
| Export to torch | ✅* | DLPack protocol |
| Python UDF (Rust plugin) | ✅ | Receives Arrow arrays by reference |
| Python UDF (Python func) | ⚠️ | Free-threaded: no GIL, but Python objects created |

*For types that numpy/pandas/torch can represent natively (int, float, fixed-size strings)

---

## 6. Key Technologies

### 6.1 Core Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Memory Format | Apache Arrow (arrow-rs) | Columnar, zero-copy, SIMD-friendly |
| Python Bridge | PyO3 + pyo3-arrow | Zero-copy FFI via PyCapsule Interface |
| Build Tool | maturin | Rust→Python wheel building |
| Async Runtime | tokio | I/O, channels, timers |
| Serialization | Arrow IPC / FlatBuffers | Zero-copy interchange |
| mmap | memmap2 | Memory-mapped file I/O |
| Safe Transmute | google/zerocopy | Compile-time type punning |

### 6.2 Integration Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Storage (warm) | Vortex | Compressed columnar, 100x faster random access vs Parquet |
| Storage (ML) | Lance | Vector embeddings, versioned data |
| Streaming | Arrow Flight (gRPC) | Zero-copy data transport |
| Kafka | rdkafka | Streaming source/sink |
| WASM UDF | extism + wasmtime | Sandboxed UDF execution |
| GPU | Arrow Device Interface | GPU-accelerated compute |

---

## 7. Supporting Academic Papers

### Zero-Copy Serialization
1. Wolnikowski et al. (2021) — "Zerializer: Towards Zero-Copy Serialization" (HotOS)
2. Raghavan et al. (2021) — "Breakfast of Champions: Zero-Copy with NIC Scatter-Gather" (HotNets)
3. Raghavan et al. (2023) — "Cornflakes: Zero-Copy for Microsecond Networking" (SOSP)
4. Liu et al. (2026) — "zBuffer: Metadata-Free Serialization" (ASPLOS)
5. Chen et al. (2024) — "Lite²: Schemaless Zero-Copy Format" (MDPI)
6. Multiple (2024) — "Streaming Tech & Serialization: Empirical Analysis" (arXiv:2407.13494)

### Arrow & Columnar Formats
7. McKinney et al. (2016) — "Apache Arrow: Cross-Language In-Memory Data" (Apache)
8. Liu et al. (2023) — "Deep Dive: Open Formats for Analytical DBMSs" (VLDB)
9. Li et al. (2021) — "Mainlining Databases on Arrow" (PVLDB)
10. Chakraborty et al. (2024) — "Thallus: RDMA Columnar Transport" (arXiv)
11. Groet et al. (2024) — "Zero-Copy Cluster Shared Memory via Arrow IPC" (arXiv)

### Rust Data Frameworks
12. Lamb et al. (2024) — "DataFusion: Fast, Embeddable, Modular Query Engine" (SIGMOD)
13. Multiple (2024) — "Evaluation of Dataframe Libraries" (arXiv:2312.11122)
14. Nahrstedt et al. (2024) — "Energy Usage: Pandas vs Polars" (EASE)

### Rust-Python Bridges
15. Lafrance et al. (2023) — "Extending Rust: Zero-Copy Communication" (PPoPP)
16. Barron (2023+) — "pyo3-arrow: Zero-Copy FFI" (crates.io)
17. Barron (2023+) — "arro3: Minimal Python Arrow" (GitHub)
18. Riba et al. (2025) — "Kornia-rs: Rust 3D Vision" (arXiv)
19. Norris (2025) — "rvLLM: Rust LLM Inference" (Web)

### Zero-Copy Systems
20. Tagliabue et al. (2024) — "Bauplan: Zero-Copy FaaS" (WoSC10)
21. **Dai et al. (2025) — "Zerrow: True Zero-Copy Arrow Pipelines" (arXiv:2504.06151)** ← Most Important
22. Su, Zhang (2026) — "Zero-Copy Lock-Free Edge Streaming" (Information Sciences)

### Memory Safety
23. Santoso et al. (2023) — "Rust's Memory Safety: Evaluation" (Procedia CS)
24. Bugden, Alahmar (2022) — "Rust: Safety and Performance" (arXiv:2206.05503)
25. Qin et al. (2024) — "Real-World Safety Issues in Rust" (IEEE TSE)
26. Zhang et al. (2024) — "Beyond Memory Safety: Bugs in Rust" (IEEE SERA)
27. Google (2024+) — "google/zerocopy" (GitHub)

---

## 8. Comparison with Existing Engines

| Engine | Rust | Zero-Copy Python | UDFs | Execution | mmap | ML | Streaming | Gaps Solved |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Relay** | ✅ | ✅ | ✅ | ✅ Unified | ✅ | ✅ | ✅ | **7/7** |
| Polars | ✅ | ⚠️ | ⚠️ | ⚠️ Split | ❌ | ❌ | ✅ | 3/7 |
| DuckDB | ❌C++ | ❌ | ❌ | ✅ SQL | ⚠️ | ❌ | ✅ | 3/7 |
| DataFusion | ✅ | ⚠️ | ⚠️ | ✅ SQL | ❌ | ❌ | ⚠️ | 2/7 |
| Daft | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 4/7 |
| Lance | ✅ | ⚠️ | ❌ | ❌ | ✅ | ✅ | ❌ | 3/7 |
| Vortex | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 2/7 |
| Velox | ❌C++ | ❌ | ⚠️ | ✅ | ❌ | ⚠️ | ✅ | 3/7 |

---

## 9. Crate Architecture

```
relay/
├── relay-core/           # Types, schema, error handling, shared utilities
│   └── deps: arrow, thiserror
├── relay-arrow/          # Arrow integration, FFI, mmap
│   └── deps: arrow, pyo3-arrow, memmap2, google-zerocopy
├── relay-expr/           # Expression DSL, type inference, optimization
│   └── deps: arrow, relay-core
├── relay-exec/           # Execution engine (eager/lazy/streaming)
│   └── deps: arrow, tokio, crossbeam, relay-core, relay-expr
├── relay-io/             # Storage adapters (IPC, Parquet, Vortex, Lance)
│   └── deps: arrow, memmap2, parquet, vortex*, lance*
├── relay-udf/            # UDF runtime (Rust plugins, Python, WASM)
│   └── deps: pyo3, extism, wasmtime, relay-core
├── relay-memory/         # Buffer pool, mmap manager, spill manager
│   └── deps: memmap2, parking_lot, relay-core
├── relay-python/         # PyO3 bindings, Python API
│   └── deps: pyo3, pyo3-arrow, relay-core, relay-exec, relay-io
└── relay-ffi/            # C Data Interface export/import
    └── deps: arrow::ffi, relay-core
```

---

## 10. Future Vision

### Phase 9+: GPU Acceleration
- Arrow Device Interface for GPU-resident arrays
- CUDA kernels for filter/project/aggregate
- Zero-copy GPU→CPU via unified memory (Apple Silicon) or pinned memory

### Phase 10+: Distributed Execution
- Arrow Flight RPC for cross-node data transport
- Partitioned parallel execution
- Distributed shuffle with zero-copy transfer

### Phase 11+: ML Pipeline Native
- Automatic feature engineering
- Model serving integration (ONNX Runtime)
- Training data versioning (Lance-style)
