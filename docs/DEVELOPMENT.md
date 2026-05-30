# Relay Development Plan

> Phased development roadmap from foundation to production-ready zero-copy data engine.

---

## Overview

```
Phase 0: Foundation                    Weeks 1-4     ████░░░░░░░░░░░░░░░░░░░░
Phase 1: Zero-Copy Python Bridge       Weeks 5-10    ░░░░██████░░░░░░░░░░░░░░
Phase 2: mmap Storage Engine           Weeks 11-16   ░░░░░░░░░░██████░░░░░░░░
Phase 3: Expression Engine             Weeks 17-22   ░░░░░░░░░░░░░░██████░░░░
Phase 4: Execution Engine              Weeks 23-30   ░░░░░░░░░░░░░░░░░░██████
Phase 5: UDF Runtime                   Weeks 31-36   ░░░░░░░░░░░░░░░░░░░░░░██
Phase 6: ML Integration                Weeks 37-42   ░░░░░░░░░░░░░░░░░░░░░░██
Phase 7: Real-Time Streaming           Weeks 43-50   ░░░░░░░░░░░░░░░░░░░░░░██
Phase 8: Lightweight Optimization      Weeks 51-56   ░░░░░░░░░░░░░░░░░░░░░░██
```

**Total: ~56 weeks (14 months)**
**Team: 2-3 engineers**

---

## Testing Strategy (All Phases)

Every phase includes 5 types of testing:

| Type | Tool | What It Measures |
|------|------|-----------------|
| **Unit Tests** | `cargo test` + `pytest` | Correctness of individual functions |
| **E2E Tests** | `pytest` + integration harness | End-to-end workflows |
| **Benchmarks** | `criterion.rs` + `pytest-benchmark` | Throughput, latency, scalability |
| **Memory Trace** | Custom allocator + `valgrind` + `heaptrack` | RSS, allocations, leaks |
| **CPU Trace** | `perf` + `cargo-flamegraph` | CPU utilization, SIMD, cache misses |

### Memory Tracing Infrastructure

```rust
// Custom allocator that tracks all allocations
#[cfg(feature = "mem-trace")]
mod trace_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub struct TracingAllocator;

    unsafe impl GlobalAlloc for TracingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            ALLOC_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            System.dealloc(ptr, layout)
        }
    }
}
```

### CPU Tracing Infrastructure

```bash
# Generate flame graph
perf record -g --call-graph dwarf cargo bench
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# Cache miss rate
perf stat -e cache-misses,cache-references cargo bench

# SIMD utilization
RUSTFLAGS="-C target-feature=+avx2" cargo bench --profile bench
```

### Benchmark Baselines

Every benchmark compares against:
- **Polars** (latest stable)
- **DuckDB** (latest stable)
- **Pandas** (latest stable)
- **DataFusion** (latest stable, where applicable)

---

## Phase 0: Foundation (Weeks 1-4)

**Goal:** Rust project scaffold + CI/CD + basic Arrow integration
**Gaps Addressed:** G6 (Lightweight — foundation for modularity)
**Dependencies:** None

### Deliverables

#### Cargo Workspace

```toml
# Cargo.toml
[workspace]
members = [
    "relay-core",
    "relay-arrow",
    "relay-expr",
    "relay-exec",
    "relay-io",
    "relay-udf",
    "relay-memory",
    "relay-python",
    "relay-ffi",
]
resolver = "2"

[workspace.dependencies]
arrow = "58"
arrow-array = "58"
arrow-buffer = "58"
arrow-schema = "58"
pyo3 = { version = "0.28", features = ["extension-module"] }
thiserror = "2"
tokio = { version = "1", features = ["full"] }
```

#### CI Pipeline (.github/workflows/ci.yml)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --workspace
  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo clippy --workspace -- -D warnings
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo +nightly fmt --check
  miri:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo miri test -p relay-core -p relay-arrow
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release
      - run: pip install target/wheels/*.whl && pytest tests/
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 20+ | Arrow array creation, type conversions, null handling, schema validation |
| E2E | 5 | Create array in Rust → access from Python, round-trip serialization |
| Benchmark | 3 | Array creation throughput, serialization throughput, type conversion speed |
| Memory | Baseline | RSS for empty engine, allocation count for 1M-row array |
| CPU | Baseline | CPU time for array creation, cache miss rate |

### Success Criteria

- [x] `cargo test --workspace` passes
- [x] `cargo clippy --workspace` has zero warnings
- [x] CI pipeline green on push
- [x] `pip install` builds wheel successfully
- [x] `import relay` in Python works without errors
- [x] Miri reports no UB for `relay-core` and `relay-arrow`

### Key Crates

| Crate | Purpose |
|-------|---------|
| `arrow` | Apache Arrow Rust implementation |
| `arrow-array` | Array types |
| `arrow-buffer` | Buffer management |
| `arrow-schema` | Schema definitions |
| `pyo3` | Rust-Python bindings |
| `maturin` | Build tool for Python wheels |
| `thiserror` | Error handling |

---

## Phase 1: Zero-Copy Python Bridge (Weeks 5-10)

**Goal:** True zero-copy data exchange between Rust and Python
**Gaps Addressed:** G1 (Zero-Copy Python Objects)
**Dependencies:** Phase 0

### Deliverables

#### relay-arrow/src/ffi.rs — PyCapsule Interface

```rust
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

#[pyfunction]
fn export_array(py: Python, data: &[u8], dtype: &str) -> PyResult<PyObject> {
    let array = create_arrow_array(data, dtype)?;
    let py_array = PyArray::from_array(array);
    py_array.to_arro3(py)  // zero-copy export via PyCapsule
}
```

#### relay-python/src/protocols.rs — Python Protocol Implementations

```rust
// __arrow_c_array__ for PyCapsule
// __array__ for numpy buffer protocol
// __dataframe__ for DataFrame interchange
// __array_namespace__ for Array API (future)
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 30+ | Each protocol (`__arrow_c_array__`, `__array__`, `__dataframe__`), type mapping |
| Unit | 10+ | Verify NO allocation on zero-copy paths (custom allocator count) |
| E2E | 10 | 10GB dataset Rust→numpy same memory address, round-trip identity |
| E2E | 5 | Export to pyarrow, polars, pandas, numpy — all zero-copy |
| Benchmark | 5 | Export GB/s vs Polars, DuckDB, PyArrow |
| Memory | 3 | Peak RSS during export: Relay vs Polars vs DuckDB |
| CPU | 2 | CPU utilization during export |

### Benchmark Targets

| Benchmark | Target | Baseline (Polars) |
|-----------|--------|-------------------|
| Export 1GB to numpy | ≥ 8 GB/s | ~2 GB/s (with copy) |
| Export 10GB to pyarrow | ≥ 15 GB/s | ~4 GB/s |
| Memory overhead (export 60GB) | < 100 MB | ~37-51 GB |

### Success Criteria

- [ ] `__arrow_c_array__` exports zero-copy to pyarrow, polars, arro3
- [ ] `__array__` exports zero-copy to numpy for all primitive types
- [ ] Custom allocator confirms 0 allocations during zero-copy export
- [ ] 60GB dataset export uses < 100 MB additional memory
- [ ] Export throughput ≥ 2x Polars for same-size datasets

### Key Crates

| Crate | Purpose |
|-------|---------|
| `pyo3-arrow` | Zero-copy Arrow↔Python FFI |
| `arro3-core` | Lightweight Python Arrow objects |
| `rust-numpy` | NumPy array interop |

---

## Phase 2: mmap Storage Engine (Weeks 11-16)

**Goal:** Memory-mapped file access for instant data loading
**Gaps Addressed:** G4 (mmap + Arrow)
**Dependencies:** Phase 0, Phase 1

### Deliverables

#### relay-io/src/mmap.rs — Memory-Mapped Readers

```rust
use memmap2::{Mmap, MmapOptions};

pub struct MmapReader {
    mmap: Arc<Mmap>,
    schema: Schema,
    num_rows: usize,
}

impl MmapReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        // Parse IPC footer, create ArrowArray pointers into mmap
        let schema = parse_ipc_schema(&mmap)?;
        let num_rows = parse_row_count(&mmap)?;
        Ok(Self { mmap: Arc::new(mmap), schema, num_rows })
    }

    pub fn column(&self, idx: usize) -> Result<ArrayRef> {
        // Zero-copy: pointers into mmap'd region
        let offset = self.column_offsets[idx];
        let len = self.column_lengths[idx];
        unsafe { create_array_from_mmap(&self.mmap, offset, len, &self.schema.fields[idx]) }
    }
}
```

#### madvise Strategies

```rust
pub enum AccessPattern {
    Sequential,  // MADV_SEQUENTIAL — aggressive readahead
    Random,      // MADV_RANDOM — disable readahead
    Prefetch,    // MADV_WILLNEED — prefetch pages
    Release,     // MADV_DONTNEED — release after use
}
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 25+ | mmap open/read/close, all Arrow types, madvise strategy selection |
| Unit | 10+ | Lifetime: Arc<Mmap> keeps data alive after file handle dropped |
| E2E | 8 | Open 100GB IPC → read first row in <1ms, verify mmap region |
| E2E | 5 | File deleted while holding reference → graceful error |
| E2E | 5 | Sequential scan vs random access: verify madvise behavior |
| Benchmark | 5 | File open time vs Polars scan_ipc vs DuckDB read_parquet |
| Memory | 3 | RSS when opening 100GB file (should be ~0MB initially) |
| CPU | 2 | Page fault rate: sequential vs random access patterns |

### Benchmark Targets

| Benchmark | Target | Baseline (Polars) |
|-----------|--------|-------------------|
| Open 100GB IPC file | < 5 ms | ~10-50 ms |
| Read first row (100GB) | < 1 ms | ~5-20 ms |
| Sequential scan (10GB) | ≥ 5 GB/s | ~3 GB/s |
| Initial RSS (100GB file) | < 10 MB | ~50 MB |

### Success Criteria

- [ ] Open 100GB uncompressed IPC in < 10ms
- [ ] Zero data copy from file to Arrow array (verified via custom allocator)
- [ ] madvise strategy correctly matches access pattern
- [ ] Initial RSS < 10 MB for 100GB file
- [ ] File open time < Polars `scan_ipc(memory_map=True)`

### Key Crates

| Crate | Purpose |
|-------|---------|
| `memmap2` | Cross-platform mmap |
| `arrow-ipc` | IPC file format parsing |
| `google-zerocopy` | Safe type transmutation from bytes |
| `vortex` (future) | Vortex format reader |

---

## Phase 3: Expression Engine (Weeks 17-22)

**Goal:** Type-safe expression DSL with zero-copy evaluation
**Gaps Addressed:** Foundation for G3 (Unified Execution)
**Dependencies:** Phase 0, Phase 1

### Deliverables

#### relay-expr/src/ast.rs — Expression AST

```rust
pub enum Expr {
    Column(String),
    Literal(ScalarValue),
    BinaryOp { left: Box<Expr>, op: BinaryOperator, right: Box<Expr> },
    UnaryOp { op: UnaryOperator, expr: Box<Expr> },
    Agg { func: AggFunction, expr: Box<Expr> },
    Window { func: WindowFunction, expr: Box<Expr>, partition_by: Vec<Expr>, order_by: Vec<Expr> },
    Cast { expr: Box<Expr>, to: DataType },
    Case { operand: Option<Box<Expr>>, when: Vec<(Expr, Expr)>, else_expr: Option<Box<Expr>> },
}

pub enum BinaryOperator {
    Eq, NotEq, Lt, LtEq, Gt, GtEq,
    And, Or,
    Add, Sub, Mul, Div, Mod,
}
```

#### relay-expr/src/optimizer.rs — Query Optimizer

```rust
pub trait OptimizerRule {
    fn name(&self) -> &str;
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan>;
}

// Rules:
// 1. PredicatePushdown — push filters before joins
// 2. ProjectionPruning — remove unused columns early
// 3. ConstantFolding — evaluate constant expressions at plan time
// 4. CommonSubexpressionElimination — deduplicate repeated expressions
// 5. JoinReordering — reorder joins for optimal hash table sizes
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 40+ | Expression parsing, type inference, all binary/unary/agg operators |
| Unit | 15+ | Optimizer rules: predicate pushdown, projection pruning, CSE |
| Unit | 10+ | Evaluate expression on RecordBatch, verify zero-copy for projections |
| E2E | 10 | Complex chains: filter + project + aggregate + window |
| Benchmark | 5 | Expression evaluation throughput vs Polars expressions |
| Memory | 2 | Peak memory during expression evaluation on 10M rows |
| CPU | 2 | SIMD utilization during vectorized evaluation |

### Success Criteria

- [ ] All Arrow data types supported in expressions
- [ ] Type inference catches type errors at expression build time
- [ ] 5 optimizer rules implemented and tested
- [ ] Expression evaluation throughput ≥ 80% of Polars
- [ ] Projection is zero-copy (verified via custom allocator)

---

## Phase 4: Execution Engine (Weeks 23-30)

**Goal:** Unified eager/lazy/streaming execution
**Gaps Addressed:** G3 (Unified Execution)
**Dependencies:** Phase 0, 1, 2, 3

### Deliverables

#### relay-exec/src/engine.rs — Morsel-Driven Execution

```rust
pub struct MorselExecutor {
    pipeline: Pipeline,
    morsel_size: usize,  // default 2048 rows
    workers: usize,      // default = num_cpus
}

impl MorselExecutor {
    pub fn execute(&self, source: Box<dyn Source>) -> Result<Box<dyn Sink>> {
        let (tx, rx) = crossbeam::channel::bounded(self.workers * 2);

        // Source: produce morsels
        let source_handle = std::thread::spawn(move || {
            while let Some(morsel) = source.next_morsel(self.morsel_size)? {
                tx.send(morsel).unwrap();
            }
        });

        // Pipeline: process morsels in parallel
        let workers: Vec<_> = (0..self.workers).map(|_| {
            let rx = rx.clone();
            let pipeline = self.pipeline.clone();
            std::thread::spawn(move || {
                while let Ok(morsel) = rx.recv() {
                    let result = pipeline.process(morsel)?;
                    // sink.write(result)?;
                }
            })
        }).collect();

        // ... collect results
    }
}
```

#### Execution Modes

```rust
pub enum ExecutionMode {
    Eager,     // Execute immediately, MemorySink
    Lazy,      // Build plan → optimize → execute → MemorySink
    Streaming, // Same plan → FileSink/IteratorSink with backpressure
}

// Key insight: SAME engine, different sink configuration
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 50+ | Each operator: filter, project, sort, hash_join, hash_aggregate, window |
| Unit | 20+ | Optimizer: all 5 rules with various query plans |
| E2E | 22 | TPC-H queries Q1-Q22 (lazy mode) |
| E2E | 10 | Stream 10TB dataset, verify constant memory usage |
| E2E | 10 | Spill-to-disk: hash join with 2x RAM data |
| Benchmark | 22 | TPC-H Q1-Q22 at 1GB, 10GB, 100GB vs Polars, DuckDB, DataFusion |
| Memory | 5 | Peak memory during TPC-H Q1 at different scale factors |
| CPU | 5 | Multi-core utilization: 1, 2, 4, 8, 16 cores |

### Benchmark Targets

| Benchmark | Target | Baseline (Polars) |
|-----------|--------|-------------------|
| TPC-H Q1 (1GB) | ≤ 1.2x Polars time | 1.0x |
| TPC-H Q1 (100GB) | ≤ 1.1x Polars time | 1.0x |
| Stream 10TB (memory) | < 2x RAM | OOM |
| 8-core utilization | ≥ 7.5x 1-core | ~6x |

### Success Criteria

- [ ] TPC-H Q1-Q22 produce correct results (verified against DuckDB)
- [ ] Lazy mode ≤ 1.5x Polars time on TPC-H 1GB
- [ ] Streaming mode: constant memory for 10TB+ data
- [ ] 8-core speedup ≥ 7x (87.5% efficiency)
- [ ] Spill-to-disk works for hash join > RAM

### Key Crates

| Crate | Purpose |
|-------|---------|
| `crossbeam` | Lock-free channels for morsel passing |
| `tokio` | Async runtime for I/O-heavy operations |
| `parking_lot` | Fast mutexes for shared state |
| `ahash` | Fast hashing for hash join/group-by |
| `arrow` | RecordBatch, compute kernels |

---

## Phase 5: UDF Runtime (Weeks 31-36)

**Goal:** Zero-copy UDFs without GIL contention
**Gaps Addressed:** G2 (Zero-Copy UDFs)
**Dependencies:** Phase 0, 1, 3, 4

### Deliverables

#### Tiered UDF Architecture

```rust
pub enum UdfBackend {
    RustPlugin(PluginHandle),     // Zero-copy, no GIL, fastest
    FreeThreadedPython(PyUdf),    // PEP 703, parallel Python
    WasmSandbox(WasmHandle),      // Sandboxed, language-agnostic
    RustExpression(ExprEvaluator),// Simple math, no Python at all
}

// Tier selection:
// 1. If UDF is simple expression → RustExpression
// 2. If UDF is Rust plugin → RustPlugin
// 3. If Python 3.14+ free-threaded → FreeThreadedPython
// 4. If sandboxing needed → WasmSandbox
// 5. Fallback: Python with GIL
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 20+ | Rust plugin registration, execution, type safety |
| Unit | 15+ | Python UDF receives Arrow arrays (verify no copy via custom allocator) |
| Unit | 10+ | WASM UDF: load, execute, unload |
| E2E | 10 | Pipeline with mix: Rust plugins + Python UDFs + WASM |
| E2E | 5 | Python UDF error handling: exceptions propagate correctly |
| Benchmark | 5 | UDF throughput (rows/sec) vs Polars map_elements, Pandas apply |
| Memory | 3 | Memory during UDF: GIL vs free-threaded comparison |
| CPU | 3 | GIL contention profiling: 1, 2, 4, 8 threads |

### Benchmark Targets

| Benchmark | Target | Baseline (Polars) |
|-----------|--------|-------------------|
| Rust plugin UDF (10M rows) | ≥ 100M rows/sec | ~50M (Polars plugins) |
| Python UDF (10M rows) | ≥ 5M rows/sec | ~0.5M (Polars map_elements) |
| WASM UDF (10M rows) | ≥ 30M rows/sec | N/A |

### Success Criteria

- [ ] Rust plugin UDF: 10x faster than Polars `map_elements`
- [ ] Python UDF: 10x faster than Pandas `apply`
- [ ] Zero-copy verified: Python UDF receives Arrow arrays without copy
- [ ] WASM UDF: sandboxed execution, no host memory access
- [ ] Free-threaded Python: 4 threads = 3.5x speedup (87.5% efficiency)

---

## Phase 6: ML Integration (Weeks 37-42)

**Goal:** Seamless integration with scikit-learn, PyTorch, matplotlib
**Gaps Addressed:** G5 (ML Integration)
**Dependencies:** Phase 0, 1, 3, 4

### Deliverables

#### Protocol Implementations

```python
# relay.DataFrame implements:

# 1. __array__ (for numpy)
def __array__(self, dtype=None, copy=False):
    # Zero-copy for primitive columns without nulls
    # Copy only when dtype conversion or nullable→non-nullable

# 2. __dataframe__ (DataFrame Interchange Protocol)
def __dataframe__(self, nan_as_null=False, allow_copy=False):
    # Column-by-column zero-copy access

# 3. __array_namespace__ (Array API)
def __array_namespace__(self, api_version=None):
    # Full Array API support for sklearn dispatch
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 15+ | sklearn fit() with Relay DataFrame (no conversion) |
| Unit | 10+ | PyTorch tensor creation from Relay array (zero-copy) |
| Unit | 10+ | matplotlib plot from Relay DataFrame |
| E2E | 10 | Train sklearn model on 10M rows, verify no intermediate copies |
| E2E | 5 | PyTorch DataLoader from Relay dataset |
| E2E | 5 | Full ML pipeline: load → feature engineer → train → evaluate |
| Benchmark | 5 | ML pipeline throughput: Relay→sklearn vs Polars→sklearn vs Pandas→sklearn |
| Memory | 3 | Peak memory during ML pipeline (target: 1x dataset size) |
| CPU | 2 | CPU during batch inference |

### Success Criteria

- [ ] `sklearn.ensemble.RandomForestClassifier().fit(relay_df, y)` works without `.to_numpy()`
- [ ] `torch.utils.data.DataLoader(RelayDataset(...))` works with zero-copy
- [ ] ML pipeline memory: ≤ 1.5x dataset size (vs Polars 3-4x)
- [ ] `matplotlib.pyplot.plot(relay_df["x"], relay_df["y"])` works directly
- [ ] `__array_namespace__` dispatches for ≥ 30 sklearn estimators

---

## Phase 7: Real-Time Streaming (Weeks 43-50)

**Goal:** Continuous streaming with zero-copy throughout
**Gaps Addressed:** G7 (Real-Time Streaming)
**Dependencies:** Phase 0, 1, 3, 4

### Deliverables

#### Streaming Architecture

```rust
pub trait StreamingSource: Send {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;
    fn watermark(&self) -> Option<Timestamp>;
    fn checkpoint(&self) -> Result<Vec<u8>>;
}

pub trait StreamingSink: Send {
    fn write(&mut self, batch: RecordBatch) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn commit(&mut self) -> Result<()>;  // exactly-once
}

// Window operators
pub enum WindowType {
    Tumbling { size: Duration },
    Sliding { size: Duration, slide: Duration },
    Session { gap: Duration },
}
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 20+ | Window semantics (tumbling, sliding, session) |
| Unit | 15+ | Watermark propagation, late data handling |
| Unit | 10+ | Checkpoint/restore state |
| E2E | 10 | Stream 1M events/sec from Kafka, process, sink to Kafka |
| E2E | 5 | Fail and restore from checkpoint, verify exactly-once |
| E2E | 5 | Arrow Flight source → processing → Arrow Flight sink |
| Benchmark | 5 | Streaming throughput vs Flink, Arroyo |
| Memory | 3 | Constant memory during infinite stream (24h test) |
| CPU | 3 | Backpressure handling: source faster than sink |

### Benchmark Targets

| Benchmark | Target |
|-----------|--------|
| Kafka source throughput | ≥ 500K events/sec |
| End-to-end (source→process→sink) | ≥ 300K events/sec |
| Memory (24h continuous) | < 2x RAM |
| Recovery time (checkpoint restore) | < 5 sec |

### Success Criteria

- [ ] Kafka source/sink working with Arrow RecordBatch
- [ ] Arrow Flight source/sink working
- [ ] 3 window types implemented with correct semantics
- [ ] Exactly-once via checkpoint/restore
- [ ] Constant memory during 1M events/sec continuous stream

---

## Phase 8: Lightweight Optimization (Weeks 51-56)

**Goal:** Minimize binary size and startup time
**Gaps Addressed:** G6 (Lightweight Composable)
**Dependencies:** All previous phases

### Deliverables

#### Feature Flags

```toml
[features]
default = ["ipc", "python"]
full = ["ipc", "parquet", "vortex", "lance", "kafka", "flight", "wasm-udf", "gpu"]
ipc = []
parquet = ["dep:parquet"]
vortex = ["dep:vortex"]
lance = ["dep:lance"]
kafka = ["dep:rdkafka"]
flight = ["dep:arrow-flight"]
wasm-udf = ["dep:extism"]
gpu = ["dep:cuda-driver-sys"]
```

### Testing

| Type | Count | Tests |
|------|-------|-------|
| Unit | 20+ | All feature flag combinations compile |
| Unit | 10+ | Lazy loading: adapters loaded on first use, not import |
| E2E | 5 | `pip install relay-engine` (default features) < 10 MB |
| E2E | 5 | `import relay` + first query < 2 seconds |
| Benchmark | 5 | Package size vs Polars, DuckDB, PyArrow |
| Memory | 3 | Baseline RSS after `import relay` (< 20 MB) |
| CPU | 2 | Import time breakdown |

### Benchmark Targets

| Benchmark | Target | Polars | DuckDB |
|-----------|--------|--------|--------|
| Package size (default) | < 10 MB | ~30 MB | ~50 MB |
| Import time | < 50 ms | ~200 ms | ~100 ms |
| Baseline RSS | < 20 MB | ~50 MB | ~80 MB |
| First query latency | < 500 ms | ~1 sec | ~500 ms |

### Success Criteria

- [ ] Default package < 10 MB (vs Polars ~30 MB)
- [ ] `import relay` < 100 ms
- [ ] All feature flag combinations compile without errors
- [ ] Lazy loading: unused adapters don't consume memory
- [ ] Full build (all features) < 50 MB

---

## CI/CD Pipeline

### GitHub Actions Workflows

```
.github/workflows/
├── ci.yml          # Test, clippy, fmt, miri (every push)
├── bench.yml       # Benchmarks (nightly, main branch)
├── mem-trace.yml   # Memory profiling (weekly)
├── release.yml     # Publish to crates.io + PyPI (tag push)
└── docs.yml        # Build and deploy docs (main branch)
```

### Benchmark Infrastructure

- **criterion.rs** for Rust benchmarks (statistical rigor)
- **pytest-benchmark** for Python benchmarks
- **GitHub Pages** for benchmark history (bencher.dev or custom)
- **Regression detection**: alert if any benchmark degrades > 10%

### Automated Quality Gates

| Gate | Threshold |
|------|-----------|
| Test coverage | ≥ 80% (tarpaulin) |
| Clippy warnings | 0 |
| Miri UB | 0 |
| Benchmark regression | < 10% |
| Memory leak | 0 (heaptrack) |
| Package size | < target |

---

## Summary

| Phase | Duration | Gaps | Key Deliverable |
|-------|----------|------|-----------------|
| 0 | 4 weeks | G6 | Foundation, CI/CD |
| 1 | 6 weeks | G1 | Zero-Copy Python Bridge |
| 2 | 6 weeks | G4 | mmap Storage Engine |
| 3 | 6 weeks | — | Expression Engine |
| 4 | 8 weeks | G3 | Unified Execution |
| 5 | 6 weeks | G2 | UDF Runtime |
| 6 | 6 weeks | G5 | ML Integration |
| 7 | 8 weeks | G7 | Real-Time Streaming |
| 8 | 6 weeks | G6 | Lightweight Optimization |
| **Total** | **56 weeks** | **7/7** | **Complete Engine** |
