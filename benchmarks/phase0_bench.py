"""
Relay Benchmark Suite — Phase 0
Compares Relay array creation with Polars, DuckDB, Pandas, and PyArrow.
"""

import time
import sys
import os

# Try to import all engines
engines = {}

try:
    import _relay
    engines["relay"] = _relay
    print(f"✓ Relay v{_relay.version()}")
except ImportError as e:
    print(f"✗ Relay: {e}")

try:
    import polars as pl
    engines["polars"] = pl
    print(f"✓ Polars v{pl.__version__}")
except ImportError as e:
    print(f"✗ Polars: {e}")

try:
    import pandas as pd
    engines["pandas"] = pd
    print(f"✓ Pandas v{pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import duckdb
    engines["duckdb"] = duckdb
    print(f"✓ DuckDB v{duckdb.__version__}")
except ImportError as e:
    print(f"✗ DuckDB: {e}")

try:
    import pyarrow as pa
    engines["pyarrow"] = pa
    print(f"✓ PyArrow v{pa.__version__}")
except ImportError as e:
    print(f"✗ PyArrow: {e}")

try:
    import numpy as np
    engines["numpy"] = np
    print(f"✓ NumPy v{np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")


def benchmark_create_i32_array(n: int, warmup: int = 3, repeats: int = 10):
    """Benchmark creating an i32 array of n elements."""
    results = {}
    
    # Relay (Rust)
    if "relay" in engines:
        # Warmup
        for _ in range(warmup):
            _relay.benchmark_create_array(n)
        # Measure
        times = []
        for _ in range(repeats):
            ns = _relay.benchmark_create_array(n)
            times.append(ns / 1e6)  # ns → ms
        results["relay"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # Polars
    if "polars" in engines:
        import time
        for _ in range(warmup):
            pl.Series(list(range(n)), dtype=pl.Int32)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pl.Series(list(range(n)), dtype=pl.Int32)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["polars"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # Pandas
    if "pandas" in engines:
        for _ in range(warmup):
            pd.Series(range(n), dtype="int32")
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pd.Series(range(n), dtype="int32")
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["pandas"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # PyArrow
    if "pyarrow" in engines:
        for _ in range(warmup):
            pa.array(list(range(n)), type=pa.int32())
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pa.array(list(range(n)), type=pa.int32())
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["pyarrow"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # NumPy
    if "numpy" in engines:
        for _ in range(warmup):
            np.arange(n, dtype=np.int32)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            np.arange(n, dtype=np.int32)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["numpy"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    return results


def benchmark_create_f64_array(n: int, warmup: int = 3, repeats: int = 10):
    """Benchmark creating a f64 array of n elements."""
    results = {}
    data = [float(i) for i in range(n)]
    
    # Relay (Rust)
    if "relay" in engines:
        for _ in range(warmup):
            _relay.from_f64_list(data)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _relay.from_f64_list(data)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["relay"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # Polars
    if "polars" in engines:
        for _ in range(warmup):
            pl.Series(data, dtype=pl.Float64)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pl.Series(data, dtype=pl.Float64)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["polars"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # Pandas
    if "pandas" in engines:
        for _ in range(warmup):
            pd.Series(data, dtype="float64")
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pd.Series(data, dtype="float64")
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["pandas"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    # NumPy
    if "numpy" in engines:
        for _ in range(warmup):
            np.array(data, dtype=np.float64)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            np.array(data, dtype=np.float64)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["numpy"] = {"mean_ms": sum(times)/len(times), "min_ms": min(times)}
    
    return results


def format_table(results: dict, n: int) -> str:
    """Format benchmark results as a table."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Benchmark: Create {n:,} element array")
    lines.append(f"{'='*60}")
    lines.append(f"  {'Engine':<12} {'Mean (ms)':>12} {'Min (ms)':>12} {'vs Relay':>12}")
    lines.append(f"  {'-'*48}")
    
    relay_mean = results.get("relay", {}).get("mean_ms", float("inf"))
    for engine, data in sorted(results.items(), key=lambda x: x[1]["mean_ms"]):
        ratio = f"{data['mean_ms']/relay_mean:.2f}x" if relay_mean < float("inf") else "-"
        lines.append(f"  {engine:<12} {data['mean_ms']:>12.3f} {data['min_ms']:>12.3f} {ratio:>12}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"\nRelay Benchmark Suite — Phase 0")
    print(f"Python: {sys.version}")
    print(f"Engines loaded: {list(engines.keys())}")
    
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        # i32 benchmark
        results = benchmark_create_i32_array(n)
        print(format_table(results, n))
        
        # f64 benchmark (only for smaller sizes)
        if n <= 100_000:
            results = benchmark_create_f64_array(n)
            print(format_table(results, n))
    
    print("\n✅ Benchmarks complete!")
