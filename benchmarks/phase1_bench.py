#!/usr/bin/env python3
"""
Phase 1 Benchmark: Relay vs PyArrow vs DuckDB vs Polars vs Pandas vs NumPy
==========================================================================
Measures:
  1. Array creation (1M, 10M, 100M elements)
  2. Zero-copy export throughput (→ PyArrow, → NumPy)
  3. Export + Aggregation (sum/mean across all engines)
  4. RecordBatch creation & export
  5. Memory efficiency
"""

import gc
import sys
import time
import statistics
from typing import Callable

# ── Warm up imports ─────────────────────────────────────────────────────
import _relay
import pyarrow as pa
import numpy as np
import duckdb
import polars as pl
import pandas as pd

SIZES = [1_000_000, 10_000_000, 100_000_000]
WARMUP = 2
ITERATIONS = 5


def benchmark(fn: Callable, name: str, iterations: int = ITERATIONS, warmup: int = WARMUP):
    """Run a function multiple times and return stats."""
    for _ in range(warmup):
        fn()
        gc.collect()
    gc.collect()

    times = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    median_ns = statistics.median(times)
    median_ms = median_ns / 1_000_000
    return median_ms


def format_ms(ms: float) -> str:
    if ms < 0.001:
        return f"{ms * 1_000_000:.0f}ns"
    elif ms < 1:
        return f"{ms * 1000:.1f}µs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms / 1000:.2f}s"


def format_throughput(size: int, ms: float) -> str:
    """Format throughput in GB/s."""
    if ms <= 0:
        return "∞"
    bytes_transferred = size * 4  # i32 = 4 bytes
    gb = bytes_transferred / (1024**3)
    sec = ms / 1000
    return f"{gb / sec:.2f} GB/s"


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_row(label: str, values: dict, width: int = 14):
    header = f"{'':>20}"
    for k in values:
        header += f"{k:>{width}}"
    print(header)
    row = f"{label:>20}"
    for v in values.values():
        row += f"{v:>{width}}"
    print(row)
    print()


# ── Benchmark 1: Array Creation ────────────────────────────────────────

def bench_array_creation():
    print_header("1. ARRAY CREATION (i32, median of 5 runs)")

    for n in SIZES:
        data = list(range(n))
        size_str = f"{n:>12,}"

        results = {}

        # Relay
        results["Relay"] = format_ms(benchmark(lambda: _relay.from_i32_list(data), "relay"))

        # PyArrow
        results["PyArrow"] = format_ms(benchmark(lambda: pa.array(data, type=pa.int32()), "pa"))

        # NumPy
        results["NumPy"] = format_ms(benchmark(lambda: np.array(data, dtype=np.int32), "np"))

        # Polars
        results["Polars"] = format_ms(benchmark(lambda: pl.Series(data, dtype=pl.Int32), "pl"))

        # Pandas
        results["Pandas"] = format_ms(benchmark(lambda: pd.array(data, dtype=np.int32), "pd"))

        print(f"  n={size_str}")
        header = f"    {'':>12}"
        for k in results:
            header += f"{k:>12}"
        print(header)
        row = f"    {'':>12}"
        for v in results.values():
            row += f"{v:>12}"
        print(row)
        print()


# ── Benchmark 2: Zero-Copy Export (Relay → PyArrow) ───────────────────

def bench_export_pyarrow():
    print_header("2. ZERO-COPY EXPORT: Relay → PyArrow (median of 5)")

    for n in SIZES:
        data = list(range(n))
        relay_arr = _relay.from_i32_list(data)
        size_str = f"{n:>12,}"

        # Relay → PyArrow (PyCapsule)
        t_relay = format_ms(benchmark(
            lambda: pa.array(relay_arr), "relay->pa"
        ))

        # NumPy → PyArrow (for comparison)
        np_arr = np.array(data, dtype=np.int32)
        t_np = format_ms(benchmark(
            lambda: pa.array(np_arr), "np->pa"
        ))

        # Polars → PyArrow
        pl_arr = pl.Series(data, dtype=pl.Int32)
        t_pl = format_ms(benchmark(
            lambda: pl_arr.to_arrow(), "pl->pa"
        ))

        print(f"  n={size_str}")
        print(f"    Relay→PyArrow:  {t_relay:>12}")
        print(f"    NumPy→PyArrow:  {t_np:>12}")
        print(f"    Polars→PyArrow: {t_pl:>12}")
        print()


# ── Benchmark 3: Export + Aggregation ──────────────────────────────────

def bench_aggregation():
    print_header("3. AGGREGATION: sum() across all engines (median of 5)")

    for n in SIZES:
        data = list(range(n))
        size_str = f"{n:>12,}"

        results = {}

        # Relay → PyArrow → DuckDB
        relay_arr = _relay.from_i32_list(data)
        def relay_duckdb():
            pa_arr = pa.array(relay_arr)
            tbl = pa.table({'v': pa_arr})
            con = duckdb.connect()
            result = con.execute("SELECT sum(v) FROM tbl").fetchone()[0]
            con.close()
            return result
        results["Relay+DuckDB"] = format_ms(benchmark(relay_duckdb, "relay_duckdb"))

        # PyArrow native
        pa_arr = pa.array(data, type=pa.int32())
        results["PyArrow"] = format_ms(benchmark(
            lambda: pa_arr.sum(), "pa_sum"
        ))

        # NumPy
        np_arr = np.array(data, dtype=np.int32)
        results["NumPy"] = format_ms(benchmark(
            lambda: np_arr.sum(), "np_sum"
        ))

        # Polars
        pl_arr = pl.Series(data, dtype=pl.Int32)
        results["Polars"] = format_ms(benchmark(
            lambda: pl_arr.sum(), "pl_sum"
        ))

        # Pandas
        pd_arr = pd.Series(data, dtype=np.int32)
        results["Pandas"] = format_ms(benchmark(
            lambda: pd_arr.sum(), "pd_sum"
        ))

        # DuckDB (from Arrow)
        pa_tbl = pa.table({'v': pa_arr})
        con = duckdb.connect()
        con.register("duck_tbl", pa_tbl)
        def duckdb_sum():
            return con.execute("SELECT sum(v) FROM duck_tbl").fetchone()[0]
        results["DuckDB"] = format_ms(benchmark(duckdb_sum, "duckdb_sum"))
        con.close()

        print(f"  n={size_str}")
        header = f"    {'':>16}"
        for k in results:
            header += f"{k:>16}"
        print(header)
        row = f"    {'':>16}"
        for v in results.values():
            row += f"{v:>16}"
        print(row)
        print()


# ── Benchmark 4: RecordBatch Creation ──────────────────────────────────

def bench_batch_creation():
    print_header("4. RECORD BATCH CREATION (10 cols × 1M rows, median of 5)")

    n = 1_000_000
    ncols = 10
    col_data = [list(range(n)) for _ in range(ncols)]
    col_names = [f"col_{i}" for i in range(ncols)]

    results = {}

    # Relay
    def create_relay():
        arrays = [_relay.from_i32_list(d) for d in col_data]
        return _relay.RelayBatch(col_names, arrays)
    results["Relay"] = format_ms(benchmark(create_relay, "relay_batch"))

    # PyArrow
    def create_pa():
        arrays = [pa.array(d, type=pa.int32()) for d in col_data]
        return pa.RecordBatch.from_arrays(arrays, names=col_names)
    results["PyArrow"] = format_ms(benchmark(create_pa, "pa_batch"))

    # Polars
    def create_pl():
        return pl.DataFrame({f"col_{i}": pl.Series(d, dtype=pl.Int32) for i, d in enumerate(col_data)})
    results["Polars"] = format_ms(benchmark(create_pl, "pl_batch"))

    # Pandas
    def create_pd():
        return pd.DataFrame({f"col_{i}": pd.array(d, dtype=np.int32) for i, d in enumerate(col_data)})
    results["Pandas"] = format_ms(benchmark(create_pd, "pd_batch"))

    header = f"    {'':>12}"
    for k in results:
        header += f"{k:>12}"
    print(header)
    row = f"    {'':>12}"
    for v in results.values():
        row += f"{v:>12}"
    print(row)
    print()


# ── Benchmark 5: Export Throughput ─────────────────────────────────────

def bench_export_throughput():
    print_header("5. EXPORT THROUGHPUT (i32 → PyArrow, median of 5)")

    for n in SIZES:
        size_str = f"{n:>12,}"
        results = {}

        # Relay FFI export (Rust-only, no Python capsule overhead)
        t_ns = _relay.benchmark_export_throughput(n)
        results["Relay FFI"] = format_ms(t_ns / 1_000_000)

        # Relay → PyArrow (full path)
        relay_arr = _relay.from_i32_list(list(range(n)))
        results["Relay→PA"] = format_ms(benchmark(
            lambda: pa.array(relay_arr), "relay_pa_full"
        ))

        # NumPy → PyArrow
        np_arr = np.array(list(range(n)), dtype=np.int32)
        results["NumPy→PA"] = format_ms(benchmark(
            lambda: pa.array(np_arr), "np_pa_full"
        ))

        # Polars → PyArrow
        pl_arr = pl.Series(list(range(n)), dtype=pl.Int32)
        results["Polars→PA"] = format_ms(benchmark(
            lambda: pl_arr.to_arrow(), "pl_pa_full"
        ))

        print(f"  n={size_str}")
        header = f"    {'':>12}"
        for k in results:
            header += f"{k:>14}"
        print(header)
        row = f"    {'':>12}"
        for v in results.values():
            row += f"{v:>14}"
        print(row)

        # Calculate Relay FFI throughput
        relay_ms = t_ns / 1_000_000
        if relay_ms > 0:
            gb = (n * 4) / (1024**3)
            throughput = gb / (relay_ms / 1000)
            print(f"    Relay FFI throughput: {throughput:.2f} GB/s")
        print()


# ── Benchmark 6: Memory Efficiency ─────────────────────────────────────

def bench_memory():
    print_header("6. MEMORY EFFICIENCY (n=10M i32)")

    n = 10_000_000
    data = list(range(n))

    # Relay
    relay_arr = _relay.from_i32_list(data)
    relay_mem = relay_arr.memory_size

    # PyArrow (no direct memory_size, use nbytes)
    pa_arr = pa.array(data, type=pa.int32())
    pa_mem = pa_arr.nbytes

    # NumPy
    np_arr = np.array(data, dtype=np.int32)
    np_mem = np_arr.nbytes

    # Polars
    pl_arr = pl.Series(data, dtype=pl.Int32)
    pl_mem = pl_arr.estimated_size()

    # Pandas
    pd_arr = pd.Series(data, dtype=np.int32)
    pd_mem = pd_arr.memory_usage(deep=True)

    print(f"    {'Engine':>12} {'Memory':>12} {'Per-element':>14}")
    print(f"    {'Relay':>12} {relay_mem:>10,} B {relay_mem/n:>10.1f} B/elem")
    print(f"    {'PyArrow':>12} {pa_mem:>10,} B {pa_mem/n:>10.1f} B/elem")
    print(f"    {'NumPy':>12} {np_mem:>10,} B {np_mem/n:>10.1f} B/elem")
    print(f"    {'Polars':>12} {pl_mem:>10,} B {pl_mem/n:>10.1f} B/elem")
    print(f"    {'Pandas':>12} {pd_mem:>10,} B {pd_mem/n:>10.1f} B/elem")
    print()


# ── Run all benchmarks ─────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'#'*70}")
    print(f"  RELAY PHASE 1 BENCHMARK")
    print(f"  Relay v{_relay.version()} | Arrow {pa.__version__} | DuckDB {duckdb.__version__}")
    print(f"  Polars {pl.__version__} | Pandas {pd.__version__} | NumPy {np.__version__}")
    print(f"{'#'*70}")

    bench_array_creation()
    bench_export_pyarrow()
    bench_aggregation()
    bench_batch_creation()
    bench_export_throughput()
    bench_memory()

    print(f"\n{'#'*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'#'*70}\n")
