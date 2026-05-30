#!/usr/bin/env python3
"""
Phase 2 Benchmark: mmap Storage Engine (OPTIMIZED v2)
======================================================
Measures BOTH open+scan and read-only (after pre-opened scan).
Fair comparison: Polars lazy scan vs Relay mmap reader.
"""
import gc
import os
import time
import statistics
import tempfile
import tracemalloc

import _relay
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import polars as pl
import duckdb

WARMUP = 2
ITERATIONS = 5
DATA_DIR = tempfile.mkdtemp(prefix="relay_bench_")


def benchmark(fn, warmup=WARMUP, iterations=ITERATIONS):
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
    return statistics.median(times) / 1_000_000


def fmt(ms):
    if ms < 0.001:
        return f"{ms*1e6:.0f}ns"
    elif ms < 1:
        return f"{ms*1000:.1f}µs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


def create_files(prefix, n, ncols=10):
    names = [f"col_{i}" for i in range(ncols)]
    arrays = [pa.array(range(n), type=pa.int64()) for _ in range(ncols)]
    table = pa.table(dict(zip(names, arrays)))
    ipc_p = f"{prefix}.ipc"
    pq_p = f"{prefix}.parquet"
    with ipc.new_file(ipc_p, table.schema) as w:
        w.write_table(table)
    pq.write_table(table, pq_p, row_group_size=min(n, 100_000))
    return ipc_p, pq_p


SIZES = [50_000, 500_000, 2_000_000]


def bench_open():
    print("\n" + "=" * 70)
    print("  1. FILE OPEN TIME")
    print("=" * 70)
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"open_{n}"), n)
        t_relay = benchmark(lambda: _relay.scan(ipc_p))
        t_pa = benchmark(lambda: ipc.open_file(ipc_p))
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).head(n).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT count(*) FROM '{pq_p}'").fetchone())
        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>10}: {fmt(t_relay):>10}")
        print(f"    {'PyArrow':>10}: {fmt(t_pa):>10}")
        print(f"    {'Polars':>10}: {fmt(t_pl):>10}")
        print(f"    {'DuckDB':>10}: {fmt(t_db):>10}")
        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_scan():
    print("\n" + "=" * 70)
    print("  2. FULL TABLE SCAN (open + read)")
    print("=" * 70)
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"scan_{n}"), n)
        def relay_s():
            sr = _relay.scan(ipc_p)
            return sr.read_all()
        t_relay = benchmark(relay_s)
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT * FROM '{pq_p}'").to_arrow_table())
        def pa_s():
            r = ipc.open_file(ipc_p)
            bs = [r.get_batch(i) for i in range(r.num_record_batches)]
            return pa.RecordBatchReader.from_batches(r.schema, bs).read_all()
        t_pa = benchmark(pa_s)
        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>10}: {fmt(t_relay):>10}")
        print(f"    {'PyArrow':>10}: {fmt(t_pa):>10}")
        print(f"    {'Polars':>10}: {fmt(t_pl):>10}")
        print(f"    {'DuckDB':>10}: {fmt(t_db):>10}")
        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_scan_read_only():
    """Benchmark: scan() once, then measure read_all() only (no open overhead)."""
    print("\n" + "=" * 70)
    print("  3. READ-ONLY (pre-opened scan, read_all)")
    print("=" * 70)
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"ro_{n}"), n)
        sr = _relay.scan(ipc_p)  # pre-open
        t_relay = benchmark(lambda: sr.read_all())
        def pl_ro():
            lf = pl.scan_parquet(pq_p)  # pre-open lazy frame
            return lf.collect()
        t_pl = benchmark(pl_ro)
        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>10}: {fmt(t_relay):>10}")
        print(f"    {'Polars':>10}: {fmt(t_pl):>10}")
        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_projection():
    print("\n" + "=" * 70)
    print("  4. COLUMN PROJECTION (2 of 10 cols, open + read)")
    print("=" * 70)
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"proj_{n}"), n)
        cols = ["col_0", "col_1"]
        def relay_p():
            sr = _relay.scan(ipc_p)
            return sr.read_columns(cols)
        t_relay = benchmark(relay_p)
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).select(cols).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT col_0, col_1 FROM '{pq_p}'").to_arrow_table())
        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>10}: {fmt(t_relay):>10}")
        print(f"    {'Polars':>10}: {fmt(t_pl):>10}")
        print(f"    {'DuckDB':>10}: {fmt(t_db):>10}")
        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_projection_read_only():
    """Projection benchmark: scan() once, measure read_columns() only."""
    print("\n" + "=" * 70)
    print("  5. PROJECTION READ-ONLY (pre-opened, 2 of 10 cols)")
    print("=" * 70)
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"projro_{n}"), n)
        cols = ["col_0", "col_1"]
        sr = _relay.scan(ipc_p)  # pre-open
        t_relay = benchmark(lambda: sr.read_columns(cols))
        def pl_ro():
            lf = pl.scan_parquet(pq_p)
            return lf.select(cols).collect()
        t_pl = benchmark(pl_ro)
        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>10}: {fmt(t_relay):>10}")
        print(f"    {'Polars':>10}: {fmt(t_pl):>10}")
        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_memory():
    print("\n" + "=" * 70)
    print("  6. MEMORY USAGE (2M rows, 10 cols, int64)")
    print("=" * 70)
    n = 2_000_000
    ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"mem_{n}"), n)
    expected = n * 10 * 8

    tracemalloc.start()
    sr = _relay.scan(ipc_p)
    _, r_open = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    rb = sr.read_all()
    _, r_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    df = pl.scan_parquet(pq_p).collect()
    _, pl_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    tbl = duckdb.query(f"SELECT * FROM '{pq_p}'").to_arrow_table()
    _, db_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Expected: {expected:,} B ({expected/1e6:.0f} MB)")
    print(f"  {'Engine':>14} {'Peak Memory':>14} {'Ratio':>8}")
    print(f"  {'Relay (open)':>14} {r_open:>11,} B {'N/A':>8}")
    print(f"  {'Relay (full)':>14} {r_full:>11,} B {r_full/expected:>7.2f}x")
    print(f"  {'Polars':>14} {pl_peak:>11,} B {pl_peak/expected:>7.2f}x")
    print(f"  {'DuckDB':>14} {db_peak:>11,} B {db_peak/expected:>7.2f}x")

    ipc_sz = os.path.getsize(ipc_p)
    pq_sz = os.path.getsize(pq_p)
    print(f"\n  IPC file:     {ipc_sz:>11,} B ({ipc_sz/1e6:.0f} MB)")
    print(f"  Parquet file: {pq_sz:>11,} B ({pq_sz/1e6:.0f} MB)")
    os.unlink(ipc_p)
    os.unlink(pq_p)


if __name__ == "__main__":
    print(f"\n{'#'*70}")
    print(f"  RELAY PHASE 2 BENCHMARK — mmap Storage Engine (OPTIMIZED v2)")
    print(f"  Relay v{_relay.version()} | Arrow {pa.__version__} | "
          f"DuckDB {duckdb.__version__} | Polars {pl.__version__}")
    print(f"{'#'*70}")

    bench_open()
    bench_scan()
    bench_scan_read_only()
    bench_projection()
    bench_projection_read_only()
    bench_memory()

    print(f"\n{'#'*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'#'*70}\n")
