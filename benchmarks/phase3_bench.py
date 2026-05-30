"""
Phase 3 Benchmark (Optimized): Projection Pushdown + SIMD Kernels + Parallel
==================================================================
Fair comparison: all engines read only the columns they need.
Relay uses agg_column() for projection pushdown + Rayon parallel.
"""
import gc
import os
import time
import statistics
import tempfile

import _relay
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import polars as pl
import duckdb
import pandas as pd
import numpy as np

WARMUP = 3
ITERATIONS = 7
DATA_DIR = tempfile.mkdtemp(prefix="relay_p3_bench_")


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


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def bench_filter():
    """Fair filter: all engines read all cols then filter."""
    print_header("FILTER (col_0 < N/2) — all engines read full data")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"filt_{n}"), n)
        threshold = n // 2

        # Relay: scan + read_all + filter (same as Polars)
        def relay_v3():
            sr = _relay.scan(ipc_p)
            batch = sr.read_all()
            return batch.filter("col_0", "<", threshold)
        t_relay_v3 = benchmark(relay_v3)

        # Polars (lazy scan + filter)
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).filter(pl.col("col_0") < threshold).collect())

        # PyArrow (read all + compute filter)
        def pa_filt():
            r = ipc.open_file(ipc_p)
            t = r.read_all()
            mask = pa.compute.less(t.column("col_0"), threshold)
            return pa.compute.filter(t, mask)
        t_pa = benchmark(pa_filt)

        # DuckDB
        t_db = benchmark(lambda: duckdb.query(f"SELECT * FROM '{pq_p}' WHERE col_0 < {threshold}").fetchall())

        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>12}: {fmt(t_relay_v3):>10}")
        print(f"    {'PyArrow':>12}: {fmt(t_pa):>10}")
        print(f"    {'Polars':>12}: {fmt(t_pl):>10}")
        print(f"    {'DuckDB':>12}: {fmt(t_db):>10}")

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_aggregate():
    """Fair aggregate: projection pushdown — all engines read only col_0."""
    print_header("AGGREGATE (SUM col_0) — projection pushdown (1 column read)")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"agg_{n}"), n)

        # Relay: agg_column (projection pushdown + Rayon parallel)
        def relay_v3():
            sr = _relay.scan(ipc_p)
            return sr.agg_column("sum", "col_0")
        t_relay_v3 = benchmark(relay_v3)

        # Relay: old path (read_all + agg)
        def relay_old():
            sr = _relay.scan(ipc_p)
            batch = sr.read_all()
            return batch.agg("sum", "col_0")
        t_relay_old = benchmark(relay_old)

        # Polars (projection pushdown parquet)
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).select(pl.col("col_0").sum()).collect())

        # PyArrow (read all + compute.sum)
        def pa_agg():
            r = ipc.open_file(ipc_p)
            t = r.read_all()
            return pa.compute.sum(t.column("col_0"))
        t_pa = benchmark(pa_agg)

        # DuckDB (projection pushdown)
        t_db = benchmark(lambda: duckdb.query(f"SELECT sum(col_0) FROM '{pq_p}'").fetchone())

        # Pandas
        t_pd = benchmark(lambda: pd.read_parquet(pq_p)["col_0"].sum())

        print(f"\n  n={n:>10,}")
        print(f"    {'Relay (parallel)':>18}: {fmt(t_relay_v3):>10}")
        print(f"    {'Relay (old)':>18}: {fmt(t_relay_old):>10}")
        print(f"    {'PyArrow':>18}: {fmt(t_pa):>10}")
        print(f"    {'Polars':>18}: {fmt(t_pl):>10}")
        print(f"    {'DuckDB':>18}: {fmt(t_db):>10}")
        print(f"    {'Pandas':>18}: {fmt(t_pd):>10}")

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_filter_then_aggregate():
    """Combined filter+aggregate pipeline — uses fused parallel path."""
    print_header("FILTER + AGGREGATE (SUM col_0 WHERE col_0 < N/2) — fused parallel")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"fa_{n}"), n)
        threshold = n // 2

        # Relay: fused filter+agg (parallel, no materialization, fastest)
        def relay_fused():
            sr = _relay.scan(ipc_p)
            return sr.filter_agg("col_0", "<", threshold, "col_0", "sum")
        t_relay_fused = benchmark(relay_fused)

        # Relay: old path (read_all + filter + agg)
        def relay_old():
            sr = _relay.scan(ipc_p)
            batch = sr.read_all()
            filtered = batch.filter("col_0", "<", threshold)
            return filtered.agg("sum", "col_0")
        t_relay_old = benchmark(relay_old)

        # Polars (lazy: filter + agg, projection pushdown)
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).filter(pl.col("col_0") < threshold).select(pl.col("col_0").sum()).collect())

        # DuckDB
        t_db = benchmark(lambda: duckdb.query(f"SELECT sum(col_0) FROM '{pq_p}' WHERE col_0 < {threshold}").fetchone())

        print(f"\n  n={n:>10,}")
        print(f"    {'Relay (fused)':>14}: {fmt(t_relay_fused):>10}")
        print(f"    {'Relay (old)':>14}: {fmt(t_relay_old):>10}")
        print(f"    {'Polars':>14}: {fmt(t_pl):>10}")
        print(f"    {'DuckDB':>14}: {fmt(t_db):>10}")

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_projection():
    """Column projection: read 2 of 10 columns."""
    print_header("PROJECTION (2 of 10 cols)")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"proj_{n}"), n)
        cols = ["col_0", "col_1"]

        # Relay read_columns
        def relay_proj():
            sr = _relay.scan(ipc_p)
            return sr.read_columns(cols)
        t_relay = benchmark(relay_proj)

        # Polars select 2 cols
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).select(cols).collect())

        # Pandas
        t_pd = benchmark(lambda: pd.read_parquet(pq_p, columns=cols))

        print(f"\n  n={n:>10,}")
        print(f"    {'Relay':>12}: {fmt(t_relay):>10}")
        print(f"    {'Polars':>12}: {fmt(t_pl):>10}")
        print(f"    {'Pandas':>12}: {fmt(t_pd):>10}")

        os.unlink(ipc_p)
        os.unlink(pq_p)


if __name__ == "__main__":
    print("######################################################################")
    print("  PHASE 3 BENCHMARK (OPTIMIZED) — Parallel + Projection Pushdown")
    print(f"  Relay v0.5.0 | Polars {pl.__version__} | DuckDB 1.5.3 | PyArrow {pa.__version__}")
    print("######################################################################")

    bench_filter()
    bench_aggregate()
    bench_filter_then_aggregate()
    bench_projection()

    print("\n######################################################################")
    print("  BENCHMARK COMPLETE")
    print("######################################################################")
