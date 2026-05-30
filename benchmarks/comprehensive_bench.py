#!/usr/bin/env python3
"""
Comprehensive Benchmark: Relay vs All Engines
==============================================
Compares: Relay, PyArrow, DuckDB, Polars, Pandas, NumPy, Daft
Workloads: open, scan, projection, filter, aggregate, join
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
import pandas as pd
import numpy as np
import daft

WARMUP = 2
ITERATIONS = 5
DATA_DIR = tempfile.mkdtemp(prefix="relay_bench_")

# Engine versions
VERSIONS = {
    "Relay": _relay.version(),
    "PyArrow": pa.__version__,
    "DuckDB": duckdb.__version__,
    "Polars": pl.__version__,
    "Pandas": pd.__version__,
    "NumPy": np.__version__,
    "Daft": daft.__version__,
}


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


def create_ipc_file(path, n, ncols=10):
    """Create IPC file with n rows and ncols int64 columns."""
    names = [f"col_{i}" for i in range(ncols)]
    arrays = [pa.array(range(n), type=pa.int64()) for _ in range(ncols)]
    table = pa.table(dict(zip(names, arrays)))
    with ipc.new_file(path, table.schema) as w:
        w.write_table(table)
    return path


def create_parquet_file(path, n, ncols=10):
    """Create Parquet file with n rows and ncols int64 columns."""
    names = [f"col_{i}" for i in range(ncols)]
    arrays = [pa.array(range(n), type=pa.int64()) for _ in range(ncols)]
    table = pa.table(dict(zip(names, arrays)))
    pq.write_table(table, path, row_group_size=min(n, 100_000))
    return path


def create_files(prefix, n, ncols=10):
    ipc_p = f"{prefix}.ipc"
    pq_p = f"{prefix}.parquet"
    create_ipc_file(ipc_p, n, ncols)
    create_parquet_file(pq_p, n, ncols)
    return ipc_p, pq_p


SIZES = [50_000, 500_000, 2_000_000]


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_row(name, ms, width=12):
    print(f"    {name:>{width}}: {fmt(ms):>10}")


def bench_file_open():
    print_header("1. FILE OPEN / SCAN INIT")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"open_{n}"), n)
        csv_p = pq_p  # use parquet for comparison

        t_relay = benchmark(lambda: _relay.scan(ipc_p))
        t_pa = benchmark(lambda: ipc.open_file(ipc_p))
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT count(*) FROM '{pq_p}'").fetchone())
        t_pd = benchmark(lambda: pd.read_parquet(pq_p))
        t_daft = benchmark(lambda: daft.read_parquet(pq_p).collect())

        print(f"\n  n={n:>10,}")
        print_row("Relay", t_relay)
        print_row("PyArrow", t_pa)
        print_row("Polars", t_pl)
        print_row("DuckDB", t_db)
        print_row("Pandas", t_pd)
        print_row("Daft", t_daft)

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_full_scan():
    print_header("2. FULL TABLE SCAN")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"scan_{n}"), n)

        def relay_s():
            sr = _relay.scan(ipc_p)
            return sr.read_all()
        t_relay = benchmark(relay_s)

        def pa_s():
            r = ipc.open_file(ipc_p)
            return r.read_all()
        t_pa = benchmark(pa_s)

        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT * FROM '{pq_p}'").fetchall())
        t_pd = benchmark(lambda: pd.read_parquet(pq_p))
        t_daft = benchmark(lambda: daft.read_parquet(pq_p).collect())

        # NumPy: direct mmap-based read
        def np_s():
            return np.memmap(ipc_p, dtype=np.uint8, mode='r')
        t_np = benchmark(np_s)

        print(f"\n  n={n:>10,}")
        print_row("Relay", t_relay)
        print_row("PyArrow", t_pa)
        print_row("Polars", t_pl)
        print_row("DuckDB", t_db)
        print_row("Pandas", t_pd)
        print_row("Daft", t_daft)
        print_row("NumPy mmap", t_np)

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_projection():
    print_header("3. COLUMN PROJECTION (2 of 10 cols)")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"proj_{n}"), n)
        cols = ["col_0", "col_1"]

        def relay_p():
            sr = _relay.scan(ipc_p)
            return sr.read_columns(cols)
        t_relay = benchmark(relay_p)

        def pa_p():
            r = ipc.open_file(ipc_p)
            return r.read_all()
        t_pa = benchmark(pa_p)

        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).select(cols).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT col_0, col_1 FROM '{pq_p}'").fetchall())
        t_pd = benchmark(lambda: pd.read_parquet(pq_p, columns=cols))
        t_daft = benchmark(lambda: daft.read_parquet(pq_p).select(*cols).collect())

        print(f"\n  n={n:>10,}")
        print_row("Relay", t_relay)
        print_row("PyArrow", t_pa)
        print_row("Polars", t_pl)
        print_row("DuckDB", t_db)
        print_row("Pandas", t_pd)
        print_row("Daft", t_daft)

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_filter():
    print_header("4. FILTER (col_0 < N/2)")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"filt_{n}"), n)
        threshold = n // 2

        def relay_f():
            sr = _relay.scan(ipc_p)
            batch = sr.read_all()
            col = batch.column("col_0")
            buf = col.to_buffer()
            arr = np.frombuffer(buf, dtype=np.int64)
            return arr[arr < threshold]
        t_relay = benchmark(relay_f)

        def pa_f():
            r = ipc.open_file(ipc_p)
            t = r.read_all()
            mask = pa.compute.less(t.column("col_0"), threshold)
            return pa.compute.filter(t, mask)
        t_pa = benchmark(pa_f)

        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).filter(pl.col("col_0") < threshold).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT * FROM '{pq_p}' WHERE col_0 < {threshold}").fetchall())
        t_pd = benchmark(lambda: pd.read_parquet(pq_p).query(f"col_0 < {threshold}"))
        t_daft = benchmark(lambda: daft.read_parquet(pq_p).where(daft.col("col_0") < threshold).collect())

        print(f"\n  n={n:>10,}")
        print_row("Relay", t_relay)
        print_row("PyArrow", t_pa)
        print_row("Polars", t_pl)
        print_row("DuckDB", t_db)
        print_row("Pandas", t_pd)
        print_row("Daft", t_daft)

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_aggregate():
    print_header("5. AGGREGATE (SUM col_0)")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"agg_{n}"), n)

        def relay_a():
            sr = _relay.scan(ipc_p)
            batch = sr.read_all()
            col = batch.column("col_0")
            buf = col.to_buffer()
            arr = np.frombuffer(buf, dtype=np.int64)
            return arr.sum()
        t_relay = benchmark(relay_a)

        def pa_a():
            r = ipc.open_file(ipc_p)
            t = r.read_all()
            return pa.compute.sum(t.column("col_0"))
        t_pa = benchmark(pa_a)

        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).select(pl.col("col_0").sum()).collect())
        t_db = benchmark(lambda: duckdb.query(f"SELECT sum(col_0) FROM '{pq_p}'").fetchone())
        t_pd = benchmark(lambda: pd.read_parquet(pq_p)["col_0"].sum())
        t_daft = benchmark(lambda: daft.read_parquet(pq_p).agg(daft.col("col_0").sum()).collect())

        # NumPy direct
        def np_a():
            return np.memmap(ipc_p, dtype=np.uint8, mode='r').sum()
        t_np = benchmark(np_a)

        print(f"\n  n={n:>10,}")
        print_row("Relay", t_relay)
        print_row("PyArrow", t_pa)
        print_row("Polars", t_pl)
        print_row("DuckDB", t_db)
        print_row("Pandas", t_pd)
        print_row("Daft", t_daft)
        print_row("NumPy", t_np)

        os.unlink(ipc_p)
        os.unlink(pq_p)


def bench_memory():
    print_header("6. MEMORY USAGE (2M rows, 10 cols, int64)")
    n = 2_000_000
    ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"mem_{n}"), n)
    expected = n * 10 * 8

    engines = {}

    # Relay
    tracemalloc.start()
    sr = _relay.scan(ipc_p)
    _, r_open = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    rb = sr.read_all()
    _, r_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engines["Relay (open)"] = r_open
    engines["Relay (full)"] = r_full

    # Polars
    tracemalloc.start()
    df = pl.scan_parquet(pq_p).collect()
    _, pl_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engines["Polars"] = pl_peak

    # DuckDB
    tracemalloc.start()
    tbl = duckdb.query(f"SELECT * FROM '{pq_p}'").to_arrow_table()
    _, db_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engines["DuckDB"] = db_peak

    # Pandas
    tracemalloc.start()
    pdf = pd.read_parquet(pq_p)
    _, pd_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engines["Pandas"] = pd_peak

    # PyArrow
    tracemalloc.start()
    pat = ipc.open_file(ipc_p).read_all()
    _, pa_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engines["PyArrow"] = pa_peak

    # Daft
    tracemalloc.start()
    dft = daft.read_parquet(pq_p).collect()
    _, daft_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engines["Daft"] = daft_peak

    print(f"\n  Expected raw data: {expected:,} B ({expected/1e6:.0f} MB)")
    print(f"\n  {'Engine':>18} {'Peak Memory':>14} {'vs Expected':>12}")
    for name, mem in engines.items():
        ratio = f"{mem/expected:.4f}x" if expected > 0 else "N/A"
        print(f"  {name:>18} {mem:>11,} B {ratio:>12}")

    ipc_sz = os.path.getsize(ipc_p)
    pq_sz = os.path.getsize(pq_p)
    print(f"\n  IPC file size:     {ipc_sz:>11,} B ({ipc_sz/1e6:.0f} MB)")
    print(f"  Parquet file size: {pq_sz:>11,} B ({pq_sz/1e6:.0f} MB)")

    os.unlink(ipc_p)
    os.unlink(pq_p)


def bench_read_only_scan():
    """Pre-opened scan: measure read_all() only (no open overhead)."""
    print_header("7. READ-ONLY SCAN (pre-opened, no open overhead)")
    for n in SIZES:
        ipc_p, pq_p = create_files(os.path.join(DATA_DIR, f"ro_{n}"), n)

        # Pre-open all engines
        sr = _relay.scan(ipc_p)
        pa_reader = ipc.open_file(ipc_p)

        t_relay = benchmark(lambda: sr.read_all())
        t_pa = benchmark(lambda: pa_reader.read_all())
        t_pl = benchmark(lambda: pl.scan_parquet(pq_p).collect())
        t_pd = benchmark(lambda: pd.read_parquet(pq_p))

        print(f"\n  n={n:>10,}")
        print_row("Relay", t_relay)
        print_row("PyArrow", t_pa)
        print_row("Polars", t_pl)
        print_row("Pandas", t_pd)

        os.unlink(ipc_p)
        os.unlink(pq_p)


if __name__ == "__main__":
    print(f"\n{'#'*70}")
    print(f"  COMPREHENSIVE BENCHMARK — Relay vs All Engines")
    print(f"{'#'*70}")
    print(f"\n  Engine Versions:")
    for name, ver in VERSIONS.items():
        print(f"    {name:>10}: {ver}")

    bench_file_open()
    bench_full_scan()
    bench_read_only_scan()
    bench_projection()
    bench_filter()
    bench_aggregate()
    bench_memory()

    print(f"\n{'#'*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'#'*70}\n")
