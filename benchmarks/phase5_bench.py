#!/usr/bin/env python3
"""
Relay v0.8 Benchmark: CSV + JSON vs Polars, DuckDB, Pandas, PyArrow
=====================================================================
"""
import gc
import os
import time
import statistics
import tempfile

import _relay as _r
import relay._relay as _relay
import polars as pl
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.json as pjson

WARMUP = 3
ITERATIONS = 7
DATA_DIR = tempfile.mkdtemp(prefix="relay_bench_")


def benchmark(fn, warmup=WARMUP, iterations=ITERATIONS):
    for _ in range(warmup):
        fn()
        gc.collect()
    times = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return statistics.median(times) / 1_000_000  # ms


def fmt(ms):
    if ms < 0.001:
        return f"{ms*1e6:.0f}ns"
    elif ms < 1:
        return f"{ms*1000:.1f}µs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


def create_csv_file(n, ncols=10):
    path = os.path.join(DATA_DIR, f"bench_{n}_{ncols}.csv")
    if os.path.exists(path):
        return path
    with open(path, "w") as f:
        header = ",".join([f"col_{i}" for i in range(ncols)])
        f.write(header + "\n")
        for row in range(n):
            vals = [str(row + c) for c in range(ncols)]
            f.write(",".join(vals) + "\n")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Created {path} ({size_mb:.1f} MB)")
    return path


def create_ndjson_file(n, ncols=10):
    path = os.path.join(DATA_DIR, f"bench_{n}_{ncols}.ndjson")
    if os.path.exists(path):
        return path
    with open(path, "w") as f:
        for row in range(n):
            obj = "{" + ",".join([f'"col_{c}":{row+c}' for c in range(ncols)]) + "}"
            f.write(obj + "\n")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Created {path} ({size_mb:.1f} MB)")
    return path


def print_header(title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def print_results(name, results):
    """results: dict of {engine: ms}"""
    engines = sorted(results.keys(), key=lambda e: results[e])
    best = engines[0]
    best_ms = results[best]

    print(f"\n  {name}")
    for engine in engines:
        ms = results[engine]
        ratio = ms / best_ms
        bar = "█" * min(int(ratio * 5), 50)
        marker = " 🏆" if engine == best else ""
        print(f"    {engine:>12}: {fmt(ms):>10}  ({ratio:.2f}x) {bar}{marker}")


# ─── CSV Benchmarks ────────────────────────────────────────────────

def bench_csv_read():
    print_header("CSV: Full Read")
    SIZES = [100_000, 500_000, 2_000_000]

    for n in SIZES:
        csv_path = create_csv_file(n, ncols=10)
        results = {}

        # Relay
        def relay_csv():
            s = _relay.scan_csv(csv_path)
            b = s.read_all()
        results["Relay"] = benchmark(relay_csv)

        # Polars
        def polars_csv():
            df = pl.read_csv(csv_path)
        results["Polars"] = benchmark(polars_csv)

        # DuckDB
        def duckdb_csv():
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM read_csv_auto('{csv_path}')").fetchdf()
            con.close()
        results["DuckDB"] = benchmark(duckdb_csv)

        # Pandas
        def pandas_csv():
            df = pd.read_csv(csv_path)
        results["Pandas"] = benchmark(pandas_csv)

        # PyArrow
        def pyarrow_csv():
            t = pcsv.read_csv(csv_path)
        results["PyArrow"] = benchmark(pyarrow_csv)

        print_results(f"n={n:,} rows × 10 columns", results)


def bench_csv_projection():
    print_header("CSV: Projection (2 of 10 columns)")
    SIZES = [100_000, 500_000, 2_000_000]

    for n in SIZES:
        csv_path = create_csv_file(n, ncols=10)
        results = {}

        # Relay
        def relay_proj():
            s = _relay.scan_csv(csv_path)
            b = s.read_columns(["col_0", "col_5"])
        results["Relay"] = benchmark(relay_proj)

        # Polars
        def polars_proj():
            df = pl.read_csv(csv_path, columns=["col_0", "col_5"])
        results["Polars"] = benchmark(polars_proj)

        # DuckDB
        def duckdb_proj():
            con = duckdb.connect()
            df = con.execute(f"SELECT col_0, col_5 FROM read_csv_auto('{csv_path}')").fetchdf()
            con.close()
        results["DuckDB"] = benchmark(duckdb_proj)

        # Pandas
        def pandas_proj():
            df = pd.read_csv(csv_path, usecols=["col_0", "col_5"])
        results["Pandas"] = benchmark(pandas_proj)

        # PyArrow
        def pyarrow_proj():
            t = pcsv.read_csv(csv_path, read_options=pa.csv.ReadOptions(column_names=None),
                              parse_options=pa.csv.ParseOptions(),
                              convert_options=pa.csv.ConvertOptions(include_columns=["col_0", "col_5"]))
        results["PyArrow"] = benchmark(pyarrow_proj)

        print_results(f"n={n:,} rows (2 of 10 cols)", results)


# ─── NDJSON Benchmarks ─────────────────────────────────────────────

def bench_ndjson_read():
    print_header("NDJSON: Full Read")
    SIZES = [100_000, 500_000, 1_000_000]

    for n in SIZES:
        json_path = create_ndjson_file(n, ncols=10)
        results = {}

        # Relay
        def relay_json():
            s = _relay.scan_json(json_path)
            b = s.read_all()
        results["Relay"] = benchmark(relay_json)

        # Polars
        def polars_json():
            df = pl.read_ndjson(json_path)
        results["Polars"] = benchmark(polars_json)

        # DuckDB
        def duckdb_json():
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM read_json_auto('{json_path}')").fetchdf()
            con.close()
        results["DuckDB"] = benchmark(duckdb_json)

        # Pandas
        def pandas_json():
            df = pd.read_json(json_path, lines=True)
        results["Pandas"] = benchmark(pandas_json)

        # PyArrow
        def pyarrow_json():
            t = pjson.read_json(json_path)
        results["PyArrow"] = benchmark(pyarrow_json)

        print_results(f"n={n:,} rows × 10 columns", results)


def bench_ndjson_projection():
    print_header("NDJSON: Projection (2 of 10 columns)")
    SIZES = [100_000, 500_000, 1_000_000]

    for n in SIZES:
        json_path = create_ndjson_file(n, ncols=10)
        results = {}

        # Relay
        def relay_proj():
            s = _relay.scan_json(json_path)
            b = s.read_columns(["col_0", "col_5"])
        results["Relay"] = benchmark(relay_proj)

        # Polars
        def polars_proj():
            df = pl.scan_ndjson(json_path).select(["col_0", "col_5"]).collect()
        results["Polars"] = benchmark(polars_proj)

        # DuckDB
        def duckdb_proj():
            con = duckdb.connect()
            df = con.execute(f"SELECT col_0, col_5 FROM read_json_auto('{json_path}')").fetchdf()
            con.close()
        results["DuckDB"] = benchmark(duckdb_proj)

        # Pandas
        def pandas_proj():
            df = pd.read_json(json_path, lines=True)[["col_0", "col_5"]]
        results["Pandas"] = benchmark(pandas_proj)

        print_results(f"n={n:,} rows (2 of 10 cols)", results)


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n📊 Relay I/O Benchmark v{_relay.version()}")
    print(f"   Engines: Relay, Polars {pl.__version__}, DuckDB {duckdb.__version__}, Pandas {pd.__version__}, PyArrow {pa.__version__}")
    print(f"   {WARMUP} warmup + {ITERATIONS} iterations per test")

    bench_csv_read()
    bench_csv_projection()
    bench_ndjson_read()
    bench_ndjson_projection()

    print(f"\n{'='*75}")
    print("  ✅ Benchmark complete!")
    print(f"{'='*75}\n")
