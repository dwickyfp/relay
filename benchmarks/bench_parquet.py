"""
Benchmark: Relay vs Polars vs DuckDB — Parquet operations on 2M rows.

Measures the operations that matter for big data:
1. Open + metadata (O(1) vs full scan)
2. Column projection (only read needed columns)
3. Streaming aggregation (sum with projection pushdown)
4. Filter + aggregate (fused, no materialization)
5. Parallel filter (materialize filtered rows)

Run: python benchmarks/bench_parquet.py
"""

import time
import os
import sys
from contextlib import contextmanager

PARQUET_PATH = "tests/data/big_2m.parquet"
ROUNDS = 5


@contextmanager
def timer(label):
    """Time a block, return elapsed ms."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


def bench_relay():
    import relay
    results = {}

    # 1. Open + metadata
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        r = relay.scan_parquet(PARQUET_PATH)
        _ = r.num_rows, r.column_names
        times.append((time.perf_counter() - t0) * 1000)
    results["open + metadata"] = min(times)

    # 2. Column projection (read 2 of 7 columns)
    times = []
    for _ in range(ROUNDS):
        r = relay.scan_parquet(PARQUET_PATH)
        t0 = time.perf_counter()
        batch = r.read_columns(["id", "amount"])
        _ = batch.num_rows
        times.append((time.perf_counter() - t0) * 1000)
    results["projection (2/7 cols)"] = min(times)

    # 3. Streaming agg sum with projection pushdown
    times = []
    for _ in range(ROUNDS):
        r = relay.scan_parquet(PARQUET_PATH)
        t0 = time.perf_counter()
        total = r.agg_column("sum", "quantity")
        times.append((time.perf_counter() - t0) * 1000)
    results["agg sum (projection)"] = min(times)

    # 4. Fused filter + agg (no materialization)
    times = []
    for _ in range(ROUNDS):
        r = relay.scan_parquet(PARQUET_PATH)
        t0 = time.perf_counter()
        total = r.filter_agg("quantity", ">", 900, "amount", "sum")
        times.append((time.perf_counter() - t0) * 1000)
    results["fused filter+agg"] = min(times)

    # 5. Parallel filter (materialize rows)
    times = []
    for _ in range(ROUNDS):
        r = relay.scan_parquet(PARQUET_PATH)
        t0 = time.perf_counter()
        batch = r.filter_parallel("quantity", ">", 950)
        _ = batch.num_rows
        times.append((time.perf_counter() - t0) * 1000)
    results["parallel filter (materialize)"] = min(times)

    return results


def bench_polars():
    import polars as pl
    results = {}

    # 1. Open + metadata
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        meta = pl.scan_parquet(PARQUET_PATH).collect_schema()
        _ = list(meta)
        times.append((time.perf_counter() - t0) * 1000)
    results["open + metadata"] = min(times)

    # 2. Column projection (read 2 of 7 columns)
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        df = pl.read_parquet(PARQUET_PATH, columns=["id", "amount"])
        _ = len(df)
        times.append((time.perf_counter() - t0) * 1000)
    results["projection (2/7 cols)"] = min(times)

    # 3. Agg sum with lazy projection pushdown
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        total = pl.scan_parquet(PARQUET_PATH).select("quantity").sum().collect()
        times.append((time.perf_counter() - t0) * 1000)
    results["agg sum (projection)"] = min(times)

    # 4. Filter + agg (lazy, predicate pushdown)
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        total = (
            pl.scan_parquet(PARQUET_PATH)
            .filter(pl.col("quantity") > 900)
            .select(pl.col("amount").sum())
            .collect()
        )
        times.append((time.perf_counter() - t0) * 1000)
    results["fused filter+agg"] = min(times)

    # 5. Filter (materialize rows)
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        df = pl.read_parquet(PARQUET_PATH).filter(pl.col("quantity") > 950)
        _ = len(df)
        times.append((time.perf_counter() - t0) * 1000)
    results["parallel filter (materialize)"] = min(times)

    return results


def bench_duckdb():
    import duckdb
    results = {}

    # 1. Open + metadata
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        conn = duckdb.connect()
        _ = conn.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM '{PARQUET_PATH}')"
        ).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    results["open + metadata"] = min(times)

    # 2. Column projection (read 2 of 7 columns)
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        conn = duckdb.connect()
        _ = conn.execute(
            f"SELECT id, amount FROM '{PARQUET_PATH}'"
        ).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    results["projection (2/7 cols)"] = min(times)

    # 3. Agg sum
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        conn = duckdb.connect()
        _ = conn.execute(f"SELECT SUM(quantity) FROM '{PARQUET_PATH}'").fetchone()
        times.append((time.perf_counter() - t0) * 1000)
    results["agg sum (projection)"] = min(times)

    # 4. Filter + agg
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        conn = duckdb.connect()
        _ = conn.execute(
            f"SELECT SUM(amount) FROM '{PARQUET_PATH}' WHERE quantity > 900"
        ).fetchone()
        times.append((time.perf_counter() - t0) * 1000)
    results["fused filter+agg"] = min(times)

    # 5. Filter (materialize)
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        conn = duckdb.connect()
        _ = conn.execute(
            f"SELECT * FROM '{PARQUET_PATH}' WHERE quantity > 950"
        ).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    results["parallel filter (materialize)"] = min(times)

    return results


def fmt_ms(ms):
    """Format milliseconds: <1ms show µs, else show ms."""
    if ms < 1.0:
        return f"{ms * 1000:.0f}µs"
    elif ms < 100:
        return f"{ms:.1f}ms"
    else:
        return f"{ms:.0f}ms"


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"ERROR: {PARQUET_PATH} not found. Generate first.")
        sys.exit(1)

    size_mb = os.path.getsize(PARQUET_PATH) / 1024 / 1024
    print(f"{'=' * 70}")
    print(f"  RELAY PARQUET BENCHMARK — 2M rows ({size_mb:.1f} MB, 8 row groups)")
    print(f"  Relay vs Polars vs DuckDB  |  {ROUNDS} rounds, best of")
    print(f"{'=' * 70}")

    print("\n🔄 Running Relay...")
    relay_results = bench_relay()

    print("🔄 Running Polars...")
    polars_results = bench_polars()

    print("🔄 Running DuckDB...")
    duckdb_results = bench_duckdb()

    # Print results table
    operations = list(relay_results.keys())

    print(f"\n{'Operation':<32} {'Relay':>10} {'Polars':>10} {'DuckDB':>10} {'Winner':>8}")
    print("-" * 72)

    relay_wins = 0
    polars_wins = 0
    duckdb_wins = 0

    for op in operations:
        r = relay_results[op]
        p = polars_results[op]
        d = duckdb_results[op]

        times = {"Relay": r, "Polars": p, "DuckDB": d}
        winner = min(times, key=times.get)

        if winner == "Relay":
            relay_wins += 1
        elif winner == "Polars":
            polars_wins += 1
        else:
            duckdb_wins += 1

        # Calculate speedup vs slowest
        slowest = max(r, p, d)
        r_speedup = f"({slowest / r:.1f}x)" if r < slowest else ""
        p_speedup = f"({slowest / p:.1f}x)" if p < slowest else ""
        d_speedup = f"({slowest / d:.1f}x)" if d < slowest else ""

        print(
            f"  {op:<30} {fmt_ms(r):>10} {fmt_ms(p):>10} {fmt_ms(d):>10}  {winner:>8}"
        )

    print("-" * 72)
    print(f"  Wins: Relay {relay_wins} | Polars {polars_wins} | DuckDB {duckdb_wins}")
    print()

    # Summary insight
    print("📊 Key insights:")

    # Filter+agg is the money operation for big data
    r_fused = relay_results["fused filter+agg"]
    p_fused = polars_results["fused filter+agg"]
    d_fused = duckdb_results["fused filter+agg"]
    slowest_fused = max(r_fused, p_fused, d_fused)
    fastest_fused = min(r_fused, p_fused, d_fused)
    print(f"  • Fused filter+agg: {fmt_ms(fastest_fused)} fastest, "
          f"{slowest_fused / fastest_fused:.1f}x spread")

    # Projection speedup
    r_proj = relay_results["projection (2/7 cols)"]
    r_full = relay_results["parallel filter (materialize)"]
    print(f"  • Relay projection (2/7 cols): {fmt_ms(r_proj)} vs full scan: {fmt_ms(r_full)}")

    # Open time
    r_open = relay_results["open + metadata"]
    p_open = polars_results["open + metadata"]
    print(f"  • Open+metadata: Relay {fmt_ms(r_open)} vs Polars {fmt_ms(p_open)}")


if __name__ == "__main__":
    main()
