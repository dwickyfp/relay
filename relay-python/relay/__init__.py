"""
Relay: High-performance zero-copy data engine.

Supports Arrow IPC (mmap) and Parquet (row group pruning, column projection,
parallel processing via Rayon, SIMD expressions, fused filter+agg).

Quick start:
    import relay

    # Arrow IPC (zero-copy mmap)
    result = relay.scan("data.arrow")
    batch = result.read_all()

    # Parquet (row group pruning + column projection)
    result = relay.scan("data.parquet")  # auto-detects format
    batch = result.read_columns(["col_a", "col_b"])

    # Aggregation with projection pushdown (fastest path)
    total = result.agg_column("sum", "amount")

    # Fused filter + aggregate (no materialization)
    filtered_sum = result.filter_agg("amount", ">", 100, "amount", "sum")
"""

from relay._relay import (
    RelayArray,
    RelayBatch,
    ScanResult,
    version,
    scan,
    scan_parquet,
    write_ipc_file,
    from_i32_list,
    from_i64_list,
    from_f32_list,
    from_f64_list,
    from_bool_list,
    from_str_list,
    from_batch,
    benchmark_create_array,
    benchmark_export_throughput,
    test_zero_copy,
)

__version__ = version()
__all__ = [
    "scan",
    "scan_parquet",
    "write_ipc_file",
    "RelayArray",
    "RelayBatch",
    "ScanResult",
    "version",
    "from_i32_list",
    "from_i64_list",
    "from_f32_list",
    "from_f64_list",
    "from_bool_list",
    "from_str_list",
    "from_batch",
    "benchmark_create_array",
    "benchmark_export_throughput",
    "test_zero_copy",
]
