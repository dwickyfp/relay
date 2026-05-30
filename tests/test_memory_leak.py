"""
Memory leak detection for Relay mmap-based IPC reader.
Tests repeated scan/read cycles to ensure no memory leaks.
"""
import gc
import tempfile
import tracemalloc

import _relay
import pyarrow as pa
import pyarrow.ipc as ipc


def test_mmap_memory_leak_scan_read():
    """
    Test that repeated scan+read cycles don't leak memory.
    
    Strategy:
    1. Create a test IPC file
    2. Run many scan+read_all cycles
    3. Track memory usage with tracemalloc
    4. Assert memory growth is bounded (no leak)
    """
    # Create test data
    schema = pa.schema([
        ('id', pa.int64()),
        ('value', pa.float64()),
        ('data', pa.string()),
    ])
    table = pa.table({
        'id': pa.array(range(10000)),
        'value': pa.array([i * 1.5 for i in range(10000)]),
        'data': pa.array([f"row_{i}" for i in range(10000)]),
    }, schema=schema)
    
    with tempfile.NamedTemporaryFile(suffix='.ipc', delete=False) as f:
        ipc_path = f.name
        writer = ipc.new_file(f, schema)
        writer.write_table(table)
        writer.close()
    
    # Warm up
    for _ in range(5):
        sr = _relay.scan(ipc_path)
        batch = sr.read_all()
        del sr, batch
    gc.collect()
    
    # Start tracking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    # Run many cycles
    num_cycles = 100
    for _ in range(num_cycles):
        sr = _relay.scan(ipc_path)
        batch = sr.read_all()
        # Access the data to ensure it's materialized
        _ = batch.num_columns
        _ = batch.num_rows
        del sr, batch
    
    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # Compare snapshots
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    # Calculate total memory growth
    total_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    
    # Allow up to 1MB growth for 100 cycles (10KB per cycle max)
    # This is generous - actual should be much less with mmap
    max_allowed_growth = 1024 * 1024  # 1MB
    
    print(f"\nMemory leak test results:")
    print(f"  Cycles: {num_cycles}")
    print(f"  Memory growth: {total_growth:,} bytes")
    print(f"  Per cycle: {total_growth / num_cycles:.1f} bytes")
    print(f"  Max allowed: {max_allowed_growth:,} bytes")
    
    # Show top memory consumers if any
    if total_growth > 1024:
        print("\nTop memory consumers:")
        for stat in stats[:5]:
            if stat.size_diff > 0:
                print(f"  {stat.size_diff:>+8d} {stat.traceback}")
    
    assert total_growth < max_allowed_growth, (
        f"Memory leak detected! Growth: {total_growth:,} bytes "
        f"(allowed: {max_allowed_growth:,} bytes)"
    )


def test_mmap_memory_leak_projection():
    """
    Test that repeated column projection reads don't leak memory.
    """
    # Create test data with many columns
    num_cols = 20
    schema = pa.schema([(f'col_{i}', pa.int64()) for i in range(num_cols)])
    table = pa.table({f'col_{i}': pa.array(range(5000)) for i in range(num_cols)}, schema=schema)
    
    with tempfile.NamedTemporaryFile(suffix='.ipc', delete=False) as f:
        ipc_path = f.name
        writer = ipc.new_file(f, schema)
        writer.write_table(table)
        writer.close()
    
    # Warm up
    for _ in range(5):
        sr = _relay.scan(ipc_path)
        batch = sr.read_columns([f'col_{i}' for i in range(5)])
        del sr, batch
    gc.collect()
    
    # Start tracking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    # Run many cycles with different column sets
    num_cycles = 100
    for cycle in range(num_cycles):
        # Rotate which columns we read
        cols = [f'col_{(cycle + i) % num_cols}' for i in range(5)]
        sr = _relay.scan(ipc_path)
        batch = sr.read_columns(cols)
        _ = batch.num_rows
        del sr, batch
    
    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    
    max_allowed_growth = 1024 * 1024  # 1MB
    
    print(f"\nProjection memory leak test results:")
    print(f"  Cycles: {num_cycles}")
    print(f"  Memory growth: {total_growth:,} bytes")
    print(f"  Per cycle: {total_growth / num_cycles:.1f} bytes")
    
    assert total_growth < max_allowed_growth, (
        f"Memory leak in projection! Growth: {total_growth:,} bytes"
    )


def test_mmap_memory_leak_batch_iteration():
    """
    Test that iterating through batches doesn't leak memory.
    Uses read_batch() with known batch count.
    """
    # Create multi-batch file
    schema = pa.schema([('id', pa.int64()), ('value', pa.float64())])
    
    with tempfile.NamedTemporaryFile(suffix='.ipc', delete=False) as f:
        ipc_path = f.name
        writer = ipc.new_file(f, schema)
        # Write 10 batches of 1000 rows each
        num_batches = 10
        for batch_idx in range(num_batches):
            batch = pa.table({
                'id': pa.array(range(batch_idx * 1000, (batch_idx + 1) * 1000)),
                'value': pa.array([i * 0.1 for i in range(1000)]),
            }, schema=schema)
            writer.write_table(batch)
        writer.close()
    
    # Warm up
    for _ in range(5):
        sr = _relay.scan(ipc_path)
        for i in range(num_batches):
            batch = sr.read_batch(i)
            del batch
        del sr
    gc.collect()
    
    # Start tracking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    # Run many cycles
    num_cycles = 50
    for _ in range(num_cycles):
        sr = _relay.scan(ipc_path)
        for i in range(num_batches):
            batch = sr.read_batch(i)
            _ = batch.num_rows
            del batch
        del sr
    
    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    
    max_allowed_growth = 1024 * 1024  # 1MB
    
    print(f"\nBatch iteration memory leak test results:")
    print(f"  Cycles: {num_cycles}")
    print(f"  Batches per cycle: {num_batches}")
    print(f"  Memory growth: {total_growth:,} bytes")
    print(f"  Per cycle: {total_growth / num_cycles:.1f} bytes")
    
    assert total_growth < max_allowed_growth, (
        f"Memory leak in batch iteration! Growth: {total_growth:,} bytes"
    )


if __name__ == '__main__':
    print("=" * 70)
    print("Running memory leak detection tests")
    print("=" * 70)
    
    test_mmap_memory_leak_scan_read()
    test_mmap_memory_leak_projection()
    test_mmap_memory_leak_batch_iteration()
    
    print("\n" + "=" * 70)
    print("All memory leak tests PASSED ✓")
    print("=" * 70)
