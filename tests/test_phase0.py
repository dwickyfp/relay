"""
Relay Phase 0 — E2E Tests
Tests the full Python→Rust→Python round-trip.
"""

import relay._relay as _relay


def test_version():
    """Relay version matches Cargo.toml."""
    assert _relay.version() == "0.6.0"


def test_create_i32_array():
    """Create i32 array from Python list."""
    arr = _relay.from_i32_list([1, 2, 3, 4, 5])
    assert arr.len == 5
    assert arr.is_empty is False
    assert arr.null_count == 0
    assert arr.dtype == "Int32"


def test_create_f64_array():
    """Create f64 array from Python list."""
    arr = _relay.from_f64_list([1.1, 2.2, 3.3])
    assert arr.len == 3
    assert arr.null_count == 0
    assert arr.dtype == "Float64"


def test_create_str_array():
    """Create string array from Python list."""
    arr = _relay.from_str_list(["hello", "world", "relay"])
    assert arr.len == 3
    assert arr.null_count == 0
    assert arr.dtype == "Utf8"


def test_empty_array():
    """Create empty array."""
    arr = _relay.from_i32_list([])
    assert arr.len == 0
    assert arr.is_empty is True


def test_large_array():
    """Create large array (1M elements)."""
    arr = _relay.from_i32_list(list(range(1_000_000)))
    assert arr.len == 1_000_000
    assert arr.null_count == 0
    assert arr.memory_size > 0


def test_slice_zero_copy():
    """Slice an array (should be zero-copy)."""
    arr = _relay.from_i32_list([1, 2, 3, 4, 5])
    sliced = arr.slice(1, 3)
    assert sliced.len == 3


def test_repr():
    """String representation."""
    arr = _relay.from_i32_list([1, 2, 3])
    r = repr(arr)
    assert "RelayArray" in r
    assert "len=3" in r
    assert "Int32" in r


def test_memory_size():
    """Memory size is reported correctly."""
    arr = _relay.from_i32_list(list(range(1000)))
    size = arr.memory_size
    assert size > 0
    # 1000 * 4 bytes (i32) + overhead
    assert size >= 4000


def test_benchmark_create():
    """Benchmark function returns valid timing."""
    ns = _relay.benchmark_create_array(1000)
    assert ns > 0


if __name__ == "__main__":
    tests = [
        test_version,
        test_create_i32_array,
        test_create_f64_array,
        test_create_str_array,
        test_empty_array,
        test_large_array,
        test_slice_zero_copy,
        test_repr,
        test_memory_size,
        test_benchmark_create,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")

    if failed > 0:
        exit(1)
    print("\n✅ All E2E tests passed!")
