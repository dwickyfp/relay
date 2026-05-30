"""
Phase 1 E2E Tests: Relay ↔ PyArrow ↔ DuckDB ↔ Polars ↔ NumPy Interop
"""
import pytest
import relay._relay as _relay
import pyarrow as pa
import numpy as np
import duckdb
import polars as pl


# ── PyArrow round-trip ─────────────────────────────────────────────────

class TestPyArrowInterop:
    def test_i32_roundtrip(self):
        arr = _relay.from_i32_list([1, 2, 3, 4, 5])
        pa_arr = pa.array(arr)
        assert pa_arr.to_pylist() == [1, 2, 3, 4, 5]
        assert pa_arr.type == pa.int32()

    def test_i64_roundtrip(self):
        arr = _relay.from_i64_list([10**12, 10**13])
        pa_arr = pa.array(arr)
        assert pa_arr.to_pylist() == [10**12, 10**13]
        assert pa_arr.type == pa.int64()

    def test_f32_roundtrip(self):
        arr = _relay.from_f32_list([1.5, 2.5, 3.5])
        pa_arr = pa.array(arr)
        assert len(pa_arr) == 3
        assert pa_arr.type == pa.float32()

    def test_f64_roundtrip(self):
        arr = _relay.from_f64_list([1.1, 2.2, 3.3])
        pa_arr = pa.array(arr)
        assert pa_arr.to_pylist() == [1.1, 2.2, 3.3]
        assert pa_arr.type == pa.float64()

    def test_bool_roundtrip(self):
        arr = _relay.from_bool_list([True, False, True])
        pa_arr = pa.array(arr)
        assert pa_arr.to_pylist() == [True, False, True]
        assert pa_arr.type == pa.bool_()

    def test_string_roundtrip(self):
        arr = _relay.from_str_list(["hello", "world"])
        pa_arr = pa.array(arr)
        assert pa_arr.to_pylist() == ["hello", "world"]

    def test_batch_roundtrip(self):
        batch = _relay.RelayBatch(
            ["x", "y"],
            [_relay.from_i32_list([1, 2, 3]), _relay.from_f64_list([1.1, 2.2, 3.3])],
        )
        pa_rb = pa.record_batch(batch)
        assert pa_rb.num_rows == 3
        assert pa_rb.num_columns == 2
        assert pa_rb.schema.field(0).name == "x"
        assert pa_rb.schema.field(1).name == "y"
        assert pa_rb.column("x").to_pylist() == [1, 2, 3]

    def test_large_array_roundtrip(self):
        n = 100_000
        arr = _relay.from_i32_list(list(range(n)))
        pa_arr = pa.array(arr)
        assert len(pa_arr) == n
        assert pa_arr[0].as_py() == 0
        assert pa_arr[n - 1].as_py() == n - 1


# ── DuckDB round-trip ──────────────────────────────────────────────────

class TestDuckDBInterop:
    def test_duckdb_aggregate(self):
        batch = _relay.RelayBatch(
            ["x", "y"],
            [_relay.from_i32_list([1, 2, 3]), _relay.from_f64_list([1.1, 2.2, 3.3])],
        )
        pa_rb = pa.record_batch(batch)
        result = duckdb.sql("SELECT sum(x), avg(y) FROM pa_rb").fetchone()
        assert result[0] == 6
        assert abs(result[1] - 2.2) < 0.01

    def test_duckdb_filter(self):
        batch = _relay.RelayBatch(
            ["id", "val"],
            [_relay.from_i32_list([1, 2, 3, 4, 5]), _relay.from_f64_list([10, 20, 30, 40, 50])],
        )
        pa_rb = pa.record_batch(batch)
        result = duckdb.sql("SELECT val FROM pa_rb WHERE id > 3 ORDER BY id").fetchall()
        assert result == [(40.0,), (50.0,)]

    def test_duckdb_roundtrip_large(self):
        n = 10_000
        batch = _relay.RelayBatch(
            ["v"],
            [_relay.from_i64_list(list(range(n)))],
        )
        pa_rb = pa.record_batch(batch)
        result = duckdb.sql("SELECT count(*), sum(v) FROM pa_rb").fetchone()
        assert result[0] == n
        assert result[1] == sum(range(n))


# ── Polars round-trip ──────────────────────────────────────────────────

class TestPolarsInterop:
    def test_polars_series(self):
        arr = _relay.from_i32_list([10, 20, 30])
        s = pl.from_arrow(pa.array(arr))
        assert s.to_list() == [10, 20, 30]

    def test_polars_dataframe(self):
        batch = _relay.RelayBatch(
            ["a", "b"],
            [_relay.from_i32_list([1, 2]), _relay.from_f64_list([3.3, 4.4])],
        )
        df = pl.from_arrow(pa.record_batch(batch))
        assert df.shape == (2, 2)
        assert df["a"].to_list() == [1, 2]
        assert df["b"].to_list() == [3.3, 4.4]


# ── NumPy round-trip ───────────────────────────────────────────────────

class TestNumPyInterop:
    def test_buffer_i32(self):
        arr = _relay.from_i32_list([1, 2, 3])
        buf = arr.to_buffer()
        np_arr = np.frombuffer(buf, dtype=np.int32)
        assert list(np_arr) == [1, 2, 3]

    def test_buffer_i64(self):
        arr = _relay.from_i64_list([10**12, 10**13])
        buf = arr.to_buffer()
        np_arr = np.frombuffer(buf, dtype=np.int64)
        assert list(np_arr) == [10**12, 10**13]

    def test_buffer_f64(self):
        arr = _relay.from_f64_list([1.1, 2.2])
        buf = arr.to_buffer()
        np_arr = np.frombuffer(buf, dtype=np.float64)
        assert list(np_arr) == [1.1, 2.2]


# ── Zero-copy verification ─────────────────────────────────────────────

class TestZeroCopy:
    def test_slice_shares_memory(self):
        arr = _relay.from_i32_list([1, 2, 3, 4, 5])
        sliced = arr.slice(1, 3)
        assert arr._shares_memory(sliced)
        assert sliced.to_pylist() == [2.0, 3.0, 4.0]

    def test_batch_column_shares_memory(self):
        batch = _relay.RelayBatch(
            ["x", "y"],
            [_relay.from_i32_list([1, 2, 3]), _relay.from_f64_list([4.0, 5.0, 6.0])],
        )
        col_x = batch.column("x")
        col_y = batch.column("y")
        # Each column should share memory with original arrays
        assert col_x._data_ptr() == batch.column("x")._data_ptr()

    def test_buffer_no_copy(self):
        arr = _relay.from_i32_list([1, 2, 3])
        ptr = arr._data_ptr()
        buf = arr.to_buffer()
        np_arr = np.frombuffer(buf, dtype=np.int32)
        # np.frombuffer is zero-copy — same memory
        assert list(np_arr) == [1, 2, 3]


# ── Schema negotiation (requested_schema) ───────────────────────────────

class TestSchemaNegotiation:
    def test_arrow_c_array_returns_tuple(self):
        arr = _relay.from_i32_list([1, 2, 3])
        caps = arr.__arrow_c_array__()
        assert isinstance(caps, tuple)
        assert len(caps) == 2

    def test_arrow_c_array_capsule_names(self):
        arr = _relay.from_i32_list([1, 2, 3])
        schema_cap, array_cap = arr.__arrow_c_array__()
        assert repr(schema_cap).startswith('<capsule object "arrow_schema"')
        assert repr(array_cap).startswith('<capsule object "arrow_array"')

    def test_batch_arrow_c_stream(self):
        batch = _relay.RelayBatch(
            ["x"], [_relay.from_i32_list([1, 2, 3])]
        )
        stream_cap = batch.__arrow_c_stream__()
        assert "arrow_array_stream" in repr(stream_cap)
