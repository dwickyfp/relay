"""
Integration tests for Relay Parquet reader.

Tests the full pipeline: Rust Parquet reader → Python bindings.
Covers scan_parquet, format-detecting scan, projection, aggregation, filtering.
"""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import tempfile
import os

import relay


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a small Parquet file for fast unit tests."""
    n = 10_000
    np.random.seed(42)
    table = pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "amount": pa.array(np.random.uniform(0, 1000, n).round(2)),
        "quantity": pa.array(np.random.randint(1, 100, n, dtype=np.int64)),
        "category": pa.array(np.random.choice(["A", "B", "C"], size=n)),
    })
    path = str(tmp_path / "sample.parquet")
    pq.write_table(table, path, row_group_size=2_500)
    return path, table


@pytest.fixture
def big_parquet():
    """Use the pre-generated 2M row Parquet file."""
    path = "tests/data/big_2m.parquet"
    if not os.path.exists(path):
        pytest.skip("big_2m.parquet not found — run gen first")
    return path


# ── scan / scan_parquet ──────────────────────────────────────────────────

class TestScanParquet:
    def test_scan_parquet_basic(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)

        assert result.num_rows == 10_000
        assert result.num_columns == 4
        assert set(result.column_names) == {"id", "amount", "quantity", "category"}
        assert result.format == "parquet"

    def test_scan_auto_detects_parquet(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan(path)  # should auto-detect .parquet

        assert result.format == "parquet"
        assert result.num_rows == 10_000

    def test_scan_all_parquet(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)
        batch = result.read_all()

        assert batch.num_rows == 10_000
        assert batch.num_columns == 4

    def test_scan_batch_by_index(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)

        # 10K rows / 2.5K per row group = 4 row groups
        batch_0 = result.read_batch(0)
        assert batch_0.num_rows == 2_500

    def test_scan_batch_out_of_bounds(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)

        with pytest.raises(IndexError):
            result.read_batch(999)


# ── Column projection ────────────────────────────────────────────────────

class TestProjection:
    def test_read_columns_subset(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)
        batch = result.read_columns(["id", "amount"])

        assert batch.num_columns == 2
        assert set(batch.column_names) == {"id", "amount"}
        assert batch.num_rows == 10_000

    def test_read_single_column(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)
        batch = result.read_columns(["quantity"])

        assert batch.num_columns == 1
        assert batch.num_rows == 10_000

    def test_column_data_integrity(self, sample_parquet):
        """Verify data survives Parquet encode → decode roundtrip."""
        path, original = sample_parquet
        result = relay.scan_parquet(path)
        batch = result.read_all()

        # id column should match
        relay_ids = batch.column("id").to_pylist()
        original_ids = original["id"].to_pylist()
        assert relay_ids == original_ids


# ── Aggregation ──────────────────────────────────────────────────────────

class TestAggregation:
    def test_sum(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)
        relay_sum = result.agg_column("sum", "quantity")

        expected = sum(original["quantity"].to_pylist())
        assert relay_sum == expected

    def test_mean(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)
        relay_mean = result.agg_column("mean", "quantity")

        expected = sum(original["quantity"].to_pylist()) / len(original["quantity"].to_pylist())
        assert abs(relay_mean - expected) < 0.01

    def test_min_max(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)

        assert result.agg_column("min", "quantity") == min(original["quantity"].to_pylist())
        assert result.agg_column("max", "quantity") == max(original["quantity"].to_pylist())

    def test_count(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)

        assert result.agg_column("count", "quantity") == 10_000


# ── Filter ───────────────────────────────────────────────────────────────

class TestFilter:
    def test_filter_gt(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)

        batch = result.filter_parallel("id", ">", 9_990)
        ids = batch.column("id").to_pylist()
        assert all(i > 9_990 for i in ids)
        assert len(ids) == 9  # 10000 - 9991

    def test_filter_eq(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)

        batch = result.filter_parallel("id", "==", 0)
        assert batch.num_rows == 1
        assert batch.column("id").to_pylist()[0] == 0

    def test_filter_column_eq(self, sample_parquet):
        """Test column projection + filter (late materialization path)."""
        path, _ = sample_parquet
        result = relay.scan_parquet(path)

        batch = result.filter_column("id", "<", 5)
        assert batch.num_rows == 5


# ── Fused filter+agg ─────────────────────────────────────────────────────

class TestFusedFilterAgg:
    def test_filter_agg(self, sample_parquet):
        """Fused filter+agg: no materialization of filtered data."""
        path, original = sample_parquet
        result = relay.scan_parquet(path)

        # Filter: quantity > 90, then sum(amount)
        relay_sum = result.filter_agg("quantity", ">", 90, "amount", "sum")

        expected = sum(
            a for q, a in zip(
                original["quantity"].to_pylist(),
                original["amount"].to_pylist(),
            )
            if q > 90
        )
        assert abs(relay_sum - expected) < 0.01

    def test_filter_agg_mean(self, sample_parquet):
        path, original = sample_parquet
        result = relay.scan_parquet(path)

        relay_mean = result.filter_agg("quantity", ">=", 50, "amount", "mean")

        vals = [
            a for q, a in zip(
                original["quantity"].to_pylist(),
                original["amount"].to_pylist(),
            )
            if q >= 50
        ]
        expected = sum(vals) / len(vals)
        assert abs(relay_mean - expected) < 0.1


# ── Big data (2M rows) ──────────────────────────────────────────────────

class TestBigData:
    def test_scan_2m(self, big_parquet):
        result = relay.scan_parquet(big_parquet)

        assert result.num_rows == 2_000_000
        assert result.num_columns == 7
        assert result.format == "parquet"

    def test_projection_2m(self, big_parquet):
        result = relay.scan_parquet(big_parquet)
        batch = result.read_columns(["id", "amount"])

        assert batch.num_columns == 2
        assert batch.num_rows == 2_000_000

    def test_agg_2m(self, big_parquet):
        result = relay.scan_parquet(big_parquet)
        total = result.agg_column("sum", "quantity")

        assert isinstance(total, (int, float))
        assert total > 0  # sanity
        # quantity is randint(1, 1000) with seed 42, mean ~500 * 2M = ~1B
        assert 800_000_000 < total < 1_200_000_000

    def test_filter_agg_2m(self, big_parquet):
        result = relay.scan_parquet(big_parquet)
        total = result.filter_agg("quantity", ">", 900, "amount", "sum")

        assert isinstance(total, (int, float))
        assert total > 0

    def test_filter_parallel_2m(self, big_parquet):
        result = relay.scan_parquet(big_parquet)
        batch = result.filter_parallel("quantity", ">", 950)

        assert batch.num_rows > 0
        # Verify all values are > 950
        quantities = batch.column("quantity").to_pylist()
        assert all(q > 950 for q in quantities)


# ── Repr / len ───────────────────────────────────────────────────────────

class TestRepr:
    def test_scan_result_repr(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)
        r = repr(result)

        assert "ScanResult" in r
        assert "parquet" in r
        assert "10000" in r or "10,000" in r

    def test_scan_result_len(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)

        assert len(result) == 10_000

    def test_batch_repr(self, sample_parquet):
        path, _ = sample_parquet
        result = relay.scan_parquet(path)
        batch = result.read_all()
        r = repr(batch)

        assert "RelayBatch" in r
