"""
Phase 3 E2E Tests: Expression Engine (filter + aggregate + projection)
=======================================================================
Tests: filter ops, aggregation ops, query plan pipeline.
"""
import os
import tempfile

import pytest
import _relay
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np

DATA_DIR = tempfile.mkdtemp(prefix="relay_test_")


@pytest.fixture
def sample_ipc():
    """Create a sample IPC file with 1000 rows and multiple column types."""
    schema = pa.schema([
        ('id', pa.int64()),
        ('value', pa.float64()),
        ('score', pa.int64()),
        ('name', pa.string()),
    ])
    table = pa.table({
        'id': pa.array(range(1000), type=pa.int64()),
        'value': pa.array([i * 1.5 for i in range(1000)], type=pa.float64()),
        'score': pa.array([i % 100 for i in range(1000)], type=pa.int64()),
        'name': pa.array([f"row_{i}" for i in range(1000)], type=pa.string()),
    })
    path = os.path.join(DATA_DIR, "sample.ipc")
    with ipc.new_file(path, table.schema) as w:
        w.write_table(table)
    return path


class TestFilterExpression:
    """Test vectorized filter engine."""

    def test_filter_gt(self, sample_ipc):
        """Filter: id > 500."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", ">", 500)
        assert filtered.num_rows == 499  # 501..999

    def test_filter_lt(self, sample_ipc):
        """Filter: id < 10."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", "<", 10)
        assert filtered.num_rows == 10

    def test_filter_eq(self, sample_ipc):
        """Filter: id == 42."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", "==", 42)
        assert filtered.num_rows == 1

    def test_filter_ne(self, sample_ipc):
        """Filter: score != 0."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("score", "!=", 0)
        # score = i % 100, so 0 appears 10 times (0, 100, 200, ..., 900)
        assert filtered.num_rows == 990

    def test_filter_le(self, sample_ipc):
        """Filter: id <= 5."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", "<=", 5)
        assert filtered.num_rows == 6  # 0,1,2,3,4,5

    def test_filter_ge(self, sample_ipc):
        """Filter: id >= 995."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", ">=", 995)
        assert filtered.num_rows == 5  # 995,996,997,998,999

    def test_filter_float(self, sample_ipc):
        """Filter on float column."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("value", "<", 15.0)
        # value = i * 1.5, so i*1.5 < 15 → i < 10
        assert filtered.num_rows == 10

    def test_filter_preserves_columns(self, sample_ipc):
        """Filtered batch should have same columns."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", "<", 5)
        assert filtered.num_columns == 4
        assert "id" in filtered.column_names
        assert "value" in filtered.column_names

    def test_filter_invalid_op(self, sample_ipc):
        """Invalid operator should raise."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        with pytest.raises(ValueError):
            batch.filter("id", "~", 5)


class TestAggregation:
    """Test vectorized aggregation engine."""

    def test_sum_int(self, sample_ipc):
        """SUM(id) for id 0..999 = 499500."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("sum", "id")
        assert result == sum(range(1000))

    def test_sum_float(self, sample_ipc):
        """SUM(value) for value = i*1.5."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("sum", "value")
        expected = sum(i * 1.5 for i in range(1000))
        assert abs(result - expected) < 0.01

    def test_mean_int(self, sample_ipc):
        """MEAN(id) for id 0..999 = 499.5."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("mean", "id")
        assert abs(result - 499.5) < 0.01

    def test_count(self, sample_ipc):
        """COUNT(id) = 1000."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("count", "id")
        assert result == 1000

    def test_min(self, sample_ipc):
        """MIN(id) = 0."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("min", "id")
        assert result == 0

    def test_max(self, sample_ipc):
        """MAX(id) = 999."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("max", "id")
        assert result == 999

    def test_avg(self, sample_ipc):
        """AVG alias for mean."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        result = batch.agg("avg", "id")
        assert abs(result - 499.5) < 0.01

    def test_agg_invalid_op(self, sample_ipc):
        """Invalid aggregation should raise."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        with pytest.raises(ValueError):
            batch.agg("median", "id")

    def test_agg_invalid_column(self, sample_ipc):
        """Invalid column should raise."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        with pytest.raises(KeyError):
            batch.agg("sum", "nonexistent")


class TestFilterThenAggregate:
    """Test combined filter + aggregate pipeline."""

    def test_filter_then_sum(self, sample_ipc):
        """SUM(id) WHERE id > 500."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", ">", 500)
        result = filtered.agg("sum", "id")
        expected = sum(range(501, 1000))
        assert result == expected

    def test_filter_then_count(self, sample_ipc):
        """COUNT(id) WHERE id < 100."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", "<", 100)
        result = filtered.agg("count", "id")
        assert result == 100

    def test_filter_then_mean(self, sample_ipc):
        """MEAN(value) WHERE id < 10."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        filtered = batch.filter("id", "<", 10)
        result = filtered.agg("mean", "value")
        expected = sum(i * 1.5 for i in range(10)) / 10
        assert abs(result - expected) < 0.01


class TestProjectionWithExpression:
    """Test select + filter + aggregate pipeline."""

    def test_project_then_filter(self, sample_ipc):
        """Select columns, then filter."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        projected = batch.select(["id", "value"])
        filtered = projected.filter("id", "<", 10)
        assert filtered.num_columns == 2
        assert filtered.num_rows == 10

    def test_project_then_aggregate(self, sample_ipc):
        """Select column, then aggregate."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        projected = batch.select(["value"])
        result = projected.agg("sum", "value")
        expected = sum(i * 1.5 for i in range(1000))
        assert abs(result - expected) < 0.01

    def test_full_pipeline(self, sample_ipc):
        """Full pipeline: scan → project → filter → aggregate."""
        sr = _relay.scan(sample_ipc)
        batch = sr.read_all()
        projected = batch.select(["id", "score"])
        filtered = projected.filter("id", "<", 100)
        result = filtered.agg("sum", "score")
        expected = sum(i % 100 for i in range(100))
        assert result == expected
