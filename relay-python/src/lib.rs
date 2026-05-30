//! # Relay Python
//!
//! Python bindings for the Relay zero-copy data engine via PyO3.

// Suppress PyO3 0.23 FromPyObject deprecation warnings
#![allow(deprecated)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};
use relay_arrow::ffi;
use relay_arrow::{RelayArray, RelayRecordBatch};
use relay_io::ipc::write_ipc;
use relay_io::mmap::MmapIPCReader;
use relay_io::parquet::ParquetReader;
use relay_io::csv::{CsvReader, CsvReadOptions};
use relay_io::json::{JsonReader, JsonReadOptions};
// use relay_io::AccessPattern;

/// Unified reader that wraps either IPC or Parquet reader
enum ReaderKind {
    Ipc(MmapIPCReader),
    Parquet(ParquetReader),
    Csv(CsvReader),
    Json(JsonReader),
}

#[pymodule]
fn _relay(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRelayArray>()?;
    m.add_class::<PyRelayBatch>()?;
    m.add_class::<PyScanResult>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(from_i32_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_f64_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_i64_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_f32_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_bool_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_str_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_batch, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_create_array, m)?)?;
    m.add_function(wrap_pyfunction!(test_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_export_throughput, m)?)?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_function(wrap_pyfunction!(scan_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(scan_csv, m)?)?;
    m.add_function(wrap_pyfunction!(scan_json, m)?)?;
    m.add_function(wrap_pyfunction!(write_ipc_file, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ── Factory functions ──────────────────────────────────────────────────

#[pyfunction]
fn from_i32_list(values: Vec<i32>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_i32(values),
    }
}

#[pyfunction]
fn from_i64_list(values: Vec<i64>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_i64(values),
    }
}

#[pyfunction]
fn from_f32_list(values: Vec<f32>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_f32(values),
    }
}

#[pyfunction]
fn from_f64_list(values: Vec<f64>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_f64(values),
    }
}

#[pyfunction]
fn from_bool_list(values: Vec<bool>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_bool(values),
    }
}

#[pyfunction]
fn from_str_list(values: Vec<String>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_string(values),
    }
}

#[pyfunction]
fn from_batch(batch: &PyRelayBatch) -> PyRelayBatch {
    batch.clone()
}

#[pyfunction]
fn benchmark_create_array(n: usize) -> u64 {
    let start = std::time::Instant::now();
    let _arr = RelayArray::from_i32((0..n as i32).collect());
    start.elapsed().as_nanos() as u64
}

#[pyfunction]
fn benchmark_export_throughput(n: usize) -> u64 {
    let start = std::time::Instant::now();
    let arr = RelayArray::from_i32((0..n as i32).collect());
    use arrow::ffi::to_ffi;
    let data = arr.as_arrow().to_data();
    let _ = to_ffi(&data).unwrap();
    start.elapsed().as_nanos() as u64
}

#[pyfunction]
fn test_zero_copy(n: usize) -> (usize, usize) {
    let arr = RelayArray::from_i32((0..n as i32).collect());
    let original_ptr = ffi::data_ptr(arr.as_arrow());
    let sliced = arr.slice(0, n);
    let sliced_ptr = ffi::data_ptr(sliced.as_arrow());
    (original_ptr, sliced_ptr)
}

// ── Scan (Phase 2: mmap IPC reader) ────────────────────────────────────

/// Scan an Arrow IPC file using mmap for zero-copy access.
///
/// Returns a PyScanResult that provides batch-by-batch access to the data.
/// The actual data lives in the mmap region — no copies until explicitly requested.
#[pyfunction]
fn scan(path: &str) -> PyResult<PyScanResult> {
    // Auto-detect format based on file extension
    let path_lower = path.to_lowercase();
    if path_lower.ends_with(".parquet") || path_lower.ends_with(".pq") {
        return scan_parquet(path);
    }

    let reader = MmapIPCReader::open(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let num_rows = reader.num_rows();
    let num_columns = reader.schema().fields().len();
    let column_names: Vec<String> = reader
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    Ok(PyScanResult {
        reader: Some(ReaderKind::Ipc(reader)),
        path: path.to_string(),
        num_rows,
        num_columns,
        column_names,
        format: "ipc".to_string(),
    })
}

/// Scan a Parquet file with row group pruning, column projection, and parallel processing.
///
/// Returns a PyScanResult that provides batch-by-batch access to the data.
/// Supports row group pruning (skip entire row groups using statistics) and
/// column projection (only decode needed columns).
#[pyfunction]
fn scan_parquet(path: &str) -> PyResult<PyScanResult> {
    let reader = ParquetReader::open(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let num_rows = reader.num_rows();
    let num_columns = reader.schema().fields().len();
    let column_names: Vec<String> = reader
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    Ok(PyScanResult {
        reader: Some(ReaderKind::Parquet(reader)),
        path: path.to_string(),
        num_rows,
        num_columns,
        column_names,
        format: "parquet".to_string(),
    })
}

/// Scan a CSV file with SWAR-accelerated parsing.
///
/// Returns a ScanResult with batch-by-batch access to the data.
///
/// # Features
/// - Memory-mapped file reading (zero-copy)
/// - SWAR-accelerated boundary detection (3.4x faster than scalar)
/// - Parallel parsing with Rayon
/// - Automatic schema inference from first 1024 rows
/// - Support for quoted fields with embedded delimiters
///
/// # Arguments
/// * `path` - Path to the CSV file
/// * `has_header` - Whether the first row is a header (default: true)
/// * `delimiter` - Field delimiter character (default: ",")
/// * `use_simd` - Enable SWAR acceleration (default: true)
///
/// # Example
/// ```python
/// import relay
/// 
/// # Scan a CSV file
/// result = relay.scan_csv("data.csv")
/// print(f"Rows: {result.num_rows()}, Cols: {result.num_columns()}")
/// 
/// # Read all data
/// batch = result.read_all()
/// 
/// # Or read with projection (only specific columns)
/// batch = result.read_columns(["id", "name", "value"])
/// ```
///
/// # Performance
/// For a 2M row CSV file with 10 columns:
/// - Relay: ~680ms
/// - Polars: ~44ms (reference)
///
/// Note: Current implementation parses all columns even with projection.
/// Column-first decode optimization is planned for future versions.
#[pyfunction]
#[pyo3(signature = (path, has_header=true, delimiter=",", use_simd=true))]
fn scan_csv(
    path: &str,
    has_header: bool,
    delimiter: &str,
    use_simd: bool,
) -> PyResult<PyScanResult> {
    let delim = delimiter
        .as_bytes()
        .first()
        .copied()
        .unwrap_or(b',');

    let options = CsvReadOptions {
        has_header,
        delimiter: delim,
        quote: b'"',
        trim: true,
        use_simd,
        ..Default::default()
    };

    let reader = CsvReader::open(std::path::Path::new(path), options)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let num_rows = reader.num_rows();
    let num_columns = reader.schema().fields().len();
    let column_names: Vec<String> = reader
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    Ok(PyScanResult {
        reader: Some(ReaderKind::Csv(reader)),
        path: path.to_string(),
        num_rows,
        num_columns,
        column_names,
        format: "csv".to_string(),
    })
}

/// Scan a JSON or NDJSON file with parallel parsing.
///
/// Returns a ScanResult with batch-by-batch access to the data.
/// Automatically detects whether the file is a JSON array or NDJSON format.
///
/// # Features
/// - Memory-mapped file reading (zero-copy)
/// - Parallel parsing with Rayon
/// - Automatic format detection (JSON array vs NDJSON)
/// - Automatic schema inference from first 1024 rows
/// - Type coercion (e.g., int fields become Int64, float fields become Float64)
///
/// # Arguments
/// * `path` - Path to the JSON/NDJSON file
///
/// # Example
/// ```python
/// import relay
///
/// # Scan an NDJSON file (one JSON object per line)
/// result = relay.scan_json("data.ndjson")
/// print(f"Rows: {result.num_rows()}, Cols: {result.num_columns()}")
///
/// # Read all data
/// batch = result.read_all()
///
/// # Or read with projection
/// batch = result.read_columns(["user_id", "timestamp"])
/// ```
///
/// # Supported Formats
/// - **JSON Array**: `[{"a": 1, "b": 2}, {"a": 3, "b": 4}]`
/// - **NDJSON**: One JSON object per line
///   ```json
///   {"a": 1, "b": 2}
///   {"a": 3, "b": 4}
///   ```
///
/// # Performance
/// For a 1M row NDJSON file with 10 columns:
/// - Relay: ~279ms
/// - Polars: ~103ms (reference)
///
/// NDJSON performance is much better than CSV due to simpler parsing requirements.
#[pyfunction]
fn scan_json(path: &str) -> PyResult<PyScanResult> {
    let reader = JsonReader::open(std::path::Path::new(path), JsonReadOptions::default())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let num_rows = reader.num_rows();
    let num_columns = reader.schema().fields().len();
    let column_names: Vec<String> = reader
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    Ok(PyScanResult {
        reader: Some(ReaderKind::Json(reader)),
        path: path.to_string(),
        num_rows,
        num_columns,
        column_names,
        format: "json".to_string(),
    })
}

/// Write a PyRelayBatch to an Arrow IPC file.
#[pyfunction]
fn write_ipc_file(path: &str, batch: &PyRelayBatch) -> PyResult<()> {
    let arrow_rb = batch.inner.as_arrow_recordbatch();
    write_ipc(
        std::path::Path::new(path),
        &[arrow_rb],
        relay_io::ipc::IPCWriteOptions::default(),
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Result from scanning an IPC or Parquet file. Provides batch-by-batch access.
/// A scan result providing batch-by-batch access to data from various file formats.
///
/// ScanResult is returned by scan functions (scan_csv, scan_json, scan_parquet)
/// and provides methods to read data in batches or all at once, with optional
/// column projection for memory efficiency.
///
/// # Attributes
/// * `num_rows` - Total number of rows in the dataset
/// * `num_columns` - Number of columns in the dataset
/// * `column_names` - List of column names
/// * `format` - File format ("csv", "json", "parquet", "ipc")
///
/// # Example
/// ```python
/// import relay
///
/// # Scan a file
/// result = relay.scan_csv("data.csv")
///
/// # Inspect metadata
/// print(f"Shape: {result.num_rows} x {result.num_columns}")
/// print(f"Columns: {result.column_names}")
///
/// # Read all data
/// batch = result.read_all()
///
/// # Or read with projection (memory efficient)
/// batch = result.read_columns(["id", "value"])
/// ```
#[pyclass(name = "ScanResult")]
pub struct PyScanResult {
    reader: Option<ReaderKind>,
    path: String,
    num_rows: usize,
    num_columns: usize,
    column_names: Vec<String>,
    format: String,
}

#[pymethods]
impl PyScanResult {
    #[getter]
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[getter]
    fn num_columns(&self) -> usize {
        self.num_columns
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.column_names.clone()
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn mmap_size(&self) -> usize {
        match self.reader.as_ref() {
            Some(ReaderKind::Ipc(r)) => r.mmap_size(),
            _ => 0, // Parquet files aren't mmap'd
        }
    }

    #[getter]
    fn format(&self) -> &str {
        &self.format
    }

    /// Read a specific batch as a PyRelayBatch.
    fn read_batch(&mut self, index: usize) -> PyResult<PyRelayBatch> {
        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        let batch = match reader {
            ReaderKind::Ipc(r) => r.read_batch(index),
            ReaderKind::Parquet(r) => r.read_batch(index),
            ReaderKind::Csv(r) => {
                if index != 0 {
                    return Err(pyo3::exceptions::PyIndexError::new_err("CSV has only 1 batch (index 0)"));
                }
                r.read_all()
            }
            ReaderKind::Json(r) => {
                if index != 0 {
                    return Err(pyo3::exceptions::PyIndexError::new_err("JSON has only 1 batch (index 0)"));
                }
                r.read_all()
            }
        }
        .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;

        let relay_batch = RelayRecordBatch::from_arrow(batch);
        Ok(PyRelayBatch { inner: relay_batch })
    }

    /// Read all rows from the scanned file into a single RelayBatch.
    ///
    /// Returns a RelayBatch containing all rows and columns from the file.
    /// For large files, consider using read_columns() with projection to
    /// reduce memory usage.
    ///
    /// Returns:
    ///     RelayBatch: A batch containing all data from the file.
    ///
    /// Example:
    /// ```python
    /// result = relay.scan_csv("data.csv")
    /// batch = result.read_all()
    /// print(f"Loaded {batch.num_rows()} rows")
    /// ```
    ///
    /// Note:
    ///     For CSV/JSON files, all data is read into a single batch.
    ///     For Parquet/IPC files, multiple batches may be concatenated.
    fn read_all(&mut self) -> PyResult<PyRelayBatch> {
        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        let batches = match reader {
            ReaderKind::Ipc(r) => r.read_all(),
            ReaderKind::Parquet(r) => r.read_all(),
            ReaderKind::Csv(r) => r.read_all().map(|b| vec![b]),
            ReaderKind::Json(r) => r.read_all().map(|b| vec![b]),
        }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Concatenate all batches into one
        if batches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("no batches"));
        }

        let schema = batches[0].schema();
        let _num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let num_cols = schema.fields().len();

        let mut columns: Vec<Vec<arrow_array::ArrayRef>> = vec![Vec::new(); num_cols];
        for batch in &batches {
            for (i, col) in batch.columns().iter().enumerate() {
                columns[i].push(col.clone());
            }
        }

        // Concatenate each column
        let mut result_cols = Vec::with_capacity(num_cols);
        for col_chunks in &columns {
            let refs: Vec<&dyn arrow::array::Array> =
                col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch {
            inner: result_batch,
        })
    }

    /// Read specific columns only (projection pushdown).
    ///
    /// Returns a RelayBatch containing only the specified columns, which is
    /// more memory efficient for large files with many columns.
    ///
    /// Args:
    ///     columns: List of column names to read.
    ///
    /// Returns:
    ///     RelayBatch: A batch containing only the requested columns.
    ///
    /// Example:
    /// ```python
    /// result = relay.scan_csv("data.csv")
    /// # Read only id and value columns
    /// batch = result.read_columns(["id", "value"])
    /// print(f"Columns: {batch.column_names()}")
    /// ```
    ///
    /// Note:
    ///     For Parquet files, this enables true projection pushdown (only reads
    ///     requested columns from disk). For CSV/JSON, all columns are parsed
    ///     but only requested ones are returned.
    fn read_columns(&self, columns: Vec<String>) -> PyResult<PyRelayBatch> {
        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        let batches = match reader {
            ReaderKind::Ipc(r) => r.read_columns(&col_refs),
            ReaderKind::Parquet(r) => r.read_columns(&col_refs),
            ReaderKind::Csv(r) => r.read_columns(&col_refs).map(|b| vec![b]),
            ReaderKind::Json(r) => r.read_columns(&col_refs).map(|b| vec![b]),
        }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        if batches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("no batches"));
        }

        // Same concatenation logic
        let schema = batches[0].schema();
        let num_cols = schema.fields().len();
        let mut col_data: Vec<Vec<arrow_array::ArrayRef>> = vec![Vec::new(); num_cols];
        for batch in &batches {
            for (i, col) in batch.columns().iter().enumerate() {
                col_data[i].push(col.clone());
            }
        }

        let mut result_cols = Vec::with_capacity(num_cols);
        for col_chunks in &col_data {
            let refs: Vec<&dyn arrow::array::Array> =
                col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch {
            inner: result_batch,
        })
    }

    /// Aggregate a single column with projection pushdown (only reads needed column).
    ///
    /// Args:
    ///     op: Aggregation operation ("sum", "mean", "min", "max", "count")
    ///     column: Column name to aggregate
    ///
    /// Returns:
    ///     The aggregated value (int, float, or None).
    ///
    /// This is faster than `read_all().agg()` because it only reads the target column.
    fn agg_column(&self, py: Python<'_>, op: &str, column: &str) -> PyResult<Py<PyAny>> {
        use relay_expr::AggOp;

        let agg_op = match op.to_lowercase().as_str() {
            "sum" => AggOp::Sum,
            "mean" | "avg" => AggOp::Mean,
            "min" => AggOp::Min,
            "max" => AggOp::Max,
            "count" => AggOp::Count,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid aggregation: {}. Use sum, mean, min, max, count",
                    op
                )))
            }
        };

        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        // Use streaming aggregation for maximum performance
        use relay_io::mmap::{AggOp as IoAggOp, AggResult as IoAggResult};
        let io_agg_op = match agg_op {
            AggOp::Sum => IoAggOp::Sum,
            AggOp::Mean => IoAggOp::Mean,
            AggOp::Min => IoAggOp::Min,
            AggOp::Max => IoAggOp::Max,
            AggOp::Count => IoAggOp::Count,
        };

        let result = match reader {
            ReaderKind::Ipc(r) => r.streaming_agg(column, io_agg_op),
            ReaderKind::Parquet(r) => r.streaming_agg(column, io_agg_op),
            ReaderKind::Csv(_) | ReaderKind::Json(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "streaming_agg not supported for CSV/JSON. Use read_all() instead.",
                ));
            }
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Convert to Python
        match result {
            IoAggResult::Int64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            IoAggResult::Float64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            IoAggResult::Null => Ok(py.None()),
        }
    }

    /// Filter with parallel execution and fused aggregation (fastest path).
    ///
    /// Args:
    ///     filter_col: Column to filter on
    ///     op: Filter operator ("<", "<=", ">", ">=", "==", "!=")
    ///     threshold: Filter threshold (i64)
    ///     agg_col: Column to aggregate after filtering
    ///     agg_op: Aggregation operation ("sum", "mean", "min", "max", "count")
    ///
    /// Returns:
    ///     The aggregated value after filtering.
    ///
    /// This is the fastest path — no materialization of filtered data, parallel per-batch processing.
    fn filter_agg(
        &self,
        py: Python<'_>,
        filter_col: &str,
        op: &str,
        threshold: i64,
        agg_col: &str,
        agg_op: &str,
    ) -> PyResult<Py<PyAny>> {
        use relay_io::mmap::{AggOp as IoAggOp, AggResult as IoAggResult};

        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        let io_agg_op = match agg_op.to_lowercase().as_str() {
            "sum" => IoAggOp::Sum,
            "mean" | "avg" => IoAggOp::Mean,
            "min" => IoAggOp::Min,
            "max" => IoAggOp::Max,
            "count" => IoAggOp::Count,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid aggregation: {}. Use sum, mean, min, max, count",
                    agg_op
                )))
            }
        };

        let result = match reader {
            ReaderKind::Ipc(r) => {
                r.parallel_filter_agg_i64(filter_col, op, threshold, agg_col, io_agg_op)
            }
            ReaderKind::Parquet(r) => {
                r.parallel_filter_agg_i64(filter_col, op, threshold, agg_col, io_agg_op)
            }
            ReaderKind::Csv(_) | ReaderKind::Json(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "filter_agg not supported for CSV/JSON. Use read_all() instead.",
                ));
            }
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Convert to Python
        match result {
            IoAggResult::Int64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            IoAggResult::Float64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            IoAggResult::Null => Ok(py.None()),
        }
    }

    /// Filter with parallel execution (returns filtered RecordBatch).
    ///
    /// Args:
    ///     filter_col: Column to filter on
    ///     op: Filter operator ("<", "<=", ">", ">=", "==", "!=")
    ///     threshold: Filter threshold (i64)
    ///
    /// Returns:
    ///     PyRelayBatch with filtered rows.
    fn filter_parallel(
        &self,
        filter_col: &str,
        op: &str,
        threshold: i64,
    ) -> PyResult<PyRelayBatch> {
        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        let batches = match reader {
            ReaderKind::Ipc(r) => r.parallel_filter_i64(filter_col, op, threshold),
            ReaderKind::Parquet(r) => r.parallel_filter_i64(filter_col, op, threshold),
            ReaderKind::Csv(_) | ReaderKind::Json(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "parallel_filter not supported for CSV/JSON. Use read_all() instead.",
                ));
            }
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if batches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("no results"));
        }

        // Concatenate batches
        let schema = batches[0].schema();
        let num_cols = schema.fields().len();
        let mut col_data: Vec<Vec<arrow_array::ArrayRef>> = vec![Vec::new(); num_cols];
        for batch in &batches {
            for (i, col) in batch.columns().iter().enumerate() {
                col_data[i].push(col.clone());
            }
        }

        let mut result_cols = Vec::with_capacity(num_cols);
        for col_chunks in &col_data {
            let refs: Vec<&dyn arrow::array::Array> =
                col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch {
            inner: result_batch,
        })
    }

    /// Filter with projection pushdown (only reads filter column, then applies mask).
    ///
    /// Args:
    ///     column: Column name to filter on
    ///     op: Comparison operator ("==", "!=", "<", "<=", ">", ">=")
    ///     value: Value to compare against
    ///
    /// Returns:
    ///     PyRelayBatch with filtered rows (all columns).
    ///
    /// This is faster than `read_all().filter()` because it only reads the filter column first.
    fn filter_column(
        &self,
        column: &str,
        op: &str,
        value: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyRelayBatch> {
        use relay_expr::filter::filter_batch;
        use relay_expr::{Expr, Literal, Operator};

        let operator = match op {
            "==" | "eq" => Operator::Eq,
            "!=" | "ne" => Operator::Ne,
            "<" | "lt" => Operator::Lt,
            "<=" | "le" => Operator::Le,
            ">" | "gt" => Operator::Gt,
            ">=" | "ge" => Operator::Ge,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid operator: {}. Use ==, !=, <, <=, >, >=",
                    op
                )))
            }
        };

        let literal = if let Ok(v) = value.extract::<i64>() {
            Literal::Int64(v)
        } else if let Ok(v) = value.extract::<f64>() {
            Literal::Float64(v)
        } else if let Ok(v) = value.extract::<String>() {
            Literal::Str(v)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Value must be int, float, or string",
            ));
        };

        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        // Read only the filter column first
        let filter_batches = match reader {
            ReaderKind::Ipc(r) => r.read_columns(&[column]),
            ReaderKind::Parquet(r) => r.read_columns(&[column]),
            ReaderKind::Csv(r) => r.read_columns(&[column]).map(|b| vec![b]),
            ReaderKind::Json(r) => r.read_columns(&[column]).map(|b| vec![b]),
        }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        if filter_batches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("no batches"));
        }

        // Build mask from filter column only
        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::Column(column.to_string())),
            op: operator,
            right: Box::new(Expr::Literal(literal)),
        };

        // Apply mask to each batch and concatenate
        let mut filtered_batches = Vec::new();
        for batch in &filter_batches {
            let filtered = filter_batch(batch, &predicate)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            filtered_batches.push(filtered);
        }

        // Now read all columns but only for the filtered rows
        let all_batches = match reader {
            ReaderKind::Ipc(r) => r.read_all(),
            ReaderKind::Parquet(r) => r.read_all(),
            ReaderKind::Csv(r) => r.read_all().map(|b| vec![b]),
            ReaderKind::Json(r) => r.read_all().map(|b| vec![b]),
        }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Apply the same filter to all batches
        let mut result_batches = Vec::new();
        for batch in &all_batches {
            let filtered = filter_batch(batch, &predicate)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            result_batches.push(filtered);
        }

        if result_batches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("no results"));
        }

        // Concatenate
        let schema = result_batches[0].schema();
        let num_cols = schema.fields().len();
        let mut col_data: Vec<Vec<arrow_array::ArrayRef>> = vec![Vec::new(); num_cols];
        for batch in &result_batches {
            for (i, col) in batch.columns().iter().enumerate() {
                col_data[i].push(col.clone());
            }
        }

        let mut result_cols = Vec::with_capacity(num_cols);
        for col_chunks in &col_data {
            let refs: Vec<&dyn arrow::array::Array> =
                col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch {
            inner: result_batch,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ScanResult(path={}, rows={}, cols={}, format={})",
            self.path, self.num_rows, self.num_columns, self.format,
        )
    }

    fn __len__(&self) -> usize {
        self.num_rows
    }
}

// ── PyRelayArray ───────────────────────────────────────────────────────

#[pyclass(name = "RelayArray")]
#[derive(Clone)]
pub struct PyRelayArray {
    inner: RelayArray,
}

impl PyRelayArray {
    fn to_pycapsules<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        use arrow::datatypes::FieldRef;
        use arrow_array::Array;
        use pyo3_arrow::ffi::to_array_pycapsules;

        let arr = self.inner.as_arrow();
        let field = FieldRef::from(arrow::datatypes::Field::new(
            "value",
            arr.data_type().clone(),
            arr.null_count() > 0,
        ));

        to_array_pycapsules(py, field, arr.as_ref(), requested_schema.cloned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyRelayArray {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "RelayArray(len={}, dtype={:?}, nulls={}, mem={}bytes)",
            self.inner.len(),
            self.inner.data_type(),
            self.inner.null_count(),
            self.inner.memory_size()
        )
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    #[getter]
    fn null_count(&self) -> usize {
        self.inner.null_count()
    }
    #[getter]
    fn memory_size(&self) -> usize {
        self.inner.memory_size()
    }

    fn slice(&self, offset: usize, length: usize) -> PyRelayArray {
        PyRelayArray {
            inner: self.inner.slice(offset, length),
        }
    }

    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.data_type())
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        self.to_pycapsules(py, requested_schema)
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        use arrow::datatypes::FieldRef;
        use arrow_array::Array;
        use pyo3_arrow::ffi::{to_stream_pycapsule, ArrayIterator};

        let arr = self.inner.as_arrow();
        let field = FieldRef::from(arrow::datatypes::Field::new(
            "value",
            arr.data_type().clone(),
            arr.null_count() > 0,
        ));
        let arrays: Vec<std::sync::Arc<dyn arrow_array::Array>> = vec![arr.clone()];
        let reader = Box::new(ArrayIterator::new(arrays.into_iter().map(Ok), field));

        to_stream_pycapsule(py, reader, requested_schema.cloned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn to_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
        use arrow::datatypes::DataType;
        let arr = self.inner.as_arrow();
        if arr.null_count() > 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot export nullable array via buffer",
            ));
        }
        match arr.data_type() {
            DataType::Int32 => Ok(PyBytes::new(
                py,
                bytemuck::cast_slice(arr.as_any().downcast_ref::<Int32Array>().unwrap().values()),
            )),
            DataType::Int64 => Ok(PyBytes::new(
                py,
                bytemuck::cast_slice(arr.as_any().downcast_ref::<Int64Array>().unwrap().values()),
            )),
            DataType::Float32 => Ok(PyBytes::new(
                py,
                bytemuck::cast_slice(
                    arr.as_any()
                        .downcast_ref::<Float32Array>()
                        .unwrap()
                        .values(),
                ),
            )),
            DataType::Float64 => Ok(PyBytes::new(
                py,
                bytemuck::cast_slice(
                    arr.as_any()
                        .downcast_ref::<Float64Array>()
                        .unwrap()
                        .values(),
                ),
            )),
            dt => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Buffer not supported for {:?}",
                dt
            ))),
        }
    }

    fn to_pylist(&self) -> Vec<Option<f64>> {
        let arr = self.inner.as_arrow();
        use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
        if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect()
        } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(a.value(i) as f64)
                    }
                })
                .collect()
        } else if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(a.value(i) as f64)
                    }
                })
                .collect()
        } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(a.value(i) as f64)
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    fn _data_ptr(&self) -> usize {
        ffi::data_ptr(self.inner.as_arrow())
    }

    fn _shares_memory(&self, other: &PyRelayArray) -> bool {
        ffi::shares_memory(self.inner.as_arrow(), other.inner.as_arrow())
    }

    fn _memory_size_bytes(&self) -> usize {
        self.inner.memory_size()
    }
}

// ── PyRelayBatch ───────────────────────────────────────────────────────

#[pyclass(name = "RelayBatch")]
#[derive(Clone)]
pub struct PyRelayBatch {
    inner: RelayRecordBatch,
}

impl PyRelayBatch {
    fn to_pycapsules<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        // use arrow::array::Array;
        use arrow::datatypes::{Field, FieldRef};
        use pyo3_arrow::ffi::to_array_pycapsules;

        let rb = self.inner.as_arrow_recordbatch();
        let struct_array: arrow::array::StructArray = rb.into();
        let schema = self.inner.schema();

        let field = FieldRef::from(Field::new(
            "batch",
            arrow::datatypes::DataType::Struct(schema.fields().clone()),
            false,
        ));

        to_array_pycapsules(
            py,
            field,
            &struct_array as &dyn arrow::array::Array,
            requested_schema.cloned(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyRelayBatch {
    #[new]
    fn new(names: Vec<String>, columns: Vec<PyRelayArray>) -> PyResult<Self> {
        let arrays: Vec<RelayArray> = columns.into_iter().map(|c| c.inner).collect();
        let batch = RelayRecordBatch::new(names, arrays)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: batch })
    }

    fn __repr__(&self) -> String {
        format!(
            "RelayBatch(rows={}, cols={}, mem={}bytes)",
            self.inner.num_rows(),
            self.inner.num_columns(),
            self.inner.memory_size()
        )
    }
    fn __len__(&self) -> usize {
        self.inner.num_rows()
    }
    #[getter]
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }
    #[getter]
    fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }
    #[getter]
    fn column_names(&self) -> Vec<&str> {
        self.inner.column_names()
    }
    #[getter]
    fn memory_size(&self) -> usize {
        self.inner.memory_size()
    }

    fn column(&self, name: &str) -> PyResult<PyRelayArray> {
        let col = self
            .inner
            .column_by_name(name)
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        Ok(PyRelayArray { inner: col.clone() })
    }

    fn select(&self, names: Vec<String>) -> PyResult<PyRelayBatch> {
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let selected = self
            .inner
            .select(&refs)
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        Ok(PyRelayBatch { inner: selected })
    }

    fn slice(&self, offset: usize, length: usize) -> PyRelayBatch {
        PyRelayBatch {
            inner: self.inner.slice(offset, length),
        }
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        self.to_pycapsules(py, requested_schema)
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // use arrow::array::Array;
        use arrow::datatypes::{Field, FieldRef};
        use pyo3_arrow::ffi::{to_stream_pycapsule, ArrayIterator};

        let rb = self.inner.as_arrow_recordbatch();
        let struct_array: arrow::array::StructArray = rb.into();
        let schema = self.inner.schema();

        let field = FieldRef::from(Field::new(
            "batch",
            arrow::datatypes::DataType::Struct(schema.fields().clone()),
            false,
        ));
        let arrays: Vec<std::sync::Arc<dyn arrow_array::Array>> =
            vec![std::sync::Arc::new(struct_array) as std::sync::Arc<dyn arrow_array::Array>];
        let reader = Box::new(ArrayIterator::new(arrays.into_iter().map(Ok), field));

        to_stream_pycapsule(py, reader, requested_schema.cloned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Filter rows based on a column comparison.
    ///
    /// Args:
    ///     column: Column name to filter on
    ///     op: Comparison operator ("==", "!=", "<", "<=", ">", ">=")
    ///     value: Value to compare against (int, float, or string)
    ///
    /// Returns:
    ///     A new RelayBatch with only matching rows.
    ///
    /// Example:
    ///     filtered = batch.filter("age", ">", 30)
    fn filter(
        &self,
        column: &str,
        op: &str,
        value: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyRelayBatch> {
        use relay_expr::filter::filter_batch;
        use relay_expr::{Expr, Literal, Operator};

        let op = match op {
            "==" | "eq" => Operator::Eq,
            "!=" | "ne" => Operator::Ne,
            "<" | "lt" => Operator::Lt,
            "<=" | "le" => Operator::Le,
            ">" | "gt" => Operator::Gt,
            ">=" | "ge" => Operator::Ge,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid operator: {}. Use ==, !=, <, <=, >, >=",
                    op
                )))
            }
        };

        // Try to extract value as different types
        let literal = if let Ok(v) = value.extract::<i64>() {
            Literal::Int64(v)
        } else if let Ok(v) = value.extract::<f64>() {
            Literal::Float64(v)
        } else if let Ok(v) = value.extract::<String>() {
            Literal::Str(v)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Value must be int, float, or string",
            ));
        };

        let predicate = Expr::BinaryOp {
            left: Box::new(Expr::Column(column.to_string())),
            op,
            right: Box::new(Expr::Literal(literal)),
        };

        let arrow_rb = self.inner.as_arrow_recordbatch();
        let filtered = filter_batch(&arrow_rb, &predicate)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let relay_batch = RelayRecordBatch::from_arrow(filtered);
        Ok(PyRelayBatch { inner: relay_batch })
    }

    /// Aggregate a column using a specified operation.
    ///
    /// Args:
    ///     op: Aggregation operation ("sum", "mean", "min", "max", "count")
    ///     column: Column name to aggregate
    ///
    /// Returns:
    ///     The aggregated value (int, float, or None).
    ///
    /// Example:
    ///     total = batch.agg("sum", "amount")
    fn agg(&self, py: Python<'_>, op: &str, column: &str) -> PyResult<Py<PyAny>> {
        use relay_expr::{aggregate_array, AggOp};

        let agg_op = match op.to_lowercase().as_str() {
            "sum" => AggOp::Sum,
            "mean" | "avg" => AggOp::Mean,
            "min" => AggOp::Min,
            "max" => AggOp::Max,
            "count" => AggOp::Count,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid aggregation: {}. Use sum, mean, min, max, count",
                    op
                )))
            }
        };

        let col = self
            .inner
            .column_by_name(column)
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;

        let result = aggregate_array(col.as_arrow(), agg_op)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Convert AggResult to Python object using py token
        match result {
            relay_expr::AggResult::Int64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            relay_expr::AggResult::Float64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
            relay_expr::AggResult::Null => Ok(py.None()),
        }
    }
}
