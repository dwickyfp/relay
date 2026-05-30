//! # Relay Python
//!
//! Python bindings for the Relay zero-copy data engine via PyO3.

// Suppress PyO3 0.23 FromPyObject deprecation warnings
#![allow(deprecated)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};
use relay_arrow::ffi;
use relay_arrow::{RelayArray, RelayRecordBatch};
use relay_io::mmap::MmapIPCReader;
use relay_io::ipc::write_ipc;
// use relay_io::AccessPattern;

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
        reader: Some(reader),
        path: path.to_string(),
        num_rows,
        num_columns,
        column_names,
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

/// Result from scanning an IPC file. Provides batch-by-batch access.
#[pyclass(name = "ScanResult")]
pub struct PyScanResult {
    reader: Option<MmapIPCReader>,
    path: String,
    num_rows: usize,
    num_columns: usize,
    column_names: Vec<String>,
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
        self.reader.as_ref().map(|r| r.mmap_size()).unwrap_or(0)
    }

    /// Read a specific batch as a PyRelayBatch.
    fn read_batch(&mut self, index: usize) -> PyResult<PyRelayBatch> {
        let reader = self
            .reader
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ScanResult already consumed"))?;

        let batch = reader
            .read_batch(index)
            .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;

        let relay_batch = RelayRecordBatch::from_arrow(batch);
        Ok(PyRelayBatch { inner: relay_batch })
    }

    /// Read all batches as a single PyRelayBatch.
    fn read_all(&mut self) -> PyResult<PyRelayBatch> {
        let reader = self
            .reader
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ScanResult already consumed"))?;

        let batches = reader
            .read_all()
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
            let refs: Vec<&dyn arrow::array::Array> = col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch { inner: result_batch })
    }

    /// Read specific columns only (projection pushdown).
    fn read_columns(&self, columns: Vec<String>) -> PyResult<PyRelayBatch> {
        let reader = self
            .reader
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ScanResult already consumed"))?;

        let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        let batches = reader
            .read_columns(&col_refs)
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
            let refs: Vec<&dyn arrow::array::Array> = col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch { inner: result_batch })
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

        let result = reader.streaming_agg(column, io_agg_op)
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
    fn filter_agg(&self, py: Python<'_>, filter_col: &str, op: &str, threshold: i64, agg_col: &str, agg_op: &str) -> PyResult<Py<PyAny>> {
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

        let result = reader.parallel_filter_agg_i64(filter_col, op, threshold, agg_col, io_agg_op)
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
    fn filter_parallel(&self, filter_col: &str, op: &str, threshold: i64) -> PyResult<PyRelayBatch> {
        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("ScanResult already consumed")
        })?;

        let batches = reader.parallel_filter_i64(filter_col, op, threshold)
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
            let refs: Vec<&dyn arrow::array::Array> = col_chunks.iter().map(|c| c.as_ref()).collect();
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
        use relay_expr::{Expr, Literal, Operator};
        use relay_expr::filter::filter_batch;

        let operator = match op {
            "==" | "eq" => Operator::Eq,
            "!=" | "ne" => Operator::Ne,
            "<" | "lt" => Operator::Lt,
            "<=" | "le" => Operator::Le,
            ">" | "gt" => Operator::Gt,
            ">=" | "ge" => Operator::Ge,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid operator: {}. Use ==, !=, <, <=, >, >=", op)
            )),
        };

        let literal = if let Ok(v) = value.extract::<i64>() {
            Literal::Int64(v)
        } else if let Ok(v) = value.extract::<f64>() {
            Literal::Float64(v)
        } else if let Ok(v) = value.extract::<String>() {
            Literal::Str(v)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Value must be int, float, or string"
            ));
        };

        let reader = self
            .reader
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ScanResult already consumed"))?;

        // Read only the filter column first
        let filter_batches = reader
            .read_columns(&[column])
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
        let all_batches = reader
            .read_all()
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
            let refs: Vec<&dyn arrow::array::Array> = col_chunks.iter().map(|c| c.as_ref()).collect();
            let concatenated = arrow::compute::concat(&refs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            result_cols.push(RelayArray::new(concatenated));
        }

        let result_batch = RelayRecordBatch::new(
            schema.fields().iter().map(|f| f.name().clone()).collect(),
            result_cols,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyRelayBatch { inner: result_batch })
    }

    fn __repr__(&self) -> String {
        format!(
            "ScanResult(path={}, rows={}, cols={}, mmap={}bytes)",
            self.path,
            self.num_rows,
            self.num_columns,
            self.mmap_size()
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
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
                .collect()
        } else if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
                .collect()
        } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
            (0..a.len())
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
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
    fn filter(&self, column: &str, op: &str, value: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRelayBatch> {
        use relay_expr::{Expr, Literal, Operator};
        use relay_expr::filter::filter_batch;

        let op = match op {
            "==" | "eq" => Operator::Eq,
            "!=" | "ne" => Operator::Ne,
            "<" | "lt" => Operator::Lt,
            "<=" | "le" => Operator::Le,
            ">" | "gt" => Operator::Gt,
            ">=" | "ge" => Operator::Ge,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid operator: {}. Use ==, !=, <, <=, >, >=", op)
            )),
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
                "Value must be int, float, or string"
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
        use relay_expr::{AggOp, aggregate_array};

        let agg_op = match op.to_lowercase().as_str() {
            "sum" => AggOp::Sum,
            "mean" | "avg" => AggOp::Mean,
            "min" => AggOp::Min,
            "max" => AggOp::Max,
            "count" => AggOp::Count,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid aggregation: {}. Use sum, mean, min, max, count", op)
            )),
        };

        let col = self.inner.column_by_name(column)
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
