//! # Relay Python
//!
//! Python bindings for the Relay zero-copy data engine via PyO3.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};
use relay_arrow::ffi;
use relay_arrow::{RelayArray, RelayRecordBatch};

#[pymodule]
fn _relay(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRelayArray>()?;
    m.add_class::<PyRelayBatch>()?;
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
    // Measure only the FFI export path (not Python capsule creation)
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

// ── PyRelayArray ───────────────────────────────────────────────────────

#[pyclass(name = "RelayArray")]
#[derive(Clone)]
pub struct PyRelayArray {
    inner: RelayArray,
}

impl PyRelayArray {
    /// Create PyCapsules using pyo3-arrow's FFI helpers (proper heap allocation
    /// and destructors per the Arrow PyCapsule Interface spec).
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

    // ── Arrow PyCapsule Interface ──────────────────────────────────────
    // Returns (schema_capsule, array_capsule) per the Arrow spec.
    // Uses pyo3-arrow's to_array_pycapsules for correct heap allocation,
    // destructor registration, and schema negotiation.

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        self.to_pycapsules(py, requested_schema)
    }

    /// Arrow PyCapsule Stream interface — exports a single batch as a stream.
    /// Enables `pa.RecordBatchReader.from_batches(schema, batches)` style import.
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

    /// Buffer protocol — zero-copy with numpy via bytes (for non-nullable primitives)
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

    /// Return all primitive values as a Python list (for non-nullable arrays).
    /// Supports Int32, Int64, Float32, Float64, Boolean.
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
    /// Export as a PyCapsule tuple using pyo3-arrow's FFI helpers.
    /// The RecordBatch is wrapped as a StructArray for the Arrow PyCapsule Interface.
    fn to_pycapsules<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        use arrow::array::Array;
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

    // ── Arrow PyCapsule Interface ──────────────────────────────────────

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        self.to_pycapsules(py, requested_schema)
    }

    /// Arrow PyCapsule Stream interface for RecordBatch.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        use arrow::array::Array;
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
        let arrays: Vec<std::sync::Arc<dyn arrow_array::Array>> = vec![
            std::sync::Arc::new(struct_array) as std::sync::Arc<dyn arrow_array::Array>,
        ];
        let reader = Box::new(ArrayIterator::new(arrays.into_iter().map(Ok), field));

        to_stream_pycapsule(py, reader, requested_schema.cloned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}
