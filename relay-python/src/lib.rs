//! # Relay Python
//!
//! Python bindings for the Relay zero-copy data engine via PyO3.

use pyo3::prelude::*;
use relay_arrow::RelayArray;

/// The main Python module for Relay.
#[pymodule]
fn _relay(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRelayArray>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(from_i32_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_f64_list, m)?)?;
    m.add_function(wrap_pyfunction!(from_str_list, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_create_array, m)?)?;
    Ok(())
}

/// Get the Relay version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Create a RelayArray from a Python list of integers.
#[pyfunction]
fn from_i32_list(values: Vec<i32>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_i32(values),
    }
}

/// Create a RelayArray from a Python list of floats.
#[pyfunction]
fn from_f64_list(values: Vec<f64>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_f64(values),
    }
}

/// Create a RelayArray from a Python list of strings.
#[pyfunction]
fn from_str_list(values: Vec<String>) -> PyRelayArray {
    PyRelayArray {
        inner: RelayArray::from_string(values),
    }
}

/// Benchmark: create an i32 array of N elements, return elapsed nanoseconds.
#[pyfunction]
fn benchmark_create_array(n: usize) -> u64 {
    let start = std::time::Instant::now();
    let _arr = RelayArray::from_i32((0..n as i32).collect());
    start.elapsed().as_nanos() as u64
}

/// Python wrapper for RelayArray.
#[pyclass(name = "RelayArray")]
#[derive(Clone)]
pub struct PyRelayArray {
    inner: RelayArray,
}

#[pymethods]
impl PyRelayArray {
    /// Number of elements.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "RelayArray(len={}, type={:?}, nulls={})",
            self.inner.len(),
            self.inner.data_type(),
            self.inner.null_count()
        )
    }

    /// Number of elements.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the array is empty.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of null values.
    #[getter]
    fn null_count(&self) -> usize {
        self.inner.null_count()
    }

    /// Memory size in bytes.
    #[getter]
    fn memory_size(&self) -> usize {
        self.inner.memory_size()
    }

    /// Slice the array (zero-copy).
    fn slice(&self, offset: usize, length: usize) -> PyRelayArray {
        PyRelayArray {
            inner: self.inner.slice(offset, length),
        }
    }

    /// Get the Arrow data type as string.
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.data_type())
    }
}
