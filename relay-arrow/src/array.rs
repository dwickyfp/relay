//! Wrapper around Arrow arrays with Relay-specific functionality.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow::datatypes::DataType;
use std::sync::Arc;

/// A zero-copy wrapper around an Apache Arrow array.
/// All operations that produce new arrays share memory where possible.
#[derive(Debug, Clone)]
pub struct RelayArray {
    inner: ArrayRef,
}

impl RelayArray {
    /// Wrap an existing Arrow ArrayRef (zero-copy).
    pub fn new(array: ArrayRef) -> Self {
        Self { inner: array }
    }

    /// Create from a Vec<i32> (allocates).
    pub fn from_i32(values: Vec<i32>) -> Self {
        Self {
            inner: Arc::new(Int32Array::from(values)),
        }
    }

    /// Create from a Vec<i64> (allocates).
    pub fn from_i64(values: Vec<i64>) -> Self {
        Self {
            inner: Arc::new(Int64Array::from(values)),
        }
    }

    /// Create from a Vec<f64> (allocates).
    pub fn from_f64(values: Vec<f64>) -> Self {
        Self {
            inner: Arc::new(Float64Array::from(values)),
        }
    }

    /// Create from a Vec<bool> (allocates).
    pub fn from_bool(values: Vec<bool>) -> Self {
        Self {
            inner: Arc::new(BooleanArray::from(values)),
        }
    }

    /// Create from string slices (allocates).
    pub fn from_str(values: Vec<&str>) -> Self {
        Self {
            inner: Arc::new(StringArray::from(values)),
        }
    }

    /// Create from owned Strings (allocates).
    pub fn from_string(values: Vec<String>) -> Self {
        Self {
            inner: Arc::new(StringArray::from(values)),
        }
    }

    /// Create a nullable i32 array (allocates).
    pub fn from_i32_nullable(values: Vec<Option<i32>>) -> Self {
        Self {
            inner: Arc::new(Int32Array::from(values)),
        }
    }

    /// Create a nullable f64 array (allocates).
    pub fn from_f64_nullable(values: Vec<Option<f64>>) -> Self {
        Self {
            inner: Arc::new(Float64Array::from(values)),
        }
    }

    /// Create a nullable string array (allocates).
    pub fn from_str_nullable(values: Vec<Option<&str>>) -> Self {
        Self {
            inner: Arc::new(StringArray::from(values)),
        }
    }

    /// Get the underlying Arrow ArrayRef (zero-copy).
    pub fn as_arrow(&self) -> &ArrayRef {
        &self.inner
    }

    /// Consume self and return the inner Arrow ArrayRef (zero-copy).
    pub fn into_arrow(self) -> ArrayRef {
        self.inner
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of null values.
    pub fn null_count(&self) -> usize {
        self.inner.null_count()
    }

    /// Whether the array has any null values.
    pub fn has_nulls(&self) -> bool {
        self.inner.null_count() > 0
    }

    /// Get the Arrow data type.
    pub fn data_type(&self) -> &DataType {
        self.inner.data_type()
    }

    /// Slice the array (zero-copy — returns a view into the same memory).
    /// Returns rows [offset..offset+length).
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        Self {
            inner: self.inner.slice(offset, length),
        }
    }

    /// Get the memory size in bytes (including validity bitmap).
    pub fn memory_size(&self) -> usize {
        self.inner.get_array_memory_size()
    }

    /// Downcast to Int32Array (zero-copy if type matches).
    pub fn as_i32(&self) -> Option<&Int32Array> {
        self.inner.as_any().downcast_ref::<Int32Array>()
    }

    /// Downcast to Int64Array.
    pub fn as_i64(&self) -> Option<&Int64Array> {
        self.inner.as_any().downcast_ref::<Int64Array>()
    }

    /// Downcast to Float64Array.
    pub fn as_f64(&self) -> Option<&Float64Array> {
        self.inner.as_any().downcast_ref::<Float64Array>()
    }

    /// Downcast to BooleanArray.
    pub fn as_bool(&self) -> Option<&BooleanArray> {
        self.inner.as_any().downcast_ref::<BooleanArray>()
    }

    /// Downcast to StringArray.
    pub fn as_str(&self) -> Option<&StringArray> {
        self.inner.as_any().downcast_ref::<StringArray>()
    }
}

impl From<ArrayRef> for RelayArray {
    fn from(array: ArrayRef) -> Self {
        Self::new(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_i32_array() {
        let arr = RelayArray::from_i32(vec![1, 2, 3, 4, 5]);
        assert_eq!(arr.len(), 5);
        assert!(!arr.is_empty());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.has_nulls());
        assert_eq!(*arr.data_type(), DataType::Int32);
    }

    #[test]
    fn test_create_i64_array() {
        let arr = RelayArray::from_i64(vec![100, 200, 300]);
        assert_eq!(arr.len(), 3);
        let i64_arr = arr.as_i64().unwrap();
        assert_eq!(i64_arr.value(0), 100);
        assert_eq!(i64_arr.value(1), 200);
    }

    #[test]
    fn test_create_f64_array() {
        let arr = RelayArray::from_f64(vec![1.1, 2.2, 3.3]);
        assert_eq!(arr.len(), 3);
        assert_eq!(*arr.data_type(), DataType::Float64);
        let f64_arr = arr.as_f64().unwrap();
        assert!((f64_arr.value(0) - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_create_bool_array() {
        let arr = RelayArray::from_bool(vec![true, false, true]);
        assert_eq!(arr.len(), 3);
        let bool_arr = arr.as_bool().unwrap();
        assert!(bool_arr.value(0));
        assert!(!bool_arr.value(1));
    }

    #[test]
    fn test_create_str_array() {
        let arr = RelayArray::from_str(vec!["hello", "world"]);
        assert_eq!(arr.len(), 2);
        let str_arr = arr.as_str().unwrap();
        assert_eq!(str_arr.value(0), "hello");
        assert_eq!(str_arr.value(1), "world");
    }

    #[test]
    fn test_nullable_array() {
        let arr = RelayArray::from_i32_nullable(vec![Some(1), None, Some(3)]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.has_nulls());
    }

    #[test]
    fn test_nullable_f64() {
        let arr = RelayArray::from_f64_nullable(vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn test_nullable_str() {
        let arr = RelayArray::from_str_nullable(vec![Some("a"), None, Some("c")]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn test_slice_zero_copy() {
        let arr = RelayArray::from_i32(vec![1, 2, 3, 4, 5]);
        let sliced = arr.slice(1, 3);
        assert_eq!(sliced.len(), 3);
        let i32_arr = sliced.as_i32().unwrap();
        assert_eq!(i32_arr.value(0), 2);
        assert_eq!(i32_arr.value(1), 3);
        assert_eq!(i32_arr.value(2), 4);
    }

    #[test]
    fn test_empty_array() {
        let arr = RelayArray::from_i32(vec![]);
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn test_memory_size() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let size = arr.memory_size();
        assert!(size > 0); // Should include validity bitmap + data buffer
    }

    #[test]
    fn test_from_arrow_ref() {
        let arrow_arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let relay_arr = RelayArray::from(arrow_arr.clone());
        assert_eq!(relay_arr.len(), 3);
        // Same underlying memory
        assert!(std::sync::Arc::ptr_eq(relay_arr.as_arrow(), &arrow_arr));
    }

    #[test]
    fn test_into_arrow() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let arrow_ref = arr.into_arrow();
        assert_eq!(arrow_ref.len(), 3);
    }

    #[test]
    fn test_downcast_wrong_type() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        assert!(arr.as_f64().is_none());
        assert!(arr.as_bool().is_none());
        assert!(arr.as_str().is_none());
        assert!(arr.as_i32().is_some());
    }

    #[test]
    fn test_clone_shares_memory() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let cloned = arr.clone();
        // Both should point to the same underlying Arrow array
        assert!(std::sync::Arc::ptr_eq(arr.as_arrow(), cloned.as_arrow()));
    }
}
