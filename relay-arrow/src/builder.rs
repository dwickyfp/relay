//! Builder pattern for constructing Arrow arrays incrementally.

use arrow::array::ArrayBuilder as ArrowBuilderTrait;
use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int32Builder, Int64Builder, StringBuilder,
};
use arrow::datatypes::DataType;
use std::sync::Arc;

use crate::array::RelayArray;

/// A typed builder for constructing RelayArrays.
pub enum ArrayBuilder {
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float64(Float64Builder),
    Bool(BooleanBuilder),
    String(StringBuilder),
}

impl ArrayBuilder {
    pub fn int32() -> Self {
        Self::Int32(Int32Builder::new())
    }
    pub fn int64() -> Self {
        Self::Int64(Int64Builder::new())
    }
    pub fn float64() -> Self {
        Self::Float64(Float64Builder::new())
    }
    pub fn boolean() -> Self {
        Self::Bool(BooleanBuilder::new())
    }
    pub fn string() -> Self {
        Self::String(StringBuilder::new())
    }

    pub fn append_i32(&mut self, value: i32) {
        match self {
            Self::Int32(b) => b.append_value(value),
            _ => panic!("not an Int32 builder"),
        }
    }

    pub fn append_i32_null(&mut self) {
        match self {
            Self::Int32(b) => b.append_null(),
            _ => panic!("not an Int32 builder"),
        }
    }

    pub fn append_f64(&mut self, value: f64) {
        match self {
            Self::Float64(b) => b.append_value(value),
            _ => panic!("not a Float64 builder"),
        }
    }

    pub fn append_f64_null(&mut self) {
        match self {
            Self::Float64(b) => b.append_null(),
            _ => panic!("not a Float64 builder"),
        }
    }

    pub fn append_bool(&mut self, value: bool) {
        match self {
            Self::Bool(b) => b.append_value(value),
            _ => panic!("not a Bool builder"),
        }
    }

    pub fn append_str(&mut self, value: &str) {
        match self {
            Self::String(b) => b.append_value(value),
            _ => panic!("not a String builder"),
        }
    }

    pub fn append_str_null(&mut self) {
        match self {
            Self::String(b) => b.append_null(),
            _ => panic!("not a String builder"),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Int32(b) => b.len(),
            Self::Int64(b) => b.len(),
            Self::Float64(b) => b.len(),
            Self::Bool(b) => b.len(),
            Self::String(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Finish building and return a RelayArray (allocates).
    pub fn finish(self) -> RelayArray {
        let array: ArrayRef = match self {
            Self::Int32(mut b) => Arc::new(b.finish()),
            Self::Int64(mut b) => Arc::new(b.finish()),
            Self::Float64(mut b) => Arc::new(b.finish()),
            Self::Bool(mut b) => Arc::new(b.finish()),
            Self::String(mut b) => Arc::new(b.finish()),
        };
        RelayArray::new(array)
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Self::Int32(_) => DataType::Int32,
            Self::Int64(_) => DataType::Int64,
            Self::Float64(_) => DataType::Float64,
            Self::Bool(_) => DataType::Boolean,
            Self::String(_) => DataType::Utf8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int32_builder() {
        let mut b = ArrayBuilder::int32();
        b.append_i32(1);
        b.append_i32(2);
        b.append_i32_null();
        b.append_i32(4);
        assert_eq!(b.len(), 4);
        let arr = b.finish();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(*arr.data_type(), DataType::Int32);
    }

    #[test]
    fn test_float64_builder() {
        let mut b = ArrayBuilder::float64();
        b.append_f64(1.1);
        b.append_f64(2.2);
        b.append_f64_null();
        let arr = b.finish();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        let f64_arr = arr.as_f64().unwrap();
        assert!((f64_arr.value(0) - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bool_builder() {
        let mut b = ArrayBuilder::boolean();
        b.append_bool(true);
        b.append_bool(false);
        let arr = b.finish();
        assert_eq!(arr.len(), 2);
        let bool_arr = arr.as_bool().unwrap();
        assert!(bool_arr.value(0));
        assert!(!bool_arr.value(1));
    }

    #[test]
    fn test_string_builder() {
        let mut b = ArrayBuilder::string();
        b.append_str("hello");
        b.append_str("world");
        b.append_str_null();
        let arr = b.finish();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        let str_arr = arr.as_str().unwrap();
        assert_eq!(str_arr.value(0), "hello");
    }

    #[test]
    fn test_builder_empty() {
        let b = ArrayBuilder::int32();
        assert!(b.is_empty());
        let arr = b.finish();
        assert!(arr.is_empty());
    }
}
