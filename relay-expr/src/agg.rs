//! SIMD-optimized aggregation using Arrow compute kernels
//!
//! Uses arrow::compute::sum, min, max for maximum throughput.

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::DataType;

use relay_core::RelayError;

/// Aggregation operation type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

impl std::fmt::Display for AggOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggOp::Sum => write!(f, "SUM"),
            AggOp::Mean => write!(f, "MEAN"),
            AggOp::Min => write!(f, "MIN"),
            AggOp::Max => write!(f, "MAX"),
            AggOp::Count => write!(f, "COUNT"),
        }
    }
}

/// Aggregate an array using SIMD-optimized Arrow kernels
pub fn aggregate_array(
    array: &dyn Array,
    op: AggOp,
) -> Result<AggResult, RelayError> {
    match op {
        AggOp::Sum => aggregate_sum(array),
        AggOp::Mean => aggregate_mean(array),
        AggOp::Min => aggregate_min(array),
        AggOp::Max => aggregate_max(array),
        AggOp::Count => Ok(AggResult::Int64(array.len() as i64 - array.null_count() as i64)),
    }
}

/// Aggregate result enum
#[derive(Debug, Clone)]
pub enum AggResult {
    Int64(i64),
    Float64(f64),
    Null,
}

impl std::fmt::Display for AggResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggResult::Int64(v) => write!(f, "{}", v),
            AggResult::Float64(v) => write!(f, "{}", v),
            AggResult::Null => write!(f, "NULL"),
        }
    }
}

fn aggregate_sum(array: &dyn Array) -> Result<AggResult, RelayError> {
    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let sum: i64 = arrow::compute::sum(arr).unwrap_or(0);
            Ok(AggResult::Int64(sum))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let sum: f64 = arrow::compute::sum(arr).unwrap_or(0.0);
            Ok(AggResult::Float64(sum))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            let sum: i64 = arrow::compute::sum(arr).map(|v: i32| v as i64).unwrap_or(0);
            Ok(AggResult::Int64(sum))
        }
        dt => Err(RelayError::Expr(format!("Sum not supported for {:?}", dt))),
    }
}

fn aggregate_mean(array: &dyn Array) -> Result<AggResult, RelayError> {
    let count = array.len() - array.null_count();
    if count == 0 {
        return Ok(AggResult::Null);
    }

    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let sum: i64 = arrow::compute::sum(arr).unwrap_or(0);
            Ok(AggResult::Float64(sum as f64 / count as f64))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let sum: f64 = arrow::compute::sum(arr).unwrap_or(0.0);
            Ok(AggResult::Float64(sum / count as f64))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            let sum: i64 = arrow::compute::sum(arr).map(|v: i32| v as i64).unwrap_or(0);
            Ok(AggResult::Float64(sum as f64 / count as f64))
        }
        dt => Err(RelayError::Expr(format!("Mean not supported for {:?}", dt))),
    }
}

fn aggregate_min(array: &dyn Array) -> Result<AggResult, RelayError> {
    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let min = arrow::compute::min(arr);
            match min {
                Some(v) => Ok(AggResult::Int64(v)),
                None => Ok(AggResult::Null),
            }
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let min = arrow::compute::min(arr);
            match min {
                Some(v) => Ok(AggResult::Float64(v)),
                None => Ok(AggResult::Null),
            }
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            let min = arrow::compute::min(arr);
            match min {
                Some(v) => Ok(AggResult::Int64(v as i64)),
                None => Ok(AggResult::Null),
            }
        }
        dt => Err(RelayError::Expr(format!("Min not supported for {:?}", dt))),
    }
}

fn aggregate_max(array: &dyn Array) -> Result<AggResult, RelayError> {
    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let max = arrow::compute::max(arr);
            match max {
                Some(v) => Ok(AggResult::Int64(v)),
                None => Ok(AggResult::Null),
            }
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let max = arrow::compute::max(arr);
            match max {
                Some(v) => Ok(AggResult::Float64(v)),
                None => Ok(AggResult::Null),
            }
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            let max = arrow::compute::max(arr);
            match max {
                Some(v) => Ok(AggResult::Int64(v as i64)),
                None => Ok(AggResult::Null),
            }
        }
        dt => Err(RelayError::Expr(format!("Max not supported for {:?}", dt))),
    }
}
