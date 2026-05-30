//! Arrow C Data Interface (FFI) for zero-copy data exchange.
//!
//! Implements the Arrow PyCapsule Interface for zero-copy FFI between
//! Relay and any Arrow-compatible library (PyArrow, Polars, pandas, NumPy).
//!
//! Reference: https://arrow.apache.org/docs/format/CDataInterface.html

use arrow::array::{make_array, Array, ArrayData, ArrayRef, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use std::sync::Arc;

use crate::array::RelayArray;
use crate::recordbatch::RelayRecordBatch;
use relay_core::{RelayError, Result};

/// Export a RelayArray as Arrow C Data Interface (zero-copy FFI).
/// Returns (FFI_ArrowArray, FFI_ArrowSchema) pair.
pub fn export_array(array: &RelayArray) -> Result<(FFI_ArrowArray, FFI_ArrowSchema)> {
    let data = array.as_arrow().to_data();
    to_ffi(&data).map_err(|e| RelayError::Arrow(format!("FFI export failed: {}", e)))
}

/// Import an ArrowArray from FFI into a RelayArray (zero-copy).
pub fn import_array(array: FFI_ArrowArray, schema: FFI_ArrowSchema) -> Result<RelayArray> {
    let data = unsafe { from_ffi(array, &schema) }
        .map_err(|e| RelayError::Arrow(format!("FFI import failed: {}", e)))?;
    Ok(RelayArray::new(make_array(data)))
}

/// Export a RecordBatch as Arrow C Data Interface.
/// Converts the batch into a StructArray, then exports via FFI.
pub fn export_recordbatch(batch: &RelayRecordBatch) -> Result<(FFI_ArrowArray, FFI_ArrowSchema)> {
    let rb = batch.as_arrow_recordbatch();
    let struct_array = StructArray::from(rb.clone());
    let data = struct_array.to_data();
    to_ffi(&data).map_err(|e| RelayError::Arrow(format!("FFI export failed: {}", e)))
}

/// Import a RecordBatch from Arrow C Data Interface (zero-copy).
pub fn import_recordbatch(
    array: FFI_ArrowArray,
    schema: FFI_ArrowSchema,
) -> Result<RelayRecordBatch> {
    let data = unsafe { from_ffi(array, &schema) }
        .map_err(|e| RelayError::Arrow(format!("FFI import failed: {}", e)))?;
    let struct_array: StructArray = data.into();
    let rb = RecordBatch::from(struct_array);
    Ok(RelayRecordBatch::from_arrow(rb))
}

/// Get the raw buffer pointers from an ArrayData.
/// Used to verify zero-copy: comparing pointer addresses before/after.
pub fn buffer_ptrs(array: &ArrayRef) -> Vec<usize> {
    let data = array.to_data();
    data.buffers().iter().map(|b| b.as_ptr() as usize).collect()
}

/// Check if two ArrayRefs share the same underlying memory (zero-copy).
/// This checks if any buffer ranges overlap, not just pointer equality.
pub fn shares_memory(a: &ArrayRef, b: &ArrayRef) -> bool {
    let a_ranges: Vec<(usize, usize)> = a
        .to_data()
        .buffers()
        .iter()
        .map(|b| (b.as_ptr() as usize, b.as_ptr() as usize + b.len()))
        .collect();
    let b_ranges: Vec<(usize, usize)> = b
        .to_data()
        .buffers()
        .iter()
        .map(|b| (b.as_ptr() as usize, b.as_ptr() as usize + b.len()))
        .collect();
    // Check if any ranges overlap
    for &(a_start, a_end) in &a_ranges {
        for &(b_start, b_end) in &b_ranges {
            if a_start < b_end && b_start < a_end {
                return true;
            }
        }
    }
    false
}

/// Verify that an array's data pointer matches a given address.
/// Used in tests to confirm zero-copy (no new allocation).
pub fn data_ptr(array: &ArrayRef) -> usize {
    let data = array.to_data();
    if data.buffers().is_empty() {
        0
    } else {
        data.buffers()[0].as_ptr() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, StringArray};

    #[test]
    fn test_export_import_roundtrip_i32() {
        let original = RelayArray::from_i32(vec![1, 2, 3, 4, 5]);
        let (array, schema) = export_array(&original).unwrap();
        let imported = import_array(array, schema).unwrap();
        assert_eq!(imported.len(), 5);
        assert_eq!(*imported.data_type(), DataType::Int32);
    }

    #[test]
    fn test_export_import_roundtrip_f64() {
        let original = RelayArray::from_f64(vec![1.1, 2.2, 3.3]);
        let (array, schema) = export_array(&original).unwrap();
        let imported = import_array(array, schema).unwrap();
        assert_eq!(imported.len(), 3);
        assert_eq!(*imported.data_type(), DataType::Float64);
    }

    #[test]
    fn test_export_import_roundtrip_str() {
        let original = RelayArray::from_str(vec!["hello", "world"]);
        let (array, schema) = export_array(&original).unwrap();
        let imported = import_array(array, schema).unwrap();
        assert_eq!(imported.len(), 2);
    }

    #[test]
    fn test_buffer_ptrs() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let ptrs = buffer_ptrs(arr.as_arrow());
        assert!(!ptrs.is_empty());
        assert!(ptrs[0] != 0);
    }

    #[test]
    fn test_shares_memory_same_array() {
        let arr = RelayArray::from_i32(vec![1, 2, 3, 4, 5]);
        let sliced = arr.slice(1, 3);
        assert!(shares_memory(arr.as_arrow(), sliced.as_arrow()));
    }

    #[test]
    fn test_shares_memory_different_array() {
        let a = RelayArray::from_i32(vec![1, 2, 3]);
        let b = RelayArray::from_i32(vec![4, 5, 6]);
        assert!(!shares_memory(a.as_arrow(), b.as_arrow()));
    }

    #[test]
    fn test_data_ptr_nonzero() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        assert!(data_ptr(arr.as_arrow()) != 0);
    }

    #[test]
    fn test_export_import_recordbatch() {
        let names = RelayArray::from_str(vec!["alice", "bob"]);
        let ages = RelayArray::from_i32(vec![25, 30]);
        let batch = RelayRecordBatch::new(
            vec!["name".to_string(), "age".to_string()],
            vec![names, ages],
        )
        .unwrap();
        let (array, schema) = export_recordbatch(&batch).unwrap();
        let imported = import_recordbatch(array, schema).unwrap();
        assert_eq!(imported.num_rows(), 2);
        assert_eq!(imported.num_columns(), 2);
    }
}
