//! Compute operations on RelayArrays (zero-copy where possible).

use crate::array::RelayArray;
use arrow::array::BooleanArray;
use arrow::compute;

/// Filter an array using a boolean mask (allocates output, input untouched).
pub fn filter(array: &RelayArray, mask: &BooleanArray) -> relay_core::Result<RelayArray> {
    let filtered = compute::filter(array.as_arrow().as_ref(), mask)
        .map_err(|e| relay_core::RelayError::Arrow(e.to_string()))?;
    Ok(RelayArray::new(filtered))
}

/// Take elements by indices (allocates output).
pub fn take(array: &RelayArray, indices: &[u32]) -> relay_core::Result<RelayArray> {
    use arrow::array::UInt32Array;
    let indices_arr = UInt32Array::from(indices.to_vec());
    let taken = compute::take(array.as_arrow().as_ref(), &indices_arr, None)
        .map_err(|e| relay_core::RelayError::Arrow(e.to_string()))?;
    Ok(RelayArray::new(taken))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RelayArray;

    #[test]
    fn test_filter_i32() {
        let arr = RelayArray::from_i32(vec![1, 2, 3, 4, 5]);
        let mask = BooleanArray::from(vec![true, false, true, false, true]);
        let result = filter(&arr, &mask).unwrap();
        assert_eq!(result.len(), 3);
        let values = result.as_i32().unwrap();
        assert_eq!(values.value(0), 1);
        assert_eq!(values.value(1), 3);
        assert_eq!(values.value(2), 5);
    }

    #[test]
    fn test_filter_f64() {
        let arr = RelayArray::from_f64(vec![1.0, 2.0, 3.0]);
        let mask = BooleanArray::from(vec![false, true, true]);
        let result = filter(&arr, &mask).unwrap();
        assert_eq!(result.len(), 2);
        let values = result.as_f64().unwrap();
        assert!((values.value(0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filter_str() {
        let arr = RelayArray::from_str(vec!["a", "b", "c", "d"]);
        let mask = BooleanArray::from(vec![true, true, false, true]);
        let result = filter(&arr, &mask).unwrap();
        assert_eq!(result.len(), 3);
        let values = result.as_str().unwrap();
        assert_eq!(values.value(0), "a");
        assert_eq!(values.value(1), "b");
        assert_eq!(values.value(2), "d");
    }

    #[test]
    fn test_filter_empty_result() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let mask = BooleanArray::from(vec![false, false, false]);
        let result = filter(&arr, &mask).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_filter_all_selected() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let mask = BooleanArray::from(vec![true, true, true]);
        let result = filter(&arr, &mask).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_take_by_indices() {
        let arr = RelayArray::from_i32(vec![10, 20, 30, 40, 50]);
        let result = take(&arr, &[0, 2, 4]).unwrap();
        assert_eq!(result.len(), 3);
        let values = result.as_i32().unwrap();
        assert_eq!(values.value(0), 10);
        assert_eq!(values.value(1), 30);
        assert_eq!(values.value(2), 50);
    }

    #[test]
    fn test_take_with_duplicates() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let result = take(&arr, &[0, 0, 0]).unwrap();
        assert_eq!(result.len(), 3);
        let values = result.as_i32().unwrap();
        assert_eq!(values.value(0), 1);
        assert_eq!(values.value(1), 1);
        assert_eq!(values.value(2), 1);
    }

    #[test]
    fn test_take_empty() {
        let arr = RelayArray::from_i32(vec![1, 2, 3]);
        let result = take(&arr, &[]).unwrap();
        assert_eq!(result.len(), 0);
    }
}
