//! RelayRecordBatch — a collection of named RelayArrays (columnar record batch).
//!
//! Analogous to Arrow's RecordBatch but with Relay-specific features.
//! Supports zero-copy column selection and slicing.

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema as ArrowSchema};
use std::sync::Arc;

use crate::array::RelayArray;
use relay_core::{RelayError, Result};

/// A columnar record batch: multiple named arrays with the same length.
#[derive(Debug, Clone)]
pub struct RelayRecordBatch {
    schema: Arc<ArrowSchema>,
    columns: Vec<RelayArray>,
    num_rows: usize,
}

impl RelayRecordBatch {
    /// Create a new RecordBatch from column names and arrays.
    pub fn new(names: Vec<String>, columns: Vec<RelayArray>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(RelayError::Schema(format!(
                "column count mismatch: {} names but {} arrays",
                names.len(),
                columns.len()
            )));
        }

        if columns.is_empty() {
            return Ok(Self {
                schema: Arc::new(ArrowSchema::empty()),
                columns: vec![],
                num_rows: 0,
            });
        }

        let num_rows = columns[0].len();
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(RelayError::Schema(format!(
                    "row count mismatch: column '{}' has {} rows but expected {}",
                    names[i],
                    col.len(),
                    num_rows
                )));
            }
        }

        let fields: Vec<Field> = names
            .iter()
            .zip(columns.iter())
            .map(|(name, col)| Field::new(name, col.data_type().clone(), col.has_nulls()))
            .collect();

        let schema = Arc::new(ArrowSchema::new(fields));

        Ok(Self {
            schema,
            columns,
            num_rows,
        })
    }

    /// Create from an Arrow RecordBatch (zero-copy).
    pub fn from_arrow(rb: RecordBatch) -> Self {
        let schema = rb.schema();
        let num_rows = rb.num_rows();
        let columns: Vec<RelayArray> = (0..rb.num_columns())
            .map(|i| RelayArray::new(rb.column(i).clone()))
            .collect();
        Self {
            schema,
            columns,
            num_rows,
        }
    }

    /// Convert to Arrow RecordBatch (zero-copy).
    pub fn as_arrow_recordbatch(&self) -> RecordBatch {
        let arrays: Vec<ArrayRef> = self.columns.iter().map(|c| c.as_arrow().clone()).collect();
        RecordBatch::try_new(self.schema.clone(), arrays).expect("schema already validated")
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get the schema.
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }

    /// Get a column by index (zero-copy).
    pub fn column(&self, index: usize) -> Result<&RelayArray> {
        self.columns
            .get(index)
            .ok_or(RelayError::OutOfBounds {
                index,
                len: self.columns.len(),
            })
    }

    /// Get a column by name (zero-copy).
    pub fn column_by_name(&self, name: &str) -> Result<&RelayArray> {
        let idx = self
            .schema
            .fields()
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| RelayError::Schema(format!("column not found: {}", name)))?;
        Ok(&self.columns[idx])
    }

    /// Get all column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect()
    }

    /// Select a subset of columns (zero-copy).
    pub fn select(&self, names: &[&str]) -> Result<RelayRecordBatch> {
        let mut selected_names = Vec::with_capacity(names.len());
        let mut selected_cols = Vec::with_capacity(names.len());
        for name in names {
            let col = self.column_by_name(name)?;
            selected_names.push(name.to_string());
            selected_cols.push(col.clone());
        }
        RelayRecordBatch::new(selected_names, selected_cols)
    }

    /// Slice rows (zero-copy — returns views into same memory).
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let columns: Vec<RelayArray> = self
            .columns
            .iter()
            .map(|c| c.slice(offset, length))
            .collect();
        Self {
            schema: self.schema.clone(),
            columns,
            num_rows: length,
        }
    }

    /// Total memory size of all columns.
    pub fn memory_size(&self) -> usize {
        self.columns.iter().map(|c| c.memory_size()).sum()
    }

    /// Check if any column has nulls.
    pub fn has_nulls(&self) -> bool {
        self.columns.iter().any(|c| c.has_nulls())
    }

    /// Total null count across all columns.
    pub fn null_count(&self) -> usize {
        self.columns.iter().map(|c| c.null_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    fn sample_batch() -> RelayRecordBatch {
        RelayRecordBatch::new(
            vec!["name".into(), "age".into(), "salary".into()],
            vec![
                RelayArray::from_strs(vec!["alice", "bob", "charlie"]),
                RelayArray::from_i32(vec![25, 30, 35]),
                RelayArray::from_f64(vec![50000.0, 60000.0, 70000.0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_create_batch() {
        let batch = sample_batch();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn test_column_by_name() {
        let batch = sample_batch();
        let age = batch.column_by_name("age").unwrap();
        assert_eq!(age.len(), 3);
        assert_eq!(*age.data_type(), DataType::Int32);
    }

    #[test]
    fn test_column_by_index() {
        let batch = sample_batch();
        let salary = batch.column(2).unwrap();
        assert_eq!(salary.len(), 3);
        assert_eq!(*salary.data_type(), DataType::Float64);
    }

    #[test]
    fn test_column_not_found() {
        let batch = sample_batch();
        assert!(batch.column_by_name("nonexistent").is_err());
        assert!(batch.column(10).is_err());
    }

    #[test]
    fn test_select_columns() {
        let batch = sample_batch();
        let selected = batch.select(&["name", "salary"]).unwrap();
        assert_eq!(selected.num_columns(), 2);
        assert_eq!(selected.column_names(), vec!["name", "salary"]);
    }

    #[test]
    fn test_select_invalid_column() {
        let batch = sample_batch();
        assert!(batch.select(&["name", "nonexistent"]).is_err());
    }

    #[test]
    fn test_slice_zero_copy() {
        let batch = sample_batch();
        let sliced = batch.slice(1, 2);
        assert_eq!(sliced.num_rows(), 2);
        let ages = sliced.column_by_name("age").unwrap();
        assert_eq!(ages.len(), 2);
        let i32_arr = ages.as_i32().unwrap();
        assert_eq!(i32_arr.value(0), 30);
        assert_eq!(i32_arr.value(1), 35);
    }

    #[test]
    fn test_memory_size() {
        let batch = sample_batch();
        let size = batch.memory_size();
        assert!(size > 0);
    }

    #[test]
    fn test_has_nulls() {
        let batch = sample_batch();
        assert!(!batch.has_nulls());
        assert_eq!(batch.null_count(), 0);
    }

    #[test]
    fn test_nullable_batch() {
        let batch = RelayRecordBatch::new(
            vec!["value".into()],
            vec![RelayArray::from_i32_nullable(vec![Some(1), None, Some(3)])],
        )
        .unwrap();
        assert!(batch.has_nulls());
        assert_eq!(batch.null_count(), 1);
    }

    #[test]
    fn test_row_count_mismatch() {
        let result = RelayRecordBatch::new(
            vec!["a".into(), "b".into()],
            vec![
                RelayArray::from_i32(vec![1, 2, 3]),
                RelayArray::from_i32(vec![1, 2]), // mismatch!
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_batch() {
        let batch = RelayRecordBatch::new(vec![], vec![]).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 0);
    }

    #[test]
    fn test_column_names() {
        let batch = sample_batch();
        assert_eq!(batch.column_names(), vec!["name", "age", "salary"]);
    }

    #[test]
    fn test_from_arrow_recordbatch() {
        use arrow::array::Int32Array;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            DataType::Int32,
            false,
        )]));
        let arr = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let rb = RecordBatch::try_new(schema, vec![arr as ArrayRef]).unwrap();
        let relay_batch = RelayRecordBatch::from_arrow(rb);
        assert_eq!(relay_batch.num_rows(), 3);
        assert_eq!(relay_batch.num_columns(), 1);
    }

    #[test]
    fn test_to_arrow_recordbatch() {
        let batch = sample_batch();
        let arrow_rb = batch.as_arrow_recordbatch();
        assert_eq!(arrow_rb.num_rows(), 3);
        assert_eq!(arrow_rb.num_columns(), 3);
    }

    #[test]
    fn test_arrow_roundtrip() {
        let original = sample_batch();
        let arrow_rb = original.as_arrow_recordbatch();
        let restored = RelayRecordBatch::from_arrow(arrow_rb);
        assert_eq!(restored.num_rows(), original.num_rows());
        assert_eq!(restored.num_columns(), original.num_columns());
    }

    #[test]
    fn test_large_batch() {
        let n = 1_000_000;
        let batch = RelayRecordBatch::new(
            vec!["id".into(), "value".into()],
            vec![
                RelayArray::from_i32((0..n as i32).collect()),
                RelayArray::from_f64((0..n).map(|i| i as f64).collect()),
            ],
        )
        .unwrap();
        assert_eq!(batch.num_rows(), n);
        assert!(batch.memory_size() > 0);
    }
}
