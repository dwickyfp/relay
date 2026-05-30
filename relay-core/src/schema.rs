//! Schema definitions for Relay.

use crate::error::{RelayError, Result};
use crate::types::RelayType;
use std::collections::HashMap;

/// A field in a Relay schema.
#[derive(Debug, Clone, PartialEq)]
pub struct RelayField {
    pub name: String,
    pub dtype: RelayType,
    pub nullable: bool,
    pub metadata: HashMap<String, String>,
}

impl RelayField {
    pub fn new(name: impl Into<String>, dtype: RelayType) -> Self {
        Self {
            name: name.into(),
            dtype,
            nullable: true,
            metadata: HashMap::new(),
        }
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A schema describes the structure of a DataFrame or RecordBatch.
#[derive(Debug, Clone, PartialEq)]
pub struct RelaySchema {
    fields: Vec<RelayField>,
    metadata: HashMap<String, String>,
    /// Fast lookup: column name -> index
    index: HashMap<String, usize>,
}

impl RelaySchema {
    pub fn new(fields: Vec<RelayField>) -> Result<Self> {
        let mut index = HashMap::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            if index.contains_key(&field.name) {
                return Err(RelayError::Schema(format!(
                    "duplicate column name: {}",
                    field.name
                )));
            }
            index.insert(field.name.clone(), i);
        }
        Ok(Self {
            fields,
            metadata: HashMap::new(),
            index,
        })
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn field(&self, index: usize) -> Result<&RelayField> {
        self.fields
            .get(index)
            .ok_or(RelayError::OutOfBounds {
                index,
                len: self.fields.len(),
            })
    }

    pub fn field_by_name(&self, name: &str) -> Result<&RelayField> {
        let idx = self.column_index(name)?;
        Ok(&self.fields[idx])
    }

    pub fn column_index(&self, name: &str) -> Result<usize> {
        self.index.get(name).copied().ok_or_else(|| {
            RelayError::Schema(format!(
                "column not found: {}. Available: {:?}",
                name,
                self.field_names()
            ))
        })
    }

    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    pub fn fields(&self) -> &[RelayField] {
        &self.fields
    }
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn project(&self, names: &[&str]) -> Result<RelaySchema> {
        let fields: Result<Vec<RelayField>> = names
            .iter()
            .map(|name| self.field_by_name(name).cloned())
            .collect();
        RelaySchema::new(fields?)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> RelaySchema {
        RelaySchema::new(vec![
            RelayField::new("name", RelayType::Utf8).with_nullable(false),
            RelayField::new("age", RelayType::Int32),
            RelayField::new("salary", RelayType::Float64),
            RelayField::new("active", RelayType::Boolean),
        ])
        .unwrap()
    }

    #[test]
    fn test_schema_creation() {
        let schema = sample_schema();
        assert_eq!(schema.len(), 4);
        assert!(!schema.is_empty());
    }

    #[test]
    fn test_field_access() {
        let schema = sample_schema();
        let f = schema.field(0).unwrap();
        assert_eq!(f.name, "name");
        assert_eq!(f.dtype, RelayType::Utf8);
        assert!(!f.nullable);
    }

    #[test]
    fn test_field_by_name() {
        let schema = sample_schema();
        let f = schema.field_by_name("age").unwrap();
        assert_eq!(f.dtype, RelayType::Int32);
        assert!(f.nullable);
    }

    #[test]
    fn test_column_index() {
        let schema = sample_schema();
        assert_eq!(schema.column_index("name").unwrap(), 0);
        assert_eq!(schema.column_index("salary").unwrap(), 2);
        assert!(schema.column_index("nonexistent").is_err());
    }

    #[test]
    fn test_duplicate_column_error() {
        let result = RelaySchema::new(vec![
            RelayField::new("id", RelayType::Int32),
            RelayField::new("id", RelayType::Utf8),
        ]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("duplicate"));
    }

    #[test]
    fn test_project() {
        let schema = sample_schema();
        let projected = schema.project(&["name", "salary"]).unwrap();
        assert_eq!(projected.len(), 2);
        assert_eq!(projected.field(0).unwrap().name, "name");
        assert_eq!(projected.field(1).unwrap().name, "salary");
    }

    #[test]
    fn test_project_invalid_column() {
        let schema = sample_schema();
        assert!(schema.project(&["name", "nonexistent"]).is_err());
    }

    #[test]
    fn test_contains() {
        let schema = sample_schema();
        assert!(schema.contains("name"));
        assert!(!schema.contains("nonexistent"));
    }

    #[test]
    fn test_field_names() {
        let schema = sample_schema();
        assert_eq!(
            schema.field_names(),
            vec!["name", "age", "salary", "active"]
        );
    }

    #[test]
    fn test_empty_schema() {
        let schema = RelaySchema::new(vec![]).unwrap();
        assert!(schema.is_empty());
        assert_eq!(schema.len(), 0);
    }

    #[test]
    fn test_oob_field_access() {
        let schema = sample_schema();
        let result = schema.field(10);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("out of bounds"));
    }

    #[test]
    fn test_field_metadata() {
        let field = RelayField::new("score", RelayType::Float64)
            .with_metadata("description", "test score 0-100")
            .with_metadata("source", "exam_system");
        assert_eq!(
            field.metadata.get("description").unwrap(),
            "test score 0-100"
        );
        assert_eq!(field.metadata.get("source").unwrap(), "exam_system");
    }
}
