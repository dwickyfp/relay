//! High-performance JSON/NDJSON reader with parallel parsing.
//!
//! Architecture:
//! 1. mmap file → zero-copy read
//! 2. Auto-detect format (JSON array vs NDJSON)
//! 3. Split into record boundaries (parallelizable)
//! 4. Schema inference from sample rows
//! 5. Each chunk parsed independently via Rayon
//! 6. Direct Arrow array construction (no intermediate row representation)
//!
//! Supported types: Int64, Float64, Boolean, Utf8 (string), Null.
//! Nested objects/arrays are serialized as JSON strings.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, NullBuilder, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow_array::RecordBatch;
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;

use relay_core::{RelayError, Result};

/// JSON input format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonFormat {
    /// Auto-detect from file content.
    Auto,
    /// JSON array: `[{...}, {...}, ...]`
    JsonArray,
    /// Newline-delimited JSON: one JSON object per line.
    Ndjson,
}

/// Configuration for JSON reading.
#[derive(Debug, Clone)]
pub struct JsonReadOptions {
    /// Input format (default: Auto).
    pub format: JsonFormat,
    /// Number of rows to sample for schema inference.
    pub infer_schema_rows: usize,
    /// Target chunk size in rows for parallel parsing.
    pub chunk_size: usize,
}

impl Default for JsonReadOptions {
    fn default() -> Self {
        Self {
            format: JsonFormat::Auto,
            infer_schema_rows: 1024,
            chunk_size: 8192,
        }
    }
}

/// A high-performance JSON/NDJSON reader.
pub struct JsonReader {
    mmap: Mmap,
    schema: SchemaRef,
    options: JsonReadOptions,
    format: JsonFormat,
    /// Byte offsets of each record's start and end.
    records: Vec<(usize, usize)>,
    /// Ordered column names (from schema inference).
    column_names: Vec<String>,
}

impl JsonReader {
    /// Open a JSON file and infer schema.
    pub fn open(path: &Path, options: JsonReadOptions) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| RelayError::Io(e))?;

        if mmap.is_empty() {
            return Err(RelayError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty JSON file",
            )));
        }

        // Detect format
        let format = if options.format == JsonFormat::Auto {
            detect_format(&mmap)
        } else {
            options.format
        };

        // Find record boundaries
        let records = match format {
            JsonFormat::Ndjson => find_ndjson_records(&mmap),
            JsonFormat::JsonArray => find_json_array_records(&mmap),
            JsonFormat::Auto => unreachable!(),
        };

        if records.is_empty() {
            return Err(RelayError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no JSON records found",
            )));
        }

        // Infer schema from sample rows
        let sample_end = std::cmp::min(options.infer_schema_rows, records.len());
        let schema = infer_schema(&mmap, &records[..sample_end])?;
        let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        Ok(Self {
            mmap,
            schema,
            options,
            format,
            records,
            column_names,
        })
    }

    /// Get the Arrow schema.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Total number of records.
    pub fn num_rows(&self) -> usize {
        self.records.len()
    }

    /// Column names from schema inference.
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Read all rows into a single RecordBatch.
    pub fn read_all(&self) -> Result<RecordBatch> {
        if self.records.is_empty() {
            return self.empty_batch();
        }

        // Split into parallel chunks
        let chunks: Vec<(usize, usize)> = (0..self.records.len())
            .step_by(self.options.chunk_size)
            .map(|start| {
                let end = std::cmp::min(start + self.options.chunk_size, self.records.len());
                (start, end)
            })
            .collect();

        let batches: Vec<RecordBatch> = chunks
            .par_iter()
            .map(|&(start, end)| self.parse_chunk(start, end))
            .collect::<Result<Vec<_>>>()?;

        if batches.is_empty() {
            self.empty_batch()
        } else if batches.len() == 1 {
            Ok(batches.into_iter().next().unwrap())
        } else {
            arrow::compute::concat_batches(&self.schema, &batches)
                .map_err(|e| RelayError::Arrow(format!("concat batches: {}", e)))
        }
    }

    /// Read only specified columns.
    pub fn read_columns(&self, columns: &[&str]) -> Result<RecordBatch> {
        let all = self.read_all()?;

        let col_indices: Vec<usize> = columns
            .iter()
            .map(|name| {
                self.column_names
                    .iter()
                    .position(|n| n == name)
                    .ok_or_else(|| {
                        RelayError::Arrow(format!("column '{}' not found", name))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let projected_fields: Vec<Field> = col_indices
            .iter()
            .map(|&i| self.schema.field(i).clone())
            .collect();
        let projected_cols: Vec<ArrayRef> =
            col_indices.iter().map(|&i| all.column(i).clone()).collect();
        let projected_schema = Arc::new(Schema::new(projected_fields));

        RecordBatch::try_new(projected_schema, projected_cols)
            .map_err(|e| RelayError::Arrow(format!("projection: {}", e)))
    }

    fn parse_chunk(&self, start: usize, end: usize) -> Result<RecordBatch> {
        let num_rows = end - start;
        let num_cols = self.schema.fields().len();

        // Create builders for each column
        let mut builders: Vec<Box<dyn JsonColumnBuilder>> = self
            .schema
            .fields()
            .iter()
            .map(|f| make_builder(f.data_type(), num_rows))
            .collect();

        // Parse each row
        for row_idx in start..end {
            let (rec_start, rec_end) = self.records[row_idx];
            let json_bytes = &self.mmap[rec_start..rec_end];

            let obj: Value = serde_json::from_slice(json_bytes).map_err(|e| {
                RelayError::Arrow(format!(
                    "JSON parse error at row {}: {}",
                    row_idx, e
                ))
            })?;

            if let Value::Object(map) = &obj {
                for (col_idx, field) in self.schema.fields().iter().enumerate() {
                    let value = map.get(field.name());
                    builders[col_idx].append_value(value);
                }
            } else {
                // Non-object row: append nulls
                for builder in builders.iter_mut() {
                    builder.append_null();
                }
            }
        }

        // Finish builders
        let arrays: Vec<ArrayRef> = builders.iter_mut().map(|b| b.finish()).collect();

        RecordBatch::try_new(self.schema.clone(), arrays)
            .map_err(|e| RelayError::Arrow(format!("build batch: {}", e)))
    }

    fn empty_batch(&self) -> Result<RecordBatch> {
        let arrays: Vec<ArrayRef> = self
            .schema
            .fields()
            .iter()
            .map(|f| arrow::array::new_empty_array(f.data_type()))
            .collect();
        RecordBatch::try_new(self.schema.clone(), arrays)
            .map_err(|e| RelayError::Arrow(format!("empty batch: {}", e)))
    }
}

// ─── Format Detection ─────────────────────────────────────────────

/// Detect whether the data is a JSON array or NDJSON.
fn detect_format(data: &[u8]) -> JsonFormat {
    // Skip whitespace and BOM
    let trimmed = data
        .iter()
        .skip_while(|&&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0xEF || b == 0xBB || b == 0xBF);

    match trimmed.clone().next() {
        Some(b'[') => JsonFormat::JsonArray,
        _ => JsonFormat::Ndjson,
    }
}

/// Find record boundaries in NDJSON format.
/// Each non-empty line is a record.
fn find_ndjson_records(data: &[u8]) -> Vec<(usize, usize)> {
    let mut records = Vec::new();
    let mut line_start = 0;

    for (i, &byte) in data.iter().enumerate() {
        if byte == b'\n' {
            let line_end = if i > 0 && data[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            if line_end > line_start {
                // Check it's not just whitespace
                let slice = &data[line_start..line_end];
                if !slice.iter().all(|&b| b == b' ' || b == b'\t') {
                    records.push((line_start, line_end));
                }
            }
            line_start = i + 1;
        }
    }

    // Handle last line without trailing newline
    if line_start < data.len() {
        let slice = &data[line_start..];
        if !slice.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r') {
            records.push((line_start, data.len()));
        }
    }

    records
}

/// Find record boundaries in a JSON array: `[{...}, {...}, ...]`
fn find_json_array_records(data: &[u8]) -> Vec<(usize, usize)> {
    let mut records = Vec::new();

    // Find the opening '['
    let array_start = match data.iter().position(|&b| b == b'[') {
        Some(pos) => pos + 1,
        None => return records,
    };

    // Find the closing ']'
    let array_end = match data.iter().rposition(|&b| b == b']') {
        Some(pos) => pos,
        None => return records,
    };

    // Walk through the array content, tracking brace depth
    let mut i = array_start;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut obj_start = None;

    while i < array_end {
        let byte = data[i];

        if escape {
            escape = false;
            i += 1;
            continue;
        }

        if byte == b'\\' && in_string {
            escape = true;
            i += 1;
            continue;
        }

        if byte == b'"' {
            in_string = !in_string;
            i += 1;
            continue;
        }

        if in_string {
            i += 1;
            continue;
        }

        match byte {
            b'{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        records.push((start, i + 1));
                    }
                }
            }
            _ => {}
        }

        i += 1;
    }

    records
}

// ─── Schema Inference ─────────────────────────────────────────────

/// Infer Arrow schema by sampling JSON records.
fn infer_schema(data: &[u8], records: &[(usize, usize)]) -> Result<SchemaRef> {
    // Collect all keys and their types from sample rows
    let mut key_types: HashMap<String, Vec<DataType>> = HashMap::new();
    let mut key_order: Vec<String> = Vec::new(); // preserve insertion order

    for &(start, end) in records {
        let json_bytes = &data[start..end];
        let value: Value = serde_json::from_slice(json_bytes).map_err(|e| {
            RelayError::Arrow(format!("JSON parse error during schema inference: {}", e))
        })?;

        if let Value::Object(map) = value {
            for (key, val) in &map {
                let dt = json_type_to_arrow(val);
                let entry = key_types.entry(key.clone()).or_default();
                entry.push(dt);
                if !key_order.contains(key) {
                    key_order.push(key.clone());
                }
            }
        }
    }

    // Build schema: pick the widest type for each key
    let fields: Vec<Field> = key_order
        .iter()
        .map(|key| {
            let types = key_types.get(key).unwrap();
            let dt = widen_types(types);
            Field::new(key, dt, true)
        })
        .collect();

    Ok(Arc::new(Schema::new(fields)))
}

/// Map a JSON value to an Arrow DataType.
fn json_type_to_arrow(value: &Value) -> DataType {
    match value {
        Value::Null => DataType::Null,
        Value::Bool(_) => DataType::Boolean,
        Value::Number(n) => {
            if n.is_i64() {
                DataType::Int64
            } else {
                DataType::Float64
            }
        }
        Value::String(_) => DataType::Utf8,
        // Arrays and objects become JSON strings
        Value::Array(_) | Value::Object(_) => DataType::Utf8,
    }
}

/// Widen a set of types to the most general one.
fn widen_types(types: &[DataType]) -> DataType {
    let mut result: Option<DataType> = None;
    for dt in types {
        result = Some(match result {
            None => dt.clone(),
            Some(prev) => widen(prev, dt.clone()),
        });
    }
    result.unwrap_or(DataType::Utf8)
}

fn widen(a: DataType, b: DataType) -> DataType {
    match (a, b) {
        (DataType::Null, other) | (other, DataType::Null) => other,
        (DataType::Int64, DataType::Int64) => DataType::Int64,
        (DataType::Float64, DataType::Float64) => DataType::Float64,
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
            DataType::Float64
        }
        (DataType::Boolean, DataType::Boolean) => DataType::Boolean,
        _ => DataType::Utf8,
    }
}

// ─── Column Builders ──────────────────────────────────────────────

trait JsonColumnBuilder: Send {
    fn append_value(&mut self, value: Option<&Value>);
    fn append_null(&mut self);
    fn finish(&mut self) -> ArrayRef;
}

struct Int64JsonBuilder(Int64Builder);
struct Float64JsonBuilder(Float64Builder);
struct BoolJsonBuilder(BooleanBuilder);
struct Utf8JsonBuilder(StringBuilder);
struct NullJsonBuilder(NullBuilder);

impl JsonColumnBuilder for Int64JsonBuilder {
    fn append_value(&mut self, value: Option<&Value>) {
        match value {
            Some(Value::Number(n)) => {
                if let Some(v) = n.as_i64() {
                    self.0.append_value(v);
                } else if let Some(v) = n.as_f64() {
                    self.0.append_value(v as i64);
                } else {
                    self.0.append_null();
                }
            }
            Some(Value::String(s)) => {
                match s.parse::<i64>() {
                    Ok(v) => self.0.append_value(v),
                    Err(_) => self.0.append_null(),
                }
            }
            _ => self.0.append_null(),
        }
    }
    fn append_null(&mut self) {
        self.0.append_null();
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }
}

impl JsonColumnBuilder for Float64JsonBuilder {
    fn append_value(&mut self, value: Option<&Value>) {
        match value {
            Some(Value::Number(n)) => {
                if let Some(v) = n.as_f64() {
                    self.0.append_value(v);
                } else {
                    self.0.append_null();
                }
            }
            Some(Value::String(s)) => {
                match s.parse::<f64>() {
                    Ok(v) => self.0.append_value(v),
                    Err(_) => self.0.append_null(),
                }
            }
            _ => self.0.append_null(),
        }
    }
    fn append_null(&mut self) {
        self.0.append_null();
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }
}

impl JsonColumnBuilder for BoolJsonBuilder {
    fn append_value(&mut self, value: Option<&Value>) {
        match value {
            Some(Value::Bool(b)) => self.0.append_value(*b),
            Some(Value::String(s)) => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" => self.0.append_value(true),
                "false" | "0" | "no" => self.0.append_value(false),
                _ => self.0.append_null(),
            },
            _ => self.0.append_null(),
        }
    }
    fn append_null(&mut self) {
        self.0.append_null();
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }
}

impl JsonColumnBuilder for Utf8JsonBuilder {
    fn append_value(&mut self, value: Option<&Value>) {
        match value {
            Some(Value::String(s)) => self.0.append_value(s),
            Some(Value::Null) | None => self.0.append_null(),
            Some(other) => self.0.append_value(other.to_string()),
        }
    }
    fn append_null(&mut self) {
        self.0.append_null();
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }
}

impl JsonColumnBuilder for NullJsonBuilder {
    fn append_value(&mut self, _value: Option<&Value>) {
        self.0.append_null();
    }
    fn append_null(&mut self) {
        self.0.append_null();
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }
}

fn make_builder(dt: &DataType, capacity: usize) -> Box<dyn JsonColumnBuilder> {
    match dt {
        DataType::Int64 => Box::new(Int64JsonBuilder(Int64Builder::with_capacity(capacity))),
        DataType::Float64 => {
            Box::new(Float64JsonBuilder(Float64Builder::with_capacity(capacity)))
        }
        DataType::Boolean => {
            Box::new(BoolJsonBuilder(BooleanBuilder::with_capacity(capacity)))
        }
        DataType::Null => Box::new(NullJsonBuilder(NullBuilder::new())),
        _ => Box::new(Utf8JsonBuilder(StringBuilder::with_capacity(
            capacity,
            capacity * 32,
        ))),
    }
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_detect_format_ndjson() {
        let data = b"{\"a\":1}\n{\"a\":2}\n";
        assert_eq!(detect_format(data), JsonFormat::Ndjson);
    }

    #[test]
    fn test_detect_format_json_array() {
        let data = b"[{\"a\":1},{\"a\":2}]";
        assert_eq!(detect_format(data), JsonFormat::JsonArray);
    }

    #[test]
    fn test_detect_format_json_array_whitespace() {
        let data = b"  \n  [{\"a\":1}]";
        assert_eq!(detect_format(data), JsonFormat::JsonArray);
    }

    #[test]
    fn test_find_ndjson_records() {
        let data = b"{\"a\":1}\n{\"b\":2}\n";
        let records = find_ndjson_records(data);
        assert_eq!(records.len(), 2);
        assert_eq!(&data[records[0].0..records[0].1], b"{\"a\":1}");
        assert_eq!(&data[records[1].0..records[1].1], b"{\"b\":2}");
    }

    #[test]
    fn test_find_ndjson_records_empty_lines() {
        let data = b"{\"a\":1}\n\n{\"b\":2}\n";
        let records = find_ndjson_records(data);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_find_ndjson_records_no_trailing_newline() {
        let data = b"{\"a\":1}\n{\"b\":2}";
        let records = find_ndjson_records(data);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_find_json_array_records() {
        let data = b"[{\"a\":1},{\"b\":2}]";
        let records = find_json_array_records(data);
        assert_eq!(records.len(), 2);
        assert_eq!(&data[records[0].0..records[0].1], b"{\"a\":1}");
        assert_eq!(&data[records[1].0..records[1].1], b"{\"b\":2}");
    }

    #[test]
    fn test_find_json_array_records_nested() {
        let data = b"[{\"a\":{\"b\":1}},{\"c\":[1,2]}]";
        let records = find_json_array_records(data);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_find_json_array_records_string_braces() {
        let data = b"[{\"a\":\"hello {world}\"},{\"b\":\"}\"}]";
        let records = find_json_array_records(data);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_ndjson_basic() {
        let f = write_json("{\"name\":\"alice\",\"age\":30}\n{\"name\":\"bob\",\"age\":25}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();

        assert_eq!(reader.num_rows(), 2);
        let names = reader.column_names();
        assert!(names.contains(&"name".to_string()));
        assert!(names.contains(&"age".to_string()));

        let batch = reader.read_all().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_ndjson_schema_inference() {
        let f = write_json(
            "{\"a\":1,\"b\":2.5,\"c\":true,\"d\":\"hello\"}\n\
             {\"a\":2,\"b\":3.5,\"c\":false,\"d\":\"world\"}\n",
        );
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let schema = reader.schema();

        // Find field by name (order may vary)
        let a = schema.field_with_name("a").unwrap();
        let b = schema.field_with_name("b").unwrap();
        let c = schema.field_with_name("c").unwrap();
        let d = schema.field_with_name("d").unwrap();

        assert_eq!(a.data_type(), &DataType::Int64);
        assert_eq!(b.data_type(), &DataType::Float64);
        assert_eq!(c.data_type(), &DataType::Boolean);
        assert_eq!(d.data_type(), &DataType::Utf8);
    }

    #[test]
    fn test_ndjson_values() {
        let f = write_json("{\"name\":\"alice\",\"age\":30}\n{\"name\":\"bob\",\"age\":25}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let batch = reader.read_all().unwrap();

        let name_col = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let age_col = batch
            .column_by_name("age")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(name_col.value(0), "alice");
        assert_eq!(name_col.value(1), "bob");
        assert_eq!(age_col.value(0), 30);
        assert_eq!(age_col.value(1), 25);
    }

    #[test]
    fn test_ndjson_null_values() {
        let f = write_json("{\"a\":1,\"b\":2}\n{\"a\":null,\"b\":3}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let batch = reader.read_all().unwrap();

        let a_col = batch
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(a_col.value(0), 1);
        assert!(a_col.is_null(1));
    }

    #[test]
    fn test_ndjson_missing_keys() {
        let f = write_json("{\"a\":1,\"b\":2}\n{\"a\":3}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let batch = reader.read_all().unwrap();

        let b_col = batch
            .column_by_name("b")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(b_col.value(0), 2);
        assert!(b_col.is_null(1));
    }

    #[test]
    fn test_ndjson_type_widening() {
        // First row has int, second has float for same key
        let f = write_json("{\"val\":1}\n{\"val\":2.5}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let schema = reader.schema();
        assert_eq!(
            schema.field_with_name("val").unwrap().data_type(),
            &DataType::Float64
        );

        let batch = reader.read_all().unwrap();
        let val_col = batch
            .column_by_name("val")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(val_col.value(0), 1.0);
        assert_eq!(val_col.value(1), 2.5);
    }

    #[test]
    fn test_ndjson_nested_objects_as_string() {
        let f = write_json("{\"data\":{\"x\":1}}\n{\"data\":{\"x\":2}}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let schema = reader.schema();

        // Nested objects become Utf8 (JSON string)
        assert_eq!(
            schema.field_with_name("data").unwrap().data_type(),
            &DataType::Utf8
        );
    }

    #[test]
    fn test_json_array_basic() {
        let f = write_json("[{\"name\":\"alice\",\"age\":30},{\"name\":\"bob\",\"age\":25}]");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();

        assert_eq!(reader.num_rows(), 2);
        let batch = reader.read_all().unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_json_array_pretty() {
        let content = r#"[
  {"name": "alice", "score": 95.5},
  {"name": "bob", "score": 87.3}
]"#;
        let f = write_json(content);
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();

        assert_eq!(reader.num_rows(), 2);
        let batch = reader.read_all().unwrap();
        assert_eq!(batch.num_rows(), 2);

        let score_col = batch
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((score_col.value(0) - 95.5).abs() < 0.001);
        assert!((score_col.value(1) - 87.3).abs() < 0.001);
    }

    #[test]
    fn test_json_array_projection() {
        let f = write_json(
            "[{\"a\":1,\"b\":2,\"c\":3},{\"a\":4,\"b\":5,\"c\":6}]",
        );
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let batch = reader.read_columns(&["a", "c"]).unwrap();

        assert_eq!(batch.num_columns(), 2);
        assert!(batch.schema().field_with_name("a").is_ok());
        assert!(batch.schema().field_with_name("c").is_ok());
    }

    #[test]
    fn test_ndjson_boolean_values() {
        let f = write_json("{\"flag\":true}\n{\"flag\":false}\n{\"flag\":true}\n");
        let reader = JsonReader::open(f.path(), JsonReadOptions::default()).unwrap();
        let batch = reader.read_all().unwrap();

        let flag_col = batch
            .column_by_name("flag")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();

        assert!(flag_col.value(0));
        assert!(!flag_col.value(1));
        assert!(flag_col.value(2));
    }

    #[test]
    fn test_ndjson_explicit_format() {
        let f = write_json("{\"x\":1}\n{\"x\":2}\n");
        let opts = JsonReadOptions {
            format: JsonFormat::Ndjson,
            ..Default::default()
        };
        let reader = JsonReader::open(f.path(), opts).unwrap();
        assert_eq!(reader.num_rows(), 2);
    }

    #[test]
    fn test_empty_json_error() {
        let f = write_json("");
        let result = JsonReader::open(f.path(), JsonReadOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_widen_null_int() {
        assert_eq!(widen(DataType::Null, DataType::Int64), DataType::Int64);
    }

    #[test]
    fn test_widen_int_float() {
        assert_eq!(
            widen(DataType::Int64, DataType::Float64),
            DataType::Float64
        );
    }

    #[test]
    fn test_widen_incompatible() {
        assert_eq!(widen(DataType::Boolean, DataType::Int64), DataType::Utf8);
    }
}
