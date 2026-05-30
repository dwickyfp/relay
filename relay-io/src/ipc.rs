//! Arrow IPC file writer for creating test files and data export.

use std::fs::File;
use std::path::Path;

use arrow::datatypes::SchemaRef;
use arrow_array::RecordBatch;
use arrow_ipc::writer::FileWriter;

use relay_core::{RelayError, Result};

/// Options for writing IPC files.
#[derive(Debug, Clone)]
pub struct IPCWriteOptions {
    /// Enable compression (zstd or lz4).
    pub compression: Option<IPCCompression>,
    /// Write batch metadata inline or in footer.
    pub inline_metadata: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum IPCCompression {
    Zstd,
    Lz4,
}

impl Default for IPCWriteOptions {
    fn default() -> Self {
        Self {
            compression: None,
            inline_metadata: true,
        }
    }
}

/// Write record batches to an Arrow IPC file.
pub fn write_ipc(path: &Path, batches: &[RecordBatch], _options: IPCWriteOptions) -> Result<()> {
    if batches.is_empty() {
        return Err(RelayError::Schema(
            "cannot write IPC file with no batches".to_string(),
        ));
    }

    let schema: SchemaRef = batches[0].schema();

    // Validate all batches have the same schema
    for (i, batch) in batches.iter().enumerate().skip(1) {
        if batch.schema() != schema {
            return Err(RelayError::Schema(format!(
                "batch {} schema mismatch: expected {:?}, got {:?}",
                i,
                schema,
                batch.schema()
            )));
        }
    }

    let file = File::create(path)?;

    let mut writer = FileWriter::try_new(file, &schema)
        .map_err(|e| RelayError::Arrow(format!("IPC writer error: {}", e)))?;

    for batch in batches {
        writer
            .write(batch)
            .map_err(|e| RelayError::Arrow(format!("IPC write error: {}", e)))?;
    }

    writer
        .finish()
        .map_err(|e| RelayError::Arrow(format!("IPC finish error: {}", e)))?;

    Ok(())
}

/// Write a single record batch to an IPC file (convenience function).
pub fn write_single_batch(batch: &RecordBatch, path: &Path) -> Result<()> {
    write_ipc(path, std::slice::from_ref(batch), IPCWriteOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::{Float64Array, Int32Array};
    use tempfile::NamedTempFile;

    fn test_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let id = Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>()));
        let val = Arc::new(Float64Array::from(
            (0..n).map(|i| i as f64).collect::<Vec<_>>(),
        ));
        RecordBatch::try_new(schema, vec![id, val]).unwrap()
    }

    #[test]
    fn test_write_ipc_single_batch() {
        let batch = test_batch(100);
        let tmp = NamedTempFile::new().unwrap();
        write_single_batch(&batch, tmp.path()).unwrap();
        assert!(tmp.path().exists());
        assert!(std::fs::metadata(tmp.path()).unwrap().len() > 0);
    }

    #[test]
    fn test_write_ipc_multiple_batches() {
        let batch1 = test_batch(50);
        let batch2 = test_batch(75);
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch1, batch2], IPCWriteOptions::default()).unwrap();
        assert!(tmp.path().exists());
    }

    #[test]
    fn test_write_ipc_empty_fails() {
        let tmp = NamedTempFile::new().unwrap();
        let result = write_ipc(tmp.path(), &[], IPCWriteOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_write_read() {
        use crate::mmap::MmapIPCReader;

        let batch = test_batch(1_000);
        let tmp = NamedTempFile::new().unwrap();
        write_single_batch(&batch, tmp.path()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        assert_eq!(reader.num_rows(), 1_000);

        let read_batch = reader.read_batch(0).unwrap();
        assert_eq!(read_batch.num_rows(), 1_000);
        assert_eq!(read_batch.num_columns(), 2);
    }
}
