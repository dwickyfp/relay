//! Memory-mapped IPC reader for zero-copy column access.
//!
//! Opens Arrow IPC files via mmap and provides zero-copy access to column data.
//! Arrow arrays returned point directly into the mmap region.
//!
//! # Performance
//! - File open: O(1) — mmap + footer parse only
//! - read_batch: O(batch_size) — reads one batch block from mmap
//! - read_columns: O(projected_batch_size) — true projection pushdown via IPC
//! - read_all: O(file_size) — reads all batches
//! - num_rows: O(1) — cached at open time

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow_array::RecordBatch;
use arrow_ipc::reader::FileReader;
use memmap2::{Mmap, MmapOptions};

use crate::madvise::{apply_madvise, AccessPattern};
use relay_core::{RelayError, Result};

/// A zero-copy reader for Arrow IPC files using mmap.
///
/// The mmap region stays alive as long as this reader (or any RecordBatch
/// derived from it) is alive, thanks to `Arc<Mmap>`.
pub struct MmapIPCReader {
    mmap: Arc<Mmap>,
    schema: SchemaRef,
    num_record_batches: usize,
    /// Cached row counts per batch (avoids re-parsing)
    batch_row_counts: Vec<usize>,
    /// Total row count (cached at open time)
    total_rows: usize,
    file_path: String,
}

impl MmapIPCReader {
    /// Open an Arrow IPC file with default (Normal) access pattern.
    /// Access pattern hints are deferred until actual reads.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_pattern(path, AccessPattern::Normal)
    }

    /// Open an Arrow IPC file with a specific access pattern.
    pub fn open_with_pattern(path: &Path, pattern: AccessPattern) -> Result<Self> {
        let file = File::open(path)?;

        // Memory-map the file (lazy — pages loaded on demand)
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                RelayError::Io(std::io::Error::new(
                    e.kind(),
                    format!("mmap failed: {}", e),
                ))
            })?
        };

        // Apply madvise hints for access pattern
        apply_madvise(&mmap, pattern);

        let mmap = Arc::new(mmap);

        // Parse IPC metadata from the mmap region (footer only)
        let cursor = std::io::Cursor::new(mmap.as_ref());
        let reader = FileReader::try_new(cursor, None)
            .map_err(|e| RelayError::Arrow(format!("IPC parse error: {}", e)))?;

        let schema = reader.schema();
        let num_record_batches = reader.num_batches();

        // Cache batch row counts at open time (one-time cost)
        // This avoids re-parsing the footer in num_rows() and other methods
        let batch_row_counts: Vec<usize> = {
            let cursor2 = std::io::Cursor::new(mmap.as_ref());
            let reader2 = FileReader::try_new(cursor2, None)
                .map_err(|e| RelayError::Arrow(format!("IPC metadata parse: {}", e)))?;
            reader2
                .filter_map(|b| b.ok())
                .map(|b| b.num_rows())
                .collect()
        };
        let total_rows: usize = batch_row_counts.iter().sum();

        Ok(Self {
            mmap,
            schema,
            num_record_batches,
            batch_row_counts,
            total_rows,
            file_path: path.to_string_lossy().to_string(),
        })
    }

    /// Get the schema of the IPC file.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Total number of rows across all batches. O(1) — cached.
    pub fn num_rows(&self) -> usize {
        self.total_rows
    }

    /// Number of record batches (row groups) in the file.
    pub fn num_record_batches(&self) -> usize {
        self.num_record_batches
    }

    /// Read a specific record batch (zero-copy from mmap).
    pub fn read_batch(&self, index: usize) -> Result<RecordBatch> {
        if index >= self.num_record_batches {
            return Err(RelayError::OutOfBounds {
                index,
                len: self.num_record_batches,
            });
        }

        let cursor = std::io::Cursor::new(self.mmap.as_ref());
        let mut reader = FileReader::try_new(cursor, None)
            .map_err(|e| RelayError::Arrow(format!("IPC reader open: {}", e)))?;

        // Use set_index for O(1) seek instead of iterating
        reader
            .set_index(index)
            .map_err(|e| RelayError::Arrow(format!("IPC seek: {}", e)))?;

        reader
            .next()
            .ok_or(RelayError::OutOfBounds {
                index,
                len: self.num_record_batches,
            })?
            .map_err(|e| RelayError::Arrow(format!("IPC read batch {}: {}", index, e)))
    }

    /// Read a specific record batch with column projection (zero-copy).
    /// Only reads the projected columns from the mmap.
    pub fn read_batch_projected(&self, index: usize, projection: &[usize]) -> Result<RecordBatch> {
        if index >= self.num_record_batches {
            return Err(RelayError::OutOfBounds {
                index,
                len: self.num_record_batches,
            });
        }

        let cursor = std::io::Cursor::new(self.mmap.as_ref());
        let mut reader = FileReader::try_new(cursor, Some(projection.to_vec()))
            .map_err(|e| RelayError::Arrow(format!("IPC projected reader: {}", e)))?;

        reader
            .set_index(index)
            .map_err(|e| RelayError::Arrow(format!("IPC projected seek: {}", e)))?;

        reader
            .next()
            .ok_or(RelayError::OutOfBounds {
                index,
                len: self.num_record_batches,
            })?
            .map_err(|e| RelayError::Arrow(format!("IPC projected read {}: {}", index, e)))
    }

    /// Read all record batches.
    pub fn read_all(&self) -> Result<Vec<RecordBatch>> {
        let cursor = std::io::Cursor::new(self.mmap.as_ref());
        let reader = FileReader::try_new(cursor, None)
            .map_err(|e| RelayError::Arrow(format!("IPC reader open: {}", e)))?;

        let mut batches = Vec::with_capacity(self.num_record_batches);
        for batch in reader {
            batches.push(batch.map_err(|e| RelayError::Arrow(format!("IPC read batch: {}", e)))?);
        }
        Ok(batches)
    }

    /// Read all batches with column projection pushdown.
    /// Only reads the projected columns — true zero-copy projection.
    pub fn read_all_projected(&self, projection: &[usize]) -> Result<Vec<RecordBatch>> {
        let cursor = std::io::Cursor::new(self.mmap.as_ref());
        let reader = FileReader::try_new(cursor, Some(projection.to_vec()))
            .map_err(|e| RelayError::Arrow(format!("IPC projected open: {}", e)))?;

        let mut batches = Vec::with_capacity(self.num_record_batches);
        for batch in reader {
            batches.push(
                batch.map_err(|e| RelayError::Arrow(format!("IPC projected read: {}", e)))?,
            );
        }
        Ok(batches)
    }

    /// Read only specific columns by name (projection pushdown, zero-copy).
    pub fn read_columns(&self, column_names: &[&str]) -> Result<Vec<RecordBatch>> {
        // Build projection indices
        let field_names: Vec<&str> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        let projection: Vec<usize> = column_names
            .iter()
            .filter_map(|name| field_names.iter().position(|f| *f == *name))
            .collect();

        if projection.is_empty() {
            // Return empty batches if no columns match
            return Ok(Vec::new());
        }

        self.read_all_projected(&projection)
    }

    /// File path (for debugging/display).
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Memory-mapped file size.
    pub fn mmap_size(&self) -> usize {
        self.mmap.len()
    }

    /// Row count for a specific batch. O(1) — cached.
    pub fn batch_row_count(&self, index: usize) -> Option<usize> {
        self.batch_row_counts.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::{write_ipc, IPCWriteOptions};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::{Float64Array, Int32Array, StringArray};
    use tempfile::NamedTempFile;

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let id = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let value = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
        let name = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
        RecordBatch::try_new(schema, vec![id, value, name]).unwrap()
    }

    fn create_large_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let id = Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>()));
        let value = Arc::new(Float64Array::from(
            (0..n).map(|i| i as f64 * 1.5).collect::<Vec<_>>(),
        ));
        RecordBatch::try_new(schema, vec![id, value]).unwrap()
    }

    #[test]
    fn test_open_ipc_file() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        assert_eq!(reader.num_rows(), 5);
        assert_eq!(reader.num_record_batches(), 1);
        assert_eq!(reader.schema().fields().len(), 3);
    }

    #[test]
    fn test_read_batch_integrity() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        let read_batch = reader.read_batch(0).unwrap();

        assert_eq!(read_batch.num_rows(), 5);
        assert_eq!(read_batch.num_columns(), 3);

        let id_col = read_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 1);
        assert_eq!(id_col.value(4), 5);
    }

    #[test]
    fn test_read_all() {
        let batch1 = create_test_batch();
        let batch2 = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch1, batch2], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        let batches = reader.read_all().unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(reader.num_rows(), 10);
    }

    #[test]
    fn test_read_columns_projection() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        let batches = reader.read_columns(&["id", "value"]).unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[0].schema().field(0).name(), "id");
        assert_eq!(batches[0].schema().field(1).name(), "value");
    }

    #[test]
    fn test_read_batch_projected() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        let rb = reader.read_batch_projected(0, &[0, 2]).unwrap();

        assert_eq!(rb.num_rows(), 5);
        assert_eq!(rb.num_columns(), 2);
        assert_eq!(rb.schema().field(0).name(), "id");
        assert_eq!(rb.schema().field(1).name(), "name");
    }

    #[test]
    fn test_large_file() {
        let batch = create_large_batch(100_000);
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        assert_eq!(reader.num_rows(), 100_000);
        assert_eq!(reader.batch_row_count(0), Some(100_000));

        let read_batch = reader.read_batch(0).unwrap();
        let id = read_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.value(99_999), 99_999);
    }

    #[test]
    fn test_out_of_bounds() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        assert!(reader.read_batch(5).is_err());
        assert!(reader.read_batch_projected(5, &[0]).is_err());
    }

    #[test]
    fn test_nonexistent_file() {
        let path = Path::new("/nonexistent/file.ipc");
        assert!(MmapIPCReader::open(path).is_err());
    }

    #[test]
    fn test_batch_row_counts_cached() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[batch.clone(), batch], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        assert_eq!(reader.batch_row_count(0), Some(5));
        assert_eq!(reader.batch_row_count(1), Some(5));
        assert_eq!(reader.batch_row_count(2), None);
        assert_eq!(reader.num_rows(), 10);
    }
}
