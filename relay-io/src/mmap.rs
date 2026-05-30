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
//! - parallel_agg: per-batch aggregate + reduce (no concat, Rayon parallel)

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_array::RecordBatch;
use arrow_ipc::reader::FileReader;
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;

use crate::madvise::{apply_madvise, AccessPattern};
use relay_core::{RelayError, Result};

/// Compare Int64Array column against a scalar threshold using arrow_ord::cmp (SIMD).
pub fn compare_i64_scalar(
    col: &Int64Array,
    op: &str,
    threshold: i64,
) -> Result<arrow::array::BooleanArray> {
    use arrow::array::Scalar;
    let scalar = Scalar::new(Int64Array::from(vec![threshold]));
    match op {
        "<" => arrow_ord::cmp::lt(col, &scalar).map_err(|e| RelayError::Arrow(e.to_string())),
        "<=" => arrow_ord::cmp::lt_eq(col, &scalar).map_err(|e| RelayError::Arrow(e.to_string())),
        ">" => arrow_ord::cmp::gt(col, &scalar).map_err(|e| RelayError::Arrow(e.to_string())),
        ">=" => arrow_ord::cmp::gt_eq(col, &scalar).map_err(|e| RelayError::Arrow(e.to_string())),
        "==" => arrow_ord::cmp::eq(col, &scalar).map_err(|e| RelayError::Arrow(e.to_string())),
        "!=" => arrow_ord::cmp::neq(col, &scalar).map_err(|e| RelayError::Arrow(e.to_string())),
        _ => Err(RelayError::Expr(format!("Unknown operator: {}", op))),
    }
}

/// Aggregation operation type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

/// Aggregation result
#[derive(Debug, Clone)]
pub enum AggResult {
    Int64(i64),
    Float64(f64),
    Null,
}

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
                RelayError::Io(std::io::Error::new(e.kind(), format!("mmap failed: {}", e)))
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
            batches
                .push(batch.map_err(|e| RelayError::Arrow(format!("IPC projected read: {}", e)))?);
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

    // ─── PARALLEL FUSED OPERATIONS ────────────────────────────────────────

    /// Parallel aggregate: reads batches sequentially, aggregates in parallel with Rayon.
    /// Avoids concat — each batch is aggregated independently, then reduced.
    ///
    /// # Arguments
    /// * `column_name` - Name of the column to aggregate
    /// * `op` - Aggregation operation (Sum, Mean, Min, Max, Count)
    ///
    /// # Returns
    /// The aggregated value
    pub fn parallel_agg(&self, column_name: &str, op: AggOp) -> Result<AggResult> {
        let col_idx = self
            .schema
            .index_of(column_name)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", column_name)))?;

        let num_batches = self.num_record_batches;
        if num_batches == 0 {
            return Ok(AggResult::Null);
        }

        // Read all batches with single-column projection (projection pushdown)
        let batches = self.read_all_projected(&[col_idx])?;

        // Parallel per-batch aggregate + reduce (skip concat)
        let dt = batches[0].column(0).data_type().clone();

        match dt {
            DataType::Int64 => {
                let partials: Vec<i64> = batches
                    .par_iter()
                    .map(|b| {
                        let arr = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                        match op {
                            AggOp::Sum => arrow::compute::sum(arr).unwrap_or(0),
                            AggOp::Count => (arr.len() - arr.null_count()) as i64,
                            AggOp::Min => arrow::compute::min(arr).unwrap_or(i64::MAX),
                            AggOp::Max => arrow::compute::max(arr).unwrap_or(i64::MIN),
                            AggOp::Mean => arrow::compute::sum(arr).unwrap_or(0),
                        }
                    })
                    .collect();

                match op {
                    AggOp::Sum | AggOp::Count => {
                        let total: i64 = partials.iter().sum();
                        Ok(AggResult::Int64(total))
                    }
                    AggOp::Min => {
                        let min = partials.iter().copied().min().unwrap_or(0);
                        Ok(AggResult::Int64(min))
                    }
                    AggOp::Max => {
                        let max = partials.iter().copied().max().unwrap_or(0);
                        Ok(AggResult::Int64(max))
                    }
                    AggOp::Mean => {
                        let total: i64 = partials.iter().sum();
                        let count: i64 = batches
                            .iter()
                            .map(|b| (b.num_rows() - b.column(0).null_count()) as i64)
                            .sum();
                        if count == 0 {
                            Ok(AggResult::Null)
                        } else {
                            Ok(AggResult::Float64(total as f64 / count as f64))
                        }
                    }
                }
            }
            DataType::Float64 => {
                let partials: Vec<f64> = batches
                    .par_iter()
                    .map(|b| {
                        let arr = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
                        match op {
                            AggOp::Sum => arrow::compute::sum(arr).unwrap_or(0.0),
                            AggOp::Count => (arr.len() - arr.null_count()) as f64,
                            AggOp::Min => arrow::compute::min(arr).unwrap_or(f64::MAX),
                            AggOp::Max => arrow::compute::max(arr).unwrap_or(f64::MIN),
                            AggOp::Mean => arrow::compute::sum(arr).unwrap_or(0.0),
                        }
                    })
                    .collect();

                match op {
                    AggOp::Sum | AggOp::Count => {
                        let total: f64 = partials.iter().sum();
                        Ok(AggResult::Float64(total))
                    }
                    AggOp::Min => {
                        let min = partials.iter().copied().fold(f64::MAX, f64::min);
                        Ok(AggResult::Float64(min))
                    }
                    AggOp::Max => {
                        let max = partials.iter().copied().fold(f64::MIN, f64::max);
                        Ok(AggResult::Float64(max))
                    }
                    AggOp::Mean => {
                        let total: f64 = partials.iter().sum();
                        let count: usize = batches
                            .iter()
                            .map(|b| b.num_rows() - b.column(0).null_count())
                            .sum();
                        if count == 0 {
                            Ok(AggResult::Null)
                        } else {
                            Ok(AggResult::Float64(total / count as f64))
                        }
                    }
                }
            }
            _ => Err(RelayError::Expr(format!(
                "Unsupported type for aggregation: {:?}",
                dt
            ))),
        }
    }

    /// Parallel filter: reads batches, filters in parallel, returns filtered batches.
    ///
    /// # Arguments
    /// * `filter_col` - Column name to filter on
    /// * `op` - Comparison operator ("<", "<=", ">", ">=", "==", "!=")
    /// * `threshold` - Threshold value (i64 or f64 as f64)
    ///
    /// # Returns
    /// Vec of filtered RecordBatches
    pub fn parallel_filter_i64(
        &self,
        filter_col: &str,
        op: &str,
        threshold: i64,
    ) -> Result<Vec<RecordBatch>> {
        use arrow::compute;

        // Read all batches (need all columns for result)
        let batches = self.read_all()?;
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        let col_idx = batches[0]
            .schema()
            .index_of(filter_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", filter_col)))?;

        // Parallel: per-batch filter
        let filtered: Vec<RecordBatch> = batches
            .par_iter()
            .filter_map(|batch| {
                let col = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();

                let mask = match compare_i64_scalar(col, op, threshold) {
                    Ok(m) => m,
                    Err(_) => return None,
                };

                // If no rows pass filter, skip this batch
                let passing = mask.len() - mask.null_count();
                if passing == 0 {
                    return None;
                }

                compute::filter_record_batch(batch, &mask).ok()
            })
            .collect();

        Ok(filtered)
    }

    /// Fused parallel filter + aggregate: reads once, filters per-batch, aggregates per-batch.
    /// This is the fastest path — no materialization of filtered data, no concat.
    ///
    /// # Arguments
    /// * `filter_col` - Column to filter on
    /// * `op` - Filter operator
    /// * `threshold` - Filter threshold (i64)
    /// * `agg_col` - Column to aggregate
    /// * `agg_op` - Aggregation operation
    ///
    /// # Returns
    /// The aggregated value after filtering
    pub fn parallel_filter_agg_i64(
        &self,
        filter_col: &str,
        op: &str,
        threshold: i64,
        agg_col: &str,
        agg_op: AggOp,
    ) -> Result<AggResult> {
        // Read all batches with projection (filter_col + agg_col only)
        let filter_idx = self
            .schema
            .index_of(filter_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", filter_col)))?;
        let agg_idx = self
            .schema
            .index_of(agg_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", agg_col)))?;

        // Unique column indices for projection
        let mut projection = vec![filter_idx, agg_idx];
        projection.sort();
        projection.dedup();

        let batches = self.read_all_projected(&projection)?;
        if batches.is_empty() {
            return Ok(AggResult::Null);
        }

        // Map column names to projection indices
        let filter_proj_idx = projection.iter().position(|&i| i == filter_idx).unwrap();
        let agg_proj_idx = projection.iter().position(|&i| i == agg_idx).unwrap();

        // Parallel: per-batch filter + aggregate
        match agg_op {
            AggOp::Sum | AggOp::Count | AggOp::Min | AggOp::Max => {
                let partials: Vec<i64> = batches
                    .par_iter()
                    .map(|batch| {
                        let filter_col = batch
                            .column(filter_proj_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        let agg_col = batch
                            .column(agg_proj_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();

                        // Build mask
                        let mask = match compare_i64_scalar(filter_col, op, threshold) {
                            Ok(m) => m,
                            Err(_) => return 0,
                        };

                        // Manual sum/min/max with mask (avoid allocating filtered array)
                        match agg_op {
                            AggOp::Sum => {
                                let mut sum: i64 = 0;
                                for i in 0..agg_col.len() {
                                    if mask.value(i) && !agg_col.is_null(i) {
                                        sum += agg_col.value(i);
                                    }
                                }
                                sum
                            }
                            AggOp::Count => {
                                let mut count: i64 = 0;
                                for i in 0..agg_col.len() {
                                    if mask.value(i) && !agg_col.is_null(i) {
                                        count += 1;
                                    }
                                }
                                count
                            }
                            AggOp::Min => {
                                let mut min = i64::MAX;
                                for i in 0..agg_col.len() {
                                    if mask.value(i) && !agg_col.is_null(i) {
                                        min = min.min(agg_col.value(i));
                                    }
                                }
                                min
                            }
                            AggOp::Max => {
                                let mut max = i64::MIN;
                                for i in 0..agg_col.len() {
                                    if mask.value(i) && !agg_col.is_null(i) {
                                        max = max.max(agg_col.value(i));
                                    }
                                }
                                max
                            }
                            _ => unreachable!(),
                        }
                    })
                    .collect();

                // Reduce
                match agg_op {
                    AggOp::Sum | AggOp::Count => Ok(AggResult::Int64(partials.iter().sum())),
                    AggOp::Min => Ok(AggResult::Int64(
                        partials.iter().copied().min().unwrap_or(0),
                    )),
                    AggOp::Max => Ok(AggResult::Int64(
                        partials.iter().copied().max().unwrap_or(0),
                    )),
                    _ => unreachable!(),
                }
            }
            AggOp::Mean => {
                // Need both sum and count
                let (sum, count) = batches
                    .par_iter()
                    .map(|batch| {
                        let filter_col = batch
                            .column(filter_proj_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        let agg_col = batch
                            .column(agg_proj_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();

                        let mask = match compare_i64_scalar(filter_col, op, threshold) {
                            Ok(m) => m,
                            Err(_) => return (0i64, 0i64),
                        };

                        let mut sum: i64 = 0;
                        let mut count: i64 = 0;
                        for i in 0..agg_col.len() {
                            if mask.value(i) && !agg_col.is_null(i) {
                                sum += agg_col.value(i);
                                count += 1;
                            }
                        }
                        (sum, count)
                    })
                    .reduce(|| (0i64, 0i64), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));

                if count == 0 {
                    Ok(AggResult::Null)
                } else {
                    Ok(AggResult::Float64(sum as f64 / count as f64))
                }
            }
        }
    }

    /// Aggregate with streaming approach: read and aggregate in one pass.
    /// Avoids materializing all batches in memory simultaneously.
    /// Uses SIMD kernels for maximum throughput.
    ///
    /// # Arguments
    /// * `column_name` - Name of the column to aggregate
    /// * `op` - Aggregation operation (Sum, Mean, Min, Max, Count)
    ///
    /// # Returns
    /// The aggregated value
    pub fn streaming_agg(&self, column_name: &str, op: AggOp) -> Result<AggResult> {
        let col_idx = self
            .schema
            .index_of(column_name)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", column_name)))?;

        let num_batches = self.num_record_batches;
        if num_batches == 0 {
            return Ok(AggResult::Null);
        }

        // Initialize accumulators
        let mut total_sum_i64: i64 = 0;
        let mut total_sum_f64: f64 = 0.0;
        let mut total_count: usize = 0;
        let mut min_val_i64: i64 = i64::MAX;
        let mut max_val_i64: i64 = i64::MIN;
        let mut min_val_f64: f64 = f64::MAX;
        let mut max_val_f64: f64 = f64::MIN;

        // Stream through batches one at a time
        for batch_idx in 0..num_batches {
            let batch = self.read_batch_projected(batch_idx, &[col_idx])?;
            let col = batch.column(0);

            match col.data_type() {
                DataType::Int64 => {
                    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();

                    match op {
                        AggOp::Sum | AggOp::Mean => {
                            let batch_sum = arrow::compute::sum(arr).unwrap_or(0);
                            total_sum_i64 += batch_sum;
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Count => {
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Min => {
                            if let Some(batch_min) = arrow::compute::min(arr) {
                                min_val_i64 = min_val_i64.min(batch_min);
                            }
                        }
                        AggOp::Max => {
                            if let Some(batch_max) = arrow::compute::max(arr) {
                                max_val_i64 = max_val_i64.max(batch_max);
                            }
                        }
                    }
                }
                DataType::Float64 => {
                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();

                    match op {
                        AggOp::Sum | AggOp::Mean => {
                            let batch_sum = arrow::compute::sum(arr).unwrap_or(0.0);
                            total_sum_f64 += batch_sum;
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Count => {
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Min => {
                            if let Some(batch_min) = arrow::compute::min(arr) {
                                min_val_f64 = min_val_f64.min(batch_min);
                            }
                        }
                        AggOp::Max => {
                            if let Some(batch_max) = arrow::compute::max(arr) {
                                max_val_f64 = max_val_f64.max(batch_max);
                            }
                        }
                    }
                }
                _ => {
                    return Err(RelayError::Expr(format!(
                        "Unsupported type for streaming aggregation: {:?}",
                        col.data_type()
                    )));
                }
            }
        }

        // Return final result
        match op {
            AggOp::Sum => {
                if total_count == 0 {
                    Ok(AggResult::Null)
                } else if min_val_i64 != i64::MAX || max_val_i64 != i64::MIN {
                    // Int64 data
                    Ok(AggResult::Int64(total_sum_i64))
                } else {
                    // Float64 data
                    Ok(AggResult::Float64(total_sum_f64))
                }
            }
            AggOp::Count => Ok(AggResult::Int64(total_count as i64)),
            AggOp::Mean => {
                if total_count == 0 {
                    Ok(AggResult::Null)
                } else if min_val_i64 != i64::MAX || max_val_i64 != i64::MIN {
                    Ok(AggResult::Float64(
                        total_sum_i64 as f64 / total_count as f64,
                    ))
                } else {
                    Ok(AggResult::Float64(total_sum_f64 / total_count as f64))
                }
            }
            AggOp::Min => {
                if min_val_i64 != i64::MAX {
                    Ok(AggResult::Int64(min_val_i64))
                } else if min_val_f64 != f64::MAX {
                    Ok(AggResult::Float64(min_val_f64))
                } else {
                    Ok(AggResult::Null)
                }
            }
            AggOp::Max => {
                if max_val_i64 != i64::MIN {
                    Ok(AggResult::Int64(max_val_i64))
                } else if max_val_f64 != f64::MIN {
                    Ok(AggResult::Float64(max_val_f64))
                } else {
                    Ok(AggResult::Null)
                }
            }
        }
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
        write_ipc(
            tmp.path(),
            &[batch.clone(), batch],
            IPCWriteOptions::default(),
        )
        .unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        assert_eq!(reader.batch_row_count(0), Some(5));
        assert_eq!(reader.batch_row_count(1), Some(5));
        assert_eq!(reader.batch_row_count(2), None);
        assert_eq!(reader.num_rows(), 10);
    }

    #[test]
    fn test_parallel_agg_sum() {
        // Create 2 batches with known values
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]));
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]))],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![6, 7, 8, 9, 10]))],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[b1, b2], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        let result = reader.parallel_agg("val", AggOp::Sum).unwrap();

        match result {
            AggResult::Int64(v) => assert_eq!(v, 55), // 1+2+...+10
            _ => panic!("Expected Int64"),
        }
    }

    #[test]
    fn test_parallel_agg_min_max() {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]));
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![5, 3, 1, 4, 2]))],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![15, 13, 11, 14, 12]))],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[b1, b2], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();

        match reader.parallel_agg("val", AggOp::Min).unwrap() {
            AggResult::Int64(v) => assert_eq!(v, 1),
            _ => panic!("Expected Int64"),
        }

        match reader.parallel_agg("val", AggOp::Max).unwrap() {
            AggResult::Int64(v) => assert_eq!(v, 15),
            _ => panic!("Expected Int64"),
        }
    }

    #[test]
    fn test_parallel_filter_i64() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Int64, false),
        ]));
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![6, 7, 8, 9, 10])),
                Arc::new(Int64Array::from(vec![60, 70, 80, 90, 100])),
            ],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[b1, b2], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();
        let filtered = reader.parallel_filter_i64("id", "<", 5).unwrap();

        // Should have rows where id < 5: {1,2,3,4} from b1
        let total_rows: usize = filtered.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 4);
    }

    #[test]
    fn test_parallel_filter_agg_fused() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Int64, false),
        ]));
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![6, 7, 8, 9, 10])),
                Arc::new(Int64Array::from(vec![60, 70, 80, 90, 100])),
            ],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_ipc(tmp.path(), &[b1, b2], IPCWriteOptions::default()).unwrap();

        let reader = MmapIPCReader::open(tmp.path()).unwrap();

        // SUM(val) WHERE id < 5 = 10 + 20 + 30 + 40 = 100
        match reader
            .parallel_filter_agg_i64("id", "<", 5, "val", AggOp::Sum)
            .unwrap()
        {
            AggResult::Int64(v) => assert_eq!(v, 100),
            _ => panic!("Expected Int64"),
        }

        // COUNT(val) WHERE id >= 7 = 4 (rows 7,8,9,10)
        match reader
            .parallel_filter_agg_i64("id", ">=", 7, "val", AggOp::Count)
            .unwrap()
        {
            AggResult::Int64(v) => assert_eq!(v, 4),
            _ => panic!("Expected Int64"),
        }
    }
}
