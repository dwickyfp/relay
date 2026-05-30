//! High-performance Parquet reader with row group pruning, column projection,
//! and parallel processing via Rayon.
//!
//! Unlike IPC (which supports zero-copy mmap), Parquet files are compressed/encoded,
//! so data must be decoded on read. However, Parquet provides powerful advantages:
//!
//! - **Row group pruning**: Skip entire row groups using min/max statistics
//! - **Column projection**: Only decode needed columns via ProjectionMask
//! - **Page-level skipping**: Skip data pages within row groups (via column index)
//! - **Late materialization**: Filter before decoding non-filter columns (RowFilter)

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::file::reader::FileReader as ParquetFileReader;
use rayon::prelude::*;

use relay_core::{RelayError, Result};

use crate::mmap::{compare_i64_scalar, AggOp, AggResult};

/// Default batch size for Parquet reading (good balance of vectorization vs memory).
const DEFAULT_BATCH_SIZE: usize = 8192;

/// A high-performance reader for Apache Parquet files.
///
/// Supports row group pruning, column projection, parallel processing,
/// streaming aggregation, and fused filter+aggregate operations.
pub struct ParquetReader {
    file: File,
    schema: SchemaRef,
    num_row_groups: usize,
    /// Cached row counts per row group
    row_group_row_counts: Vec<usize>,
    /// Total row count (cached at open time)
    total_rows: usize,
    /// Schema descriptor for ProjectionMask construction
    schema_descr: parquet::schema::types::SchemaDescPtr,
    file_path: String,
}

impl ParquetReader {
    /// Open a Parquet file and parse metadata.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(
            file.try_clone().map_err(|e| RelayError::Io(e))?,
        )
        .map_err(|e| RelayError::Arrow(format!("Parquet parse error: {}", e)))?;

        let schema = builder.schema().clone();
        let metadata = builder.metadata();
        let num_row_groups = metadata.num_row_groups();
        let schema_descr = metadata.file_metadata().schema_descr_ptr();

        // Cache row group row counts at open time
        let row_group_row_counts: Vec<usize> = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .collect();
        let total_rows: usize = row_group_row_counts.iter().sum();

        Ok(Self {
            file,
            schema,
            num_row_groups,
            row_group_row_counts,
            total_rows,
            schema_descr,
            file_path: path.to_string_lossy().to_string(),
        })
    }

    /// Get the Arrow schema of the Parquet file.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Total number of rows across all row groups. O(1) — cached.
    pub fn num_rows(&self) -> usize {
        self.total_rows
    }

    /// Number of row groups in the file.
    pub fn num_row_groups(&self) -> usize {
        self.num_row_groups
    }

    /// Row count for a specific row group. O(1) — cached.
    pub fn row_group_row_count(&self, index: usize) -> Option<usize> {
        self.row_group_row_counts.get(index).copied()
    }

    /// File path (for debugging/display).
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Build a ParquetRecordBatchReaderBuilder from the current file.
    fn new_builder(&self) -> Result<ParquetRecordBatchReaderBuilder<File>> {
        let file = self.file.try_clone().map_err(|e| RelayError::Io(e))?;
        ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| RelayError::Arrow(format!("Parquet builder error: {}", e)))
    }

    /// Read a specific row group as RecordBatch(es).
    pub fn read_batch(&self, index: usize) -> Result<RecordBatch> {
        if index >= self.num_row_groups {
            return Err(RelayError::OutOfBounds {
                index,
                len: self.num_row_groups,
            });
        }

        let builder = self.new_builder()?;
        let reader = builder
            .with_row_groups(vec![index])
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build()
            .map_err(|e| RelayError::Arrow(format!("Parquet read error: {}", e)))?;

        // Collect all batches from this row group and concat
        let mut batches: Vec<RecordBatch> = Vec::new();
        for batch in reader {
            batches
                .push(batch.map_err(|e| RelayError::Arrow(format!("Parquet batch read: {}", e)))?);
        }

        if batches.is_empty() {
            return Err(RelayError::Arrow(format!("Row group {} is empty", index)));
        }

        if batches.len() == 1 {
            Ok(batches.into_iter().next().unwrap())
        } else {
            // Concatenate batches within the row group
            concat_batches(&batches)
        }
    }

    /// Read a specific row group with column projection.
    pub fn read_batch_projected(&self, index: usize, projection: &[usize]) -> Result<RecordBatch> {
        if index >= self.num_row_groups {
            return Err(RelayError::OutOfBounds {
                index,
                len: self.num_row_groups,
            });
        }

        let mask = ProjectionMask::leaves(&self.schema_descr, projection.to_vec());
        let builder = self.new_builder()?;
        let reader = builder
            .with_row_groups(vec![index])
            .with_projection(mask)
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build()
            .map_err(|e| RelayError::Arrow(format!("Parquet projected read: {}", e)))?;

        let mut batches: Vec<RecordBatch> = Vec::new();
        for batch in reader {
            batches.push(
                batch.map_err(|e| RelayError::Arrow(format!("Parquet projected batch: {}", e)))?,
            );
        }

        if batches.is_empty() {
            return Err(RelayError::Arrow(format!("Row group {} is empty", index)));
        }

        if batches.len() == 1 {
            Ok(batches.into_iter().next().unwrap())
        } else {
            concat_batches(&batches)
        }
    }

    /// Read all row groups.
    pub fn read_all(&self) -> Result<Vec<RecordBatch>> {
        let builder = self.new_builder()?;
        let reader = builder
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build()
            .map_err(|e| RelayError::Arrow(format!("Parquet read all: {}", e)))?;

        let mut batches = Vec::with_capacity(self.num_row_groups);
        for batch in reader {
            batches
                .push(batch.map_err(|e| RelayError::Arrow(format!("Parquet read batch: {}", e)))?);
        }
        Ok(batches)
    }

    /// Read all row groups with column projection pushdown.
    pub fn read_all_projected(&self, projection: &[usize]) -> Result<Vec<RecordBatch>> {
        let mask = ProjectionMask::leaves(&self.schema_descr, projection.to_vec());
        let builder = self.new_builder()?;
        let reader = builder
            .with_projection(mask)
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build()
            .map_err(|e| RelayError::Arrow(format!("Parquet projected read all: {}", e)))?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(
                batch.map_err(|e| RelayError::Arrow(format!("Parquet projected batch: {}", e)))?,
            );
        }
        Ok(batches)
    }

    /// Read only specific columns by name (projection pushdown).
    pub fn read_columns(&self, column_names: &[&str]) -> Result<Vec<RecordBatch>> {
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
            return Ok(Vec::new());
        }

        self.read_all_projected(&projection)
    }

    // ─── ROW GROUP PRUNING (Parquet-specific) ────────────────────────────

    /// Determine which row groups match a filter predicate using min/max statistics.
    /// Returns indices of row groups that COULD contain matching rows.
    ///
    /// This is the key Parquet advantage: skip entire row groups before any I/O.
    pub fn row_groups_matching_filter(
        &self,
        filter_col: &str,
        op: &str,
        threshold: i64,
    ) -> Result<Vec<usize>> {
        let file = self.file.try_clone().map_err(|e| RelayError::Io(e))?;
        let metadata = parquet::file::reader::SerializedFileReader::new(file)
            .map_err(|e| RelayError::Arrow(format!("Parquet metadata: {}", e)))?
            .metadata()
            .clone();

        // Find column index
        let col_idx = self
            .schema
            .index_of(filter_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", filter_col)))?;

        let mut matching = Vec::with_capacity(self.num_row_groups);

        for rg_idx in 0..self.num_row_groups {
            let rg = metadata.row_group(rg_idx);
            let col_chunk = rg.column(col_idx);

            let stats = match col_chunk.statistics() {
                Some(s) => s,
                None => {
                    // No statistics? Include row group (can't prune)
                    matching.push(rg_idx);
                    continue;
                }
            };

            // Extract min/max as i64 (assuming Int64 type)
            let (min_val, max_val) = if let Some(min_bytes) = stats.min_bytes_opt() {
                if let Some(max_bytes) = stats.max_bytes_opt() {
                    if min_bytes.len() == 8 && max_bytes.len() == 8 {
                        let min = i64::from_le_bytes(min_bytes.try_into().unwrap());
                        let max = i64::from_le_bytes(max_bytes.try_into().unwrap());
                        (min, max)
                    } else {
                        matching.push(rg_idx);
                        continue;
                    }
                } else {
                    matching.push(rg_idx);
                    continue;
                }
            } else {
                matching.push(rg_idx);
                continue;
            };

            // Evaluate predicate against row group statistics
            let could_match = match op {
                "<" => min_val < threshold,   // min must be < threshold
                "<=" => min_val <= threshold, // min must be <= threshold
                ">" => max_val > threshold,   // max must be > threshold
                ">=" => max_val >= threshold, // max must be >= threshold
                "==" => min_val <= threshold && max_val >= threshold, // range must contain threshold
                "!=" => !(min_val == threshold && max_val == threshold), // not all same value
                _ => true,
            };

            if could_match {
                matching.push(rg_idx);
            }
        }

        Ok(matching)
    }

    // ─── PARALLEL FUSED OPERATIONS ────────────────────────────────────────

    /// Parallel aggregate: reads row groups, aggregates in parallel with Rayon.
    /// Uses column projection pushdown (only reads the target column).
    pub fn parallel_agg(&self, column_name: &str, op: AggOp) -> Result<AggResult> {
        let col_idx = self
            .schema
            .index_of(column_name)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", column_name)))?;

        if self.num_row_groups == 0 {
            return Ok(AggResult::Null);
        }

        // Read all with projection (single column)
        let batches = self.read_all_projected(&[col_idx])?;
        if batches.is_empty() {
            return Ok(AggResult::Null);
        }

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
                    AggOp::Sum | AggOp::Count => Ok(AggResult::Int64(partials.iter().sum())),
                    AggOp::Min => Ok(AggResult::Int64(
                        partials.iter().copied().min().unwrap_or(0),
                    )),
                    AggOp::Max => Ok(AggResult::Int64(
                        partials.iter().copied().max().unwrap_or(0),
                    )),
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
                    AggOp::Sum | AggOp::Count => Ok(AggResult::Float64(partials.iter().sum())),
                    AggOp::Min => Ok(AggResult::Float64(
                        partials.iter().copied().fold(f64::MAX, f64::min),
                    )),
                    AggOp::Max => Ok(AggResult::Float64(
                        partials.iter().copied().fold(f64::MIN, f64::max),
                    )),
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

    /// Streaming aggregation: read one row group at a time, aggregate incrementally.
    /// Lower memory footprint than parallel_agg (never holds all batches).
    pub fn streaming_agg(&self, column_name: &str, op: AggOp) -> Result<AggResult> {
        let col_idx = self
            .schema
            .index_of(column_name)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", column_name)))?;

        if self.num_row_groups == 0 {
            return Ok(AggResult::Null);
        }

        let mut total_sum_i64: i64 = 0;
        let mut total_sum_f64: f64 = 0.0;
        let mut total_count: usize = 0;
        let mut min_val_i64: i64 = i64::MAX;
        let mut max_val_i64: i64 = i64::MIN;
        let mut min_val_f64: f64 = f64::MAX;
        let mut max_val_f64: f64 = f64::MIN;
        let mut is_int = false;
        let mut is_float = false;

        for rg_idx in 0..self.num_row_groups {
            let batch = self.read_batch_projected(rg_idx, &[col_idx])?;
            let col = batch.column(0);

            match col.data_type() {
                DataType::Int64 => {
                    is_int = true;
                    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    match op {
                        AggOp::Sum | AggOp::Mean => {
                            total_sum_i64 += arrow::compute::sum(arr).unwrap_or(0);
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Count => {
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Min => {
                            if let Some(v) = arrow::compute::min(arr) {
                                min_val_i64 = min_val_i64.min(v);
                            }
                        }
                        AggOp::Max => {
                            if let Some(v) = arrow::compute::max(arr) {
                                max_val_i64 = max_val_i64.max(v);
                            }
                        }
                    }
                }
                DataType::Float64 => {
                    is_float = true;
                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    match op {
                        AggOp::Sum | AggOp::Mean => {
                            total_sum_f64 += arrow::compute::sum(arr).unwrap_or(0.0);
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Count => {
                            total_count += arr.len() - arr.null_count();
                        }
                        AggOp::Min => {
                            if let Some(v) = arrow::compute::min(arr) {
                                min_val_f64 = min_val_f64.min(v);
                            }
                        }
                        AggOp::Max => {
                            if let Some(v) = arrow::compute::max(arr) {
                                max_val_f64 = max_val_f64.max(v);
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

        match op {
            AggOp::Sum => {
                if total_count == 0 {
                    Ok(AggResult::Null)
                } else if is_int {
                    Ok(AggResult::Int64(total_sum_i64))
                } else {
                    Ok(AggResult::Float64(total_sum_f64))
                }
            }
            AggOp::Count => Ok(AggResult::Int64(total_count as i64)),
            AggOp::Mean => {
                if total_count == 0 {
                    Ok(AggResult::Null)
                } else if is_int {
                    Ok(AggResult::Float64(
                        total_sum_i64 as f64 / total_count as f64,
                    ))
                } else {
                    Ok(AggResult::Float64(total_sum_f64 / total_count as f64))
                }
            }
            AggOp::Min => {
                if is_int && min_val_i64 != i64::MAX {
                    Ok(AggResult::Int64(min_val_i64))
                } else if is_float && min_val_f64 != f64::MAX {
                    Ok(AggResult::Float64(min_val_f64))
                } else {
                    Ok(AggResult::Null)
                }
            }
            AggOp::Max => {
                if is_int && max_val_i64 != i64::MIN {
                    Ok(AggResult::Int64(max_val_i64))
                } else if is_float && max_val_f64 != f64::MIN {
                    Ok(AggResult::Float64(max_val_f64))
                } else {
                    Ok(AggResult::Null)
                }
            }
        }
    }

    /// Parallel filter: reads row groups, filters in parallel with SIMD.
    /// Uses row group pruning to skip non-matching row groups.
    pub fn parallel_filter_i64(
        &self,
        filter_col: &str,
        op: &str,
        threshold: i64,
    ) -> Result<Vec<RecordBatch>> {
        use arrow::compute;

        // Row group pruning: skip entire row groups that can't match
        let matching_rgs = self.row_groups_matching_filter(filter_col, op, threshold)?;
        if matching_rgs.is_empty() {
            return Ok(Vec::new());
        }

        // Read only matching row groups (with all columns for result)
        let builder = self.new_builder()?;
        let reader = builder
            .with_row_groups(matching_rgs)
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build()
            .map_err(|e| RelayError::Arrow(format!("Parquet filter read: {}", e)))?;

        let col_idx = self
            .schema
            .index_of(filter_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", filter_col)))?;

        let batches: Vec<RecordBatch> = reader.filter_map(|b| b.ok()).collect();

        // Parallel filter per batch
        let filtered: Vec<RecordBatch> = batches
            .par_iter()
            .filter_map(|batch| {
                let col = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<Int64Array>()?;

                let mask = compare_i64_scalar(col, op, threshold).ok()?;

                // Skip empty results
                let passing = mask.iter().filter(|v| v == &Some(true)).count();
                if passing == 0 {
                    return None;
                }

                compute::filter_record_batch(batch, &mask).ok()
            })
            .collect();

        Ok(filtered)
    }

    /// Fused parallel filter + aggregate with row group pruning.
    /// The fastest path: skip non-matching row groups, filter + agg in one pass.
    pub fn parallel_filter_agg_i64(
        &self,
        filter_col: &str,
        op: &str,
        threshold: i64,
        agg_col: &str,
        agg_op: AggOp,
    ) -> Result<AggResult> {
        // Row group pruning
        let matching_rgs = self.row_groups_matching_filter(filter_col, op, threshold)?;
        if matching_rgs.is_empty() {
            return Ok(AggResult::Null);
        }

        let filter_idx = self
            .schema
            .index_of(filter_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", filter_col)))?;
        let agg_idx = self
            .schema
            .index_of(agg_col)
            .map_err(|_| RelayError::Expr(format!("Column '{}' not found", agg_col)))?;

        // Project only needed columns
        let mut projection = vec![filter_idx, agg_idx];
        projection.sort();
        projection.dedup();

        let mask_proj = ProjectionMask::leaves(&self.schema_descr, projection.clone());
        let builder = self.new_builder()?;
        let reader = builder
            .with_row_groups(matching_rgs)
            .with_projection(mask_proj)
            .with_batch_size(DEFAULT_BATCH_SIZE)
            .build()
            .map_err(|e| RelayError::Arrow(format!("Parquet fused read: {}", e)))?;

        let batches: Vec<RecordBatch> = reader.filter_map(|b| b.ok()).collect();
        if batches.is_empty() {
            return Ok(AggResult::Null);
        }

        // Map column names to projection indices
        let filter_proj_idx = projection.iter().position(|&i| i == filter_idx).unwrap();
        let agg_proj_idx = projection.iter().position(|&i| i == agg_idx).unwrap();

        // Detect aggregation column type from first batch
        let agg_data_type = batches[0].column(agg_proj_idx).data_type().clone();

        // Parallel: per-batch filter + aggregate (fused, no materialization)
        match agg_op {
            AggOp::Sum | AggOp::Count | AggOp::Min | AggOp::Max => {
                match agg_data_type {
                    DataType::Float64 => {
                        let partials_f64: Vec<f64> = batches
                            .par_iter()
                            .map(|batch| {
                                let filter_col_arr = batch
                                    .column(filter_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                let agg_arr = batch
                                    .column(agg_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .unwrap();
                                let mask = match compare_i64_scalar(filter_col_arr, op, threshold) {
                                    Ok(m) => m,
                                    Err(_) => return 0.0,
                                };
                                match agg_op {
                                    AggOp::Sum => {
                                        let mut sum: f64 = 0.0;
                                        for i in 0..agg_arr.len() {
                                            if mask.value(i) && !agg_arr.is_null(i) {
                                                sum += agg_arr.value(i);
                                            }
                                        }
                                        sum
                                    }
                                    AggOp::Count => {
                                        let mut count: f64 = 0.0;
                                        for i in 0..agg_arr.len() {
                                            if mask.value(i) && !agg_arr.is_null(i) {
                                                count += 1.0;
                                            }
                                        }
                                        count
                                    }
                                    AggOp::Min => {
                                        let mut min = f64::MAX;
                                        for i in 0..agg_arr.len() {
                                            if mask.value(i) && !agg_arr.is_null(i) {
                                                min = min.min(agg_arr.value(i));
                                            }
                                        }
                                        min
                                    }
                                    AggOp::Max => {
                                        let mut max = f64::MIN;
                                        for i in 0..agg_arr.len() {
                                            if mask.value(i) && !agg_arr.is_null(i) {
                                                max = max.max(agg_arr.value(i));
                                            }
                                        }
                                        max
                                    }
                                    _ => unreachable!(),
                                }
                            })
                            .collect();
                        match agg_op {
                            AggOp::Sum | AggOp::Count => {
                                Ok(AggResult::Float64(partials_f64.iter().sum()))
                            }
                            AggOp::Min => Ok(AggResult::Float64(
                                partials_f64.iter().copied().fold(f64::MAX, f64::min),
                            )),
                            AggOp::Max => Ok(AggResult::Float64(
                                partials_f64.iter().copied().fold(f64::MIN, f64::max),
                            )),
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        // Default: Int64 aggregation column
                        let partials: Vec<i64> = batches
                            .par_iter()
                            .map(|batch| {
                                let filter_col_arr = batch
                                    .column(filter_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                let agg_col = batch
                                    .column(agg_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                let mask = match compare_i64_scalar(filter_col_arr, op, threshold) {
                                    Ok(m) => m,
                                    Err(_) => return 0,
                                };
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
                }
            }
            AggOp::Mean => {
                // Mean always returns Float64 — handle both Int64 and Float64 agg columns
                match agg_data_type {
                    DataType::Float64 => {
                        let (sum, count) = batches
                            .par_iter()
                            .map(|batch| {
                                let filter_col_arr = batch
                                    .column(filter_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                let agg_arr = batch
                                    .column(agg_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .unwrap();
                                let mask = match compare_i64_scalar(filter_col_arr, op, threshold) {
                                    Ok(m) => m,
                                    Err(_) => return (0.0f64, 0i64),
                                };
                                let mut sum: f64 = 0.0;
                                let mut count: i64 = 0;
                                for i in 0..agg_arr.len() {
                                    if mask.value(i) && !agg_arr.is_null(i) {
                                        sum += agg_arr.value(i);
                                        count += 1;
                                    }
                                }
                                (sum, count)
                            })
                            .reduce(
                                || (0.0f64, 0i64),
                                |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2),
                            );
                        if count == 0 {
                            Ok(AggResult::Null)
                        } else {
                            Ok(AggResult::Float64(sum / count as f64))
                        }
                    }
                    _ => {
                        let (sum, count) = batches
                            .par_iter()
                            .map(|batch| {
                                let filter_col_arr = batch
                                    .column(filter_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                let agg_col = batch
                                    .column(agg_proj_idx)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .unwrap();
                                let mask = match compare_i64_scalar(filter_col_arr, op, threshold) {
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
        }
    }
}

/// Concatenate multiple RecordBatches into one.
fn concat_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Err(RelayError::Arrow("No batches to concat".into()));
    }
    if batches.len() == 1 {
        return Ok(batches[0].clone());
    }

    let schema = batches[0].schema();
    let num_cols = schema.fields().len();
    let mut result_cols = Vec::with_capacity(num_cols);

    for col_idx in 0..num_cols {
        let refs: Vec<&dyn Array> = batches.iter().map(|b| b.column(col_idx).as_ref()).collect();
        let concatenated = arrow::compute::concat(&refs)
            .map_err(|e| RelayError::Arrow(format!("Concat error: {}", e)))?;
        result_cols.push(concatenated);
    }

    RecordBatch::try_new(schema, result_cols)
        .map_err(|e| RelayError::Arrow(format!("RecordBatch::try_new: {}", e)))
}

// ─── TESTS ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::Int32Array;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let id = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let value = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
        let name = Arc::new(arrow_array::StringArray::from(vec![
            "a", "b", "c", "d", "e",
        ]));
        RecordBatch::try_new(schema, vec![id, value, name]).unwrap()
    }

    fn write_parquet_test(
        path: &Path,
        batches: &[RecordBatch],
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1000))
            .build();
        let mut writer = ArrowWriter::try_new(file, batches[0].schema(), Some(props))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
        Ok(())
    }

    fn write_parquet_with_row_groups(
        path: &Path,
        batches: &[RecordBatch],
        row_group_size: usize,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group_size))
            .build();
        let mut writer = ArrowWriter::try_new(file, batches[0].schema(), Some(props))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
        Ok(())
    }

    #[test]
    fn test_open_parquet() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[batch]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
        assert_eq!(reader.num_rows(), 5);
        assert!(reader.num_row_groups() >= 1);
        assert_eq!(reader.schema().fields().len(), 3);
    }

    #[test]
    fn test_read_batch_integrity() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[batch]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
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
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[batch.clone(), batch]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
        let batches = reader.read_all().unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_read_columns_projection() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[batch]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
        let batches = reader.read_columns(&["id", "value"]).unwrap();

        assert!(!batches.is_empty());
        let total_cols = batches[0].num_columns();
        assert_eq!(total_cols, 2);
    }

    #[test]
    fn test_parallel_agg_sum() {
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
        write_parquet_test(tmp.path(), &[b1, b2]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
        let result = reader.parallel_agg("val", AggOp::Sum).unwrap();

        match result {
            AggResult::Int64(v) => assert_eq!(v, 55),
            _ => panic!("Expected Int64"),
        }
    }

    #[test]
    fn test_streaming_agg() {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![10, 20, 30]))])
            .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[b]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();

        match reader.streaming_agg("val", AggOp::Sum).unwrap() {
            AggResult::Int64(v) => assert_eq!(v, 60),
            _ => panic!("Expected Int64"),
        }

        match reader.streaming_agg("val", AggOp::Min).unwrap() {
            AggResult::Int64(v) => assert_eq!(v, 10),
            _ => panic!("Expected Int64"),
        }

        match reader.streaming_agg("val", AggOp::Max).unwrap() {
            AggResult::Int64(v) => assert_eq!(v, 30),
            _ => panic!("Expected Int64"),
        }
    }

    #[test]
    fn test_row_group_pruning() {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)]));

        // Create data where row groups have distinct ranges:
        // RG0: [0..999], RG1: [1000..1999], RG2: [2000..2999]
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from((0..1000i64).collect::<Vec<_>>()))],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(
                (1000..2000i64).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        let b3 = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(
                (2000..3000i64).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_parquet_with_row_groups(tmp.path(), &[b1, b2, b3], 1000).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
        assert!(reader.num_row_groups() >= 3);

        // Filter: val > 1500 → should skip RG0 (max=999, 999 > 1500 is false)
        let matching = reader.row_groups_matching_filter("val", ">", 1500).unwrap();
        assert!(
            matching.len() < reader.num_row_groups(),
            "Row group pruning should skip some"
        );

        // Filter: val == 500 → should only match RG0
        let matching = reader.row_groups_matching_filter("val", "==", 500).unwrap();
        assert!(matching.contains(&0), "RG0 should match val==500");
        assert!(!matching.contains(&2), "RG2 should NOT match val==500");
    }

    #[test]
    fn test_fused_filter_agg() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("filter_col", DataType::Int64, false),
            Field::new("agg_col", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[batch]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();

        // filter_col > 3, sum(agg_col) → 40 + 50 = 90
        let result = reader
            .parallel_filter_agg_i64("filter_col", ">", 3, "agg_col", AggOp::Sum)
            .unwrap();

        match result {
            AggResult::Int64(v) => assert_eq!(v, 90),
            _ => panic!("Expected Int64(90)"),
        }
    }

    #[test]
    fn test_out_of_bounds() {
        let batch = create_test_batch();
        let tmp = NamedTempFile::new().unwrap();
        write_parquet_test(tmp.path(), &[batch]).unwrap();

        let reader = ParquetReader::open(tmp.path()).unwrap();
        assert!(reader.read_batch(99).is_err());
    }

    #[test]
    fn test_nonexistent_file() {
        let path = Path::new("/nonexistent/file.parquet");
        assert!(ParquetReader::open(path).is_err());
    }
}
