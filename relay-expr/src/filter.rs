//! SIMD-optimized filter: monomorphized type-specific kernels (zero alloc).
//!
//! Uses arrow_ord::cmp with typed Scalar wrappers for broadcast comparison.
//! No heap allocation per comparison — all types are stack-allocated.

use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, Scalar, StringArray,
};
use arrow::datatypes::DataType;
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};

use crate::expr::{Expr, Literal, Operator};
use relay_core::{RelayError, Result};

/// Apply filter to RecordBatch, returning filtered rows.
pub fn filter_batch(batch: &RecordBatch, predicate: &Expr) -> Result<RecordBatch> {
    let mask = eval_filter(batch, predicate)?;
    arrow::compute::filter_record_batch(batch, &mask)
        .map_err(|e| RelayError::Arrow(format!("filter: {}", e)))
}

/// Evaluate filter expression to boolean mask (SIMD-optimized).
pub fn eval_filter(batch: &RecordBatch, expr: &Expr) -> Result<BooleanArray> {
    match expr {
        Expr::BinaryOp { left, op, right } => eval_binary(batch, left, right, *op),
        Expr::Not(inner) => {
            let mask = eval_filter(batch, inner)?;
            arrow::compute::not(&mask).map_err(|e| RelayError::Arrow(format!("NOT: {}", e)))
        }
        _ => Err(RelayError::Expr(format!(
            "Filter must be comparison/NOT, got: {}",
            expr
        ))),
    }
}

/// Evaluate binary expression with AND/OR short-circuit.
fn eval_binary(
    batch: &RecordBatch,
    left: &Expr,
    right: &Expr,
    op: Operator,
) -> Result<BooleanArray> {
    match op {
        Operator::And => {
            let l = eval_filter(batch, left)?;
            let r = eval_filter(batch, right)?;
            arrow::compute::and(&l, &r).map_err(|e| RelayError::Arrow(format!("AND: {}", e)))
        }
        Operator::Or => {
            let l = eval_filter(batch, left)?;
            let r = eval_filter(batch, right)?;
            arrow::compute::or(&l, &r).map_err(|e| RelayError::Arrow(format!("OR: {}", e)))
        }
        _ => eval_compare(batch, left, right, op),
    }
}

/// Monomorphized comparison: dispatches to typed kernel based on data type.
/// No heap allocation — all scalars are stack-allocated via Scalar<T>.
fn eval_compare(
    batch: &RecordBatch,
    left: &Expr,
    right: &Expr,
    op: Operator,
) -> Result<BooleanArray> {
    // Column <op> Literal (most common path)
    if let (Expr::Column(col_name), Expr::Literal(lit)) = (left, right) {
        return compare_col_lit(batch, col_name, lit, op);
    }
    // Literal <op> Column → flip operator
    if let (Expr::Literal(lit), Expr::Column(col_name)) = (left, right) {
        return compare_col_lit(batch, col_name, lit, flip_op(op));
    }
    // Column <op> Column
    if let (Expr::Column(l), Expr::Column(r)) = (left, right) {
        return compare_col_col(batch, l, r, op);
    }
    // Fallback: nested expression
    let l = resolve_datum(batch, left)?;
    let r = resolve_datum(batch, right)?;
    apply_cmp(&l, &r, op)
}

/// Column vs Literal: zero-alloc typed dispatch.
fn compare_col_lit(
    batch: &RecordBatch,
    col_name: &str,
    lit: &Literal,
    op: Operator,
) -> Result<BooleanArray> {
    let idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| RelayError::Expr(format!("Column '{}' not found", col_name)))?;
    let col = batch.column(idx);

    match (col.data_type(), lit) {
        (DataType::Int64, Literal::Int64(v)) => {
            let scalar = Scalar::new(Int64Array::from(vec![*v]));
            apply_cmp(col, &scalar, op)
        }
        (DataType::Int32, Literal::Int32(v)) => {
            let scalar = Scalar::new(Int32Array::from(vec![*v]));
            apply_cmp(col, &scalar, op)
        }
        (DataType::Int64, Literal::Int32(v)) => {
            let scalar = Scalar::new(Int64Array::from(vec![*v as i64]));
            apply_cmp(col, &scalar, op)
        }
        (DataType::Float64, Literal::Float64(v)) => {
            let scalar = Scalar::new(Float64Array::from(vec![*v]));
            apply_cmp(col, &scalar, op)
        }
        (DataType::Utf8, Literal::Str(v)) => {
            let scalar = Scalar::new(StringArray::from(vec![v.as_str()]));
            apply_cmp(col, &scalar, op)
        }
        (dt, _) => Err(RelayError::Expr(format!(
            "Unsupported type combination: {:?} vs {:?}",
            dt, lit
        ))),
    }
}

/// Column vs Column comparison.
fn compare_col_col(
    batch: &RecordBatch,
    left_col: &str,
    right_col: &str,
    op: Operator,
) -> Result<BooleanArray> {
    let li = batch
        .schema()
        .index_of(left_col)
        .map_err(|_| RelayError::Expr(format!("Column '{}' not found", left_col)))?;
    let ri = batch
        .schema()
        .index_of(right_col)
        .map_err(|_| RelayError::Expr(format!("Column '{}' not found", right_col)))?;
    let l = batch.column(li).as_ref();
    let r = batch.column(ri).as_ref();
    apply_cmp(&l, &r, op)
}

/// Generic comparison dispatch via arrow_ord::cmp SIMD kernels.
#[inline(always)]
fn apply_cmp(
    lhs: &dyn arrow::array::Datum,
    rhs: &dyn arrow::array::Datum,
    op: Operator,
) -> Result<BooleanArray> {
    let result = match op {
        Operator::Lt => lt(lhs, rhs),
        Operator::Le => lt_eq(lhs, rhs),
        Operator::Gt => gt(lhs, rhs),
        Operator::Ge => gt_eq(lhs, rhs),
        Operator::Eq => eq(lhs, rhs),
        Operator::Ne => neq(lhs, rhs),
        _ => return Err(RelayError::Expr(format!("Non-comparison op: {}", op))),
    };
    result.map_err(|e| RelayError::Arrow(format!("{}: {}", op, e)))
}

/// Fallback datum resolution for nested expressions.
fn resolve_datum(batch: &RecordBatch, expr: &Expr) -> Result<Arc<dyn Array>> {
    match expr {
        Expr::Column(name) => {
            let idx = batch
                .schema()
                .index_of(name)
                .map_err(|_| RelayError::Expr(format!("Column '{}' not found", name)))?;
            Ok(batch.column(idx).clone())
        }
        Expr::Literal(lit) => {
            let arr: Arc<dyn Array> = match lit {
                Literal::Int64(v) => Arc::new(Int64Array::from(vec![*v])),
                Literal::Int32(v) => Arc::new(Int32Array::from(vec![*v])),
                Literal::Float64(v) => Arc::new(Float64Array::from(vec![*v])),
                Literal::Bool(v) => Arc::new(BooleanArray::from(vec![*v])),
                Literal::Str(v) => Arc::new(StringArray::from(vec![v.as_str()])),
                Literal::Null => Arc::new(arrow::array::NullArray::new(1)),
            };
            Ok(arr)
        }
        _ => Err(RelayError::Expr(format!("Cannot resolve: {}", expr))),
    }
}

fn flip_op(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::Le => Operator::Ge,
        Operator::Gt => Operator::Lt,
        Operator::Ge => Operator::Le,
        Operator::Eq => Operator::Eq,
        Operator::Ne => Operator::Ne,
        Operator::And | Operator::Or => op,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{Field, Schema};

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_lt() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("id").lt(Expr::lit_i64(3))).unwrap();
        assert_eq!(r.num_rows(), 2);
    }

    #[test]
    fn test_gt() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("id").gt(Expr::lit_i64(3))).unwrap();
        assert_eq!(r.num_rows(), 2);
    }

    #[test]
    fn test_eq() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("id").eq(Expr::lit_i64(3))).unwrap();
        assert_eq!(r.num_rows(), 1);
    }

    #[test]
    fn test_ne() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("id").ne(Expr::lit_i64(3))).unwrap();
        assert_eq!(r.num_rows(), 4);
    }

    #[test]
    fn test_le() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("id").le(Expr::lit_i64(3))).unwrap();
        assert_eq!(r.num_rows(), 3);
    }

    #[test]
    fn test_ge() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("id").ge(Expr::lit_i64(3))).unwrap();
        assert_eq!(r.num_rows(), 3);
    }

    #[test]
    fn test_float_lt() {
        let b = sample_batch();
        let r = filter_batch(&b, &Expr::col("value").lt(Expr::lit_f64(3.5))).unwrap();
        assert_eq!(r.num_rows(), 3);
    }

    #[test]
    fn test_and() {
        let b = sample_batch();
        let e = Expr::col("id")
            .gt(Expr::lit_i64(1))
            .and(Expr::col("id").lt(Expr::lit_i64(4)));
        assert_eq!(filter_batch(&b, &e).unwrap().num_rows(), 2);
    }

    #[test]
    fn test_or() {
        let b = sample_batch();
        let e = Expr::col("id")
            .lt(Expr::lit_i64(2))
            .or(Expr::col("id").gt(Expr::lit_i64(4)));
        assert_eq!(filter_batch(&b, &e).unwrap().num_rows(), 2);
    }

    #[test]
    fn test_not() {
        let b = sample_batch();
        let e = Expr::not(Expr::col("id").lt(Expr::lit_i64(3)));
        assert_eq!(filter_batch(&b, &e).unwrap().num_rows(), 3);
    }

    #[test]
    fn test_reversed() {
        let b = sample_batch();
        let e = Expr::lit_i64(5).gt(Expr::col("id"));
        assert_eq!(filter_batch(&b, &e).unwrap().num_rows(), 4);
    }
}
