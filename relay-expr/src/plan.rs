//! Query plan — composes filter, projection, and aggregation into a pipeline.

use arrow_array::RecordBatch;
use std::sync::Arc;

use crate::agg::{aggregate_array, AggOp, AggResult};
use crate::expr::Expr;
use crate::filter::filter_batch;

/// A composable query plan that can be executed against RecordBatches.
///
/// Pipeline order: filter → project → aggregate
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Optional filter predicate.
    pub filter: Option<Expr>,
    /// Optional column projection (names to keep).
    pub projection: Option<Vec<String>>,
    /// Optional aggregation operations.
    pub aggregations: Vec<AggExpr>,
}

/// An aggregation expression: (column_name, operation).
#[derive(Debug, Clone)]
pub struct AggExpr {
    pub column: String,
    pub op: AggOp,
    pub alias: String,
}

impl AggExpr {
    pub fn new(column: &str, op: AggOp) -> Self {
        let alias = format!("{}_{}", op, column);
        Self {
            column: column.to_string(),
            op,
            alias,
        }
    }

    pub fn with_alias(column: &str, op: AggOp, alias: &str) -> Self {
        Self {
            column: column.to_string(),
            op,
            alias: alias.to_string(),
        }
    }
}

impl QueryPlan {
    /// Create an empty plan (scan everything).
    pub fn new() -> Self {
        Self {
            filter: None,
            projection: None,
            aggregations: Vec::new(),
        }
    }

    /// Add a filter predicate.
    pub fn filter(mut self, predicate: Expr) -> Self {
        self.filter = Some(predicate);
        self
    }

    /// Add a column projection.
    pub fn project(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }

    /// Add an aggregation.
    pub fn agg(mut self, expr: AggExpr) -> Self {
        self.aggregations.push(expr);
        self
    }

    /// Execute the plan against a RecordBatch.
    /// Returns either a filtered/projected batch or aggregation results.
    pub fn execute(&self, batch: RecordBatch) -> Result<PlanOutput, relay_core::RelayError> {
        let mut current = batch;

        // Step 1: Filter
        if let Some(ref predicate) = self.filter {
            current = filter_batch(&current, predicate)?;
        }

        // Step 2: Project
        if let Some(ref columns) = self.projection {
            let mut indices = Vec::new();
            for col_name in columns {
                if let Ok(idx) = current.schema().index_of(col_name) {
                    indices.push(idx);
                }
            }
            let projected_cols: Vec<_> =
                indices.iter().map(|&i| current.column(i).clone()).collect();
            let projected_fields: Vec<_> = indices
                .iter()
                .map(|&i| current.schema().field(i).clone())
                .collect();
            let projected_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
                projected_fields,
                current.schema().metadata().clone(),
            ));
            current = RecordBatch::try_new(projected_schema, projected_cols)
                .map_err(|e| relay_core::RelayError::Arrow(format!("projection: {}", e)))?;
        }

        // Step 3: Aggregation (if requested)
        if !self.aggregations.is_empty() {
            let mut results = Vec::new();
            for agg_expr in &self.aggregations {
                let col_idx = current.schema().index_of(&agg_expr.column).map_err(|_| {
                    relay_core::RelayError::Arrow(format!("column not found: {}", agg_expr.column))
                })?;
                let col = current.column(col_idx);
                let result = aggregate_array(col.as_ref(), agg_expr.op)?;
                results.push((agg_expr.alias.clone(), result));
            }
            return Ok(PlanOutput::Aggregated(results));
        }

        Ok(PlanOutput::Batch(current))
    }
}

impl Default for QueryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Output of executing a query plan.
#[derive(Debug)]
pub enum PlanOutput {
    /// A filtered/projected RecordBatch.
    Batch(RecordBatch),
    /// Aggregation results: (alias, value).
    Aggregated(Vec<(String, AggResult)>),
}

impl std::fmt::Display for PlanOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanOutput::Batch(batch) => {
                write!(
                    f,
                    "Batch(rows={}, cols={})",
                    batch.num_rows(),
                    batch.num_columns()
                )
            }
            PlanOutput::Aggregated(results) => {
                let parts: Vec<String> = results
                    .iter()
                    .map(|(name, val)| format!("{}={}", name, val))
                    .collect();
                write!(f, "Agg({})", parts.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array};
    use arrow_schema::{DataType, Field};

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_filter_only() {
        let batch = sample_batch();
        let plan = QueryPlan::new().filter(Expr::col("id").gt(Expr::lit_i64(3)));
        let output = plan.execute(batch).unwrap();
        match output {
            PlanOutput::Batch(b) => assert_eq!(b.num_rows(), 2),
            _ => panic!("expected batch"),
        }
    }

    #[test]
    fn test_project_only() {
        let batch = sample_batch();
        let plan = QueryPlan::new().project(vec!["value".to_string()]);
        let output = plan.execute(batch).unwrap();
        match output {
            PlanOutput::Batch(b) => {
                assert_eq!(b.num_columns(), 1);
                assert_eq!(b.schema().field(0).name(), "value");
            }
            _ => panic!("expected batch"),
        }
    }

    #[test]
    fn test_filter_then_project() {
        let batch = sample_batch();
        let plan = QueryPlan::new()
            .filter(Expr::col("id").lt(Expr::lit_i64(4)))
            .project(vec!["id".to_string()]);
        let output = plan.execute(batch).unwrap();
        match output {
            PlanOutput::Batch(b) => {
                assert_eq!(b.num_rows(), 3);
                assert_eq!(b.num_columns(), 1);
            }
            _ => panic!("expected batch"),
        }
    }

    #[test]
    fn test_aggregate() {
        let batch = sample_batch();
        let plan = QueryPlan::new().agg(AggExpr::new("value", AggOp::Sum));
        let output = plan.execute(batch).unwrap();
        match output {
            PlanOutput::Aggregated(results) => {
                assert_eq!(results.len(), 1);
                match &results[0].1 {
                    AggResult::Float64(v) => assert_eq!(*v, 150.0),
                    _ => panic!("expected float"),
                }
            }
            _ => panic!("expected aggregated"),
        }
    }

    #[test]
    fn test_filter_then_aggregate() {
        let batch = sample_batch();
        let plan = QueryPlan::new()
            .filter(Expr::col("id").gt(Expr::lit_i64(3)))
            .agg(AggExpr::new("id", AggOp::Sum));
        let output = plan.execute(batch).unwrap();
        match output {
            PlanOutput::Aggregated(results) => {
                match &results[0].1 {
                    AggResult::Int64(v) => assert_eq!(*v, 9), // 4 + 5
                    _ => panic!("expected int"),
                }
            }
            _ => panic!("expected aggregated"),
        }
    }
}
