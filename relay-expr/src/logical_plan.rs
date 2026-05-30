//! Logical plan IR — tree of operators represented as Arena nodes.
//!
//! Each node is a `LogicalOp` stored in an `Arena<LogicalOp>`.
//! Children are referenced by `NodeId` — no Rc/Arc, no cloning overhead.

use arrow_schema::SchemaRef;

use crate::arena::{Arena, NodeId};
use crate::expr::Expr;

/// A logical plan operator node.
#[derive(Debug, Clone)]
pub enum LogicalOp {
    /// Scan a data source (IPC file, Parquet file, CSV, etc.)
    Scan {
        source: ScanSource,
        /// Schema of the source.
        schema: SchemaRef,
        /// Optional column projection (indices into source schema).
        projection: Option<Vec<usize>>,
        /// Optional row limit.
        limit: Option<usize>,
    },

    /// Filter rows by a predicate expression.
    Filter {
        input: NodeId,
        predicate: Expr,
    },

    /// Project (select) specific columns.
    Project {
        input: NodeId,
        columns: Vec<String>,
        /// Whether this is a "fast projection" (just column reorder, no expressions).
        is_reorder_only: bool,
    },

    /// Global aggregation (no GROUP BY).
    Aggregate {
        input: NodeId,
        /// Aggregation expressions: (column, op, alias).
        aggs: Vec<(String, crate::agg::AggOp, String)>,
    },

    /// Grouped aggregation (GROUP BY).
    GroupBy {
        input: NodeId,
        /// Group-by column names.
        keys: Vec<String>,
        /// Aggregation expressions: (column, op, alias).
        aggs: Vec<(String, crate::agg::AggOp, String)>,
    },

    /// Sort by one or more columns.
    Sort {
        input: NodeId,
        /// (column_name, ascending).
        by: Vec<(String, bool)>,
    },

    /// Limit the number of output rows.
    Limit {
        input: NodeId,
        n: usize,
        /// Optional offset (skip first N rows).
        offset: usize,
    },

    /// Join two inputs.
    Join {
        left: NodeId,
        right: NodeId,
        /// Join type.
        join_type: JoinType,
        /// Join condition (ON clause).
        on: Vec<(String, String)>,
    },

    /// Union of multiple inputs (UNION ALL).
    Union {
        inputs: Vec<NodeId>,
    },

    /// Empty relation (zero rows). Used by optimizer for dead branches.
    Empty {
        schema: SchemaRef,
    },
}

/// Data source types.
#[derive(Debug, Clone)]
pub enum ScanSource {
    /// Arrow IPC file (mmap).
    Ipc(String),
    /// Parquet file.
    Parquet(String),
    /// CSV file.
    Csv(String),
    /// JSON/NDJSON file.
    Json(String),
    /// In-memory RecordBatch (for testing).
    Memory(String),
}

impl ScanSource {
    pub fn path(&self) -> &str {
        match self {
            ScanSource::Ipc(p) => p,
            ScanSource::Parquet(p) => p,
            ScanSource::Csv(p) => p,
            ScanSource::Json(p) => p,
            ScanSource::Memory(name) => name,
        }
    }
}

/// Join types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Semi,
    Anti,
}

/// A logical plan is a reference to the root node in an Arena.
#[derive(Debug, Clone)]
pub struct LogicalPlan {
    /// The arena containing all plan nodes.
    pub arena: Arena<LogicalOp>,
    /// The root node of the plan.
    pub root: NodeId,
}

impl LogicalPlan {
    /// Create a new logical plan from an arena and root node.
    pub fn new(arena: Arena<LogicalOp>, root: NodeId) -> Self {
        Self { arena, root }
    }

    /// Get the root operator.
    pub fn root_op(&self) -> &LogicalOp {
        self.arena.get(self.root)
    }

    /// Collect all column names referenced in filter predicates in this plan.
    pub fn filter_columns(&self) -> Vec<String> {
        let mut cols = Vec::new();
        for (_, op) in self.arena.iter() {
            if let LogicalOp::Filter { predicate, .. } = op {
                cols.extend(predicate.referenced_columns());
            }
        }
        cols.sort();
        cols.dedup();
        cols
    }

    /// Pretty-print the plan tree.
    pub fn display(&self) -> String {
        let mut out = String::new();
        self.display_node(self.root, 0, &mut out);
        out
    }

    fn display_node(&self, node: NodeId, depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        match self.arena.get(node) {
            LogicalOp::Scan { source, projection, limit, .. } => {
                out.push_str(&format!("{}Scan({}", indent, source.path()));
                if let Some(proj) = projection {
                    out.push_str(&format!(", proj={:?}", proj));
                }
                if let Some(lim) = limit {
                    out.push_str(&format!(", limit={}", lim));
                }
                out.push_str(")\n");
            }
            LogicalOp::Filter { input, predicate } => {
                out.push_str(&format!("{}Filter({})\n", indent, predicate));
                self.display_node(*input, depth + 1, out);
            }
            LogicalOp::Project { input, columns, .. } => {
                out.push_str(&format!("{}Project({:?})\n", indent, columns));
                self.display_node(*input, depth + 1, out);
            }
            LogicalOp::Aggregate { input, aggs, .. } => {
                let agg_strs: Vec<String> = aggs
                    .iter()
                    .map(|(col, op, alias)| format!("{}({}) as {}", op, col, alias))
                    .collect();
                out.push_str(&format!("{}Aggregate({})\n", indent, agg_strs.join(", ")));
                self.display_node(*input, depth + 1, out);
            }
            LogicalOp::GroupBy { input, keys, aggs, .. } => {
                let agg_strs: Vec<String> = aggs
                    .iter()
                    .map(|(col, op, alias)| format!("{}({}) as {}", op, col, alias))
                    .collect();
                out.push_str(&format!(
                    "{}GroupBy(keys={:?}, aggs={})\n",
                    indent,
                    keys,
                    agg_strs.join(", ")
                ));
                self.display_node(*input, depth + 1, out);
            }
            LogicalOp::Sort { input, by } => {
                out.push_str(&format!("{}Sort({:?})\n", indent, by));
                self.display_node(*input, depth + 1, out);
            }
            LogicalOp::Limit { input, n, offset } => {
                out.push_str(&format!("{}Limit(n={}, offset={})\n", indent, n, offset));
                self.display_node(*input, depth + 1, out);
            }
            LogicalOp::Join {
                left,
                right,
                join_type,
                on,
            } => {
                out.push_str(&format!(
                    "{}Join({:?}, on={:?})\n",
                    indent, join_type, on
                ));
                self.display_node(*left, depth + 1, out);
                self.display_node(*right, depth + 1, out);
            }
            LogicalOp::Union { inputs } => {
                out.push_str(&format!("{}Union({} inputs)\n", indent, inputs.len()));
                for input in inputs {
                    self.display_node(*input, depth + 1, out);
                }
            }
            LogicalOp::Empty { .. } => {
                out.push_str(&format!("{}Empty\n", indent));
            }
        }
    }
}

impl std::fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]))
    }

    #[test]
    fn test_build_plan() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let filter = arena.add(LogicalOp::Filter {
            input: scan,
            predicate: Expr::col("id").gt(Expr::lit_i64(10)),
        });
        let project = arena.add(LogicalOp::Project {
            input: filter,
            columns: vec!["id".into(), "name".into()],
            is_reorder_only: true,
        });
        let plan = LogicalPlan::new(arena, project);
        let display = plan.display();
        assert!(display.contains("Project"));
        assert!(display.contains("Filter"));
        assert!(display.contains("Scan"));
    }
}
