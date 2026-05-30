//! Expression DSL, type inference, and vectorized evaluation engine.
//!
//! Provides column expressions, comparison operators, and aggregations
//! that execute directly on Arrow arrays — no Python loops.

pub mod agg;
pub mod arena;
pub mod expr;
pub mod filter;
pub mod logical_plan;
pub mod optimizer;
pub mod plan;

pub use agg::{aggregate_array, AggOp, AggResult};
pub use arena::{Arena, NodeId};
pub use expr::{Expr, Literal, Operator};
pub use filter::{eval_filter, filter_batch};
pub use logical_plan::{JoinType, LogicalOp, LogicalPlan, ScanSource};
pub use optimizer::{ApplyOrder, Optimizer, OptimizerRule, Transformed};
pub use plan::QueryPlan;
