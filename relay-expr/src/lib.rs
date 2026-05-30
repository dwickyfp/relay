//! Expression DSL, type inference, and vectorized evaluation engine.
//!
//! Provides column expressions, comparison operators, and aggregations
//! that execute directly on Arrow arrays — no Python loops.

pub mod agg;
pub mod expr;
pub mod filter;
pub mod plan;

pub use agg::{aggregate_array, AggOp, AggResult};
pub use expr::{Expr, Literal, Operator};
pub use filter::{eval_filter, filter_batch};
pub use plan::QueryPlan;
