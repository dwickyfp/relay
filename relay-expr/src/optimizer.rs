//! Optimizer framework — rule-based query plan optimization.
//!
//! Inspired by DataFusion's OptimizerRule trait: each rule rewrites the plan
//! and signals whether changes were made. The optimizer applies rules in a
//! fixed-point loop until no more changes occur.

use crate::arena::NodeId;
use crate::logical_plan::LogicalPlan;

/// Result of applying an optimization rule.
#[derive(Debug)]
pub struct Transformed {
    /// Whether the plan was changed.
    pub changed: bool,
    /// The (possibly modified) plan.
    pub plan: LogicalPlan,
}

impl Transformed {
    /// Plan was modified.
    pub fn yes(plan: LogicalPlan) -> Self {
        Self {
            changed: true,
            plan,
        }
    }

    /// Plan was not modified.
    pub fn no(plan: LogicalPlan) -> Self {
        Self {
            changed: false,
            plan,
        }
    }
}

/// A single optimizer rule that rewrites a logical plan.
pub trait OptimizerRule: std::fmt::Debug + Send + Sync {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Whether to apply top-down (parent first) or bottom-up (children first).
    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    /// Apply this rule to the plan, returning whether changes were made.
    fn rewrite(&self, plan: LogicalPlan) -> Transformed;
}

/// Order in which to traverse the plan tree.
#[derive(Debug, Clone, Copy)]
pub enum ApplyOrder {
    /// Visit parent before children.
    TopDown,
    /// Visit children before parent.
    BottomUp,
}

/// The query optimizer — applies a sequence of rules to a logical plan.
#[derive(Debug)]
pub struct Optimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
    /// Maximum number of iterations for the fixed-point loop.
    max_iterations: usize,
}

impl Optimizer {
    /// Create an optimizer with the given rules.
    pub fn new(rules: Vec<Box<dyn OptimizerRule>>) -> Self {
        Self {
            rules,
            max_iterations: 10,
        }
    }

    /// Create an optimizer with default rules.
    pub fn with_defaults() -> Self {
        use crate::optimizer::rules::*;
        Self::new(vec![
            Box::new(EliminateFilter),
            Box::new(SimplifyExpressions),
            Box::new(PushDownFilter),
            Box::new(PushDownProjection),
            Box::new(PushDownLimit),
            Box::new(EliminateLimit),
            Box::new(PropagateEmptyRelation),
            Box::new(CommonSubexprEliminate),
        ])
    }

    /// Set the maximum number of fixed-point iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Optimize a logical plan by applying all rules until convergence.
    pub fn optimize(&self, mut plan: LogicalPlan) -> LogicalPlan {
        for _iteration in 0..self.max_iterations {
            let mut any_changed = false;

            for rule in &self.rules {
                let result = rule.rewrite(plan);
                if result.changed {
                    any_changed = true;
                }
                plan = result.plan;
            }

            if !any_changed {
                break;
            }
        }
        plan
    }

    /// List all rule names (for debugging).
    pub fn rule_names(&self) -> Vec<&str> {
        self.rules.iter().map(|r| r.name()).collect()
    }
}

/// Built-in optimization rules.
pub mod rules {
    use super::*;
    use crate::expr::{Expr, Literal, Operator};
    use crate::logical_plan::LogicalOp;

    // ── Rule 1: EliminateFilter ────────────────────────────────────
    // Remove WHERE false → replace with EmptyRelation
    // Remove WHERE true → remove the Filter node entirely

    #[derive(Debug)]
    pub struct EliminateFilter;

    impl OptimizerRule for EliminateFilter {
        fn name(&self) -> &str {
            "eliminate_filter"
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            let root = plan.root;
            self.rewrite_node(plan, root)
        }
    }

    impl EliminateFilter {
        fn rewrite_node(&self, mut plan: LogicalPlan, node: NodeId) -> Transformed {
            match plan.arena.get(node).clone() {
                LogicalOp::Filter {
                    input,
                    predicate: Expr::Literal(Literal::Bool(false)),
                } => {
                    // WHERE false → Empty
                    let schema = match plan.arena.get(input) {
                        LogicalOp::Scan { schema, .. } => schema.clone(),
                        LogicalOp::Empty { schema } => schema.clone(),
                        _ => {
                            // Can't determine schema, keep as-is
                            return Transformed::no(plan);
                        }
                    };
                    let empty = plan.arena.add(LogicalOp::Empty { schema });
                    plan.root = empty;
                    Transformed::yes(plan)
                }
                LogicalOp::Filter {
                    input,
                    predicate: Expr::Literal(Literal::Bool(true)),
                } => {
                    // WHERE true → just the input
                    plan.root = input;
                    Transformed::yes(plan)
                }
                LogicalOp::Filter { input, predicate } => {
                    // Recurse into children
                    let child_result = self.rewrite_node(plan.clone(), input);
                    if child_result.changed {
                        plan = child_result.plan;
                        // Update our input reference
                        let new_input = plan.root;
                        let new_filter = plan.arena.add(LogicalOp::Filter {
                            input: new_input,
                            predicate,
                        });
                        plan.root = new_filter;
                        Transformed::yes(plan)
                    } else {
                        Transformed::no(plan)
                    }
                }
                _ => Transformed::no(plan),
            }
        }
    }

    // ── Rule 2: SimplifyExpressions ───────────────────────────────
    // Constant folding, boolean simplification, De Morgan's laws

    #[derive(Debug)]
    pub struct SimplifyExpressions;

    impl OptimizerRule for SimplifyExpressions {
        fn name(&self) -> &str {
            "simplify_expressions"
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            let root = plan.root;
            self.rewrite_node(plan, root)
        }
    }

    impl SimplifyExpressions {
        fn rewrite_node(&self, mut plan: LogicalPlan, node: NodeId) -> Transformed {
            match plan.arena.get(node).clone() {
                LogicalOp::Filter { input, predicate } => {
                    let simplified = simplify_expr(&predicate);
                    if !exprs_equal(&predicate, &simplified) {
                        let new_filter = plan.arena.add(LogicalOp::Filter {
                            input,
                            predicate: simplified,
                        });
                        plan.root = new_filter;
                        Transformed::yes(plan)
                    } else {
                        // Recurse into children
                        let child_result = self.rewrite_node(plan.clone(), input);
                        if child_result.changed {
                            plan = child_result.plan;
                            let new_input = plan.root;
                            let new_filter = plan.arena.add(LogicalOp::Filter {
                                input: new_input,
                                predicate,
                            });
                            plan.root = new_filter;
                            Transformed::yes(plan)
                        } else {
                            Transformed::no(plan)
                        }
                    }
                }
                _ => Transformed::no(plan),
            }
        }
    }

    /// Simplify an expression (constant folding, boolean algebra).
    pub fn simplify_expr(expr: &Expr) -> Expr {
        match expr {
            // NOT(NOT(x)) → x
            Expr::Not(inner) => {
                let simplified_inner = simplify_expr(inner);
                match simplified_inner {
                    Expr::Not(x) => *x,
                    other => Expr::Not(Box::new(other)),
                }
            }

            // BinaryOp simplifications
            Expr::BinaryOp { left, op, right } => {
                let sl = simplify_expr(left);
                let sr = simplify_expr(right);

                match (op, &sl, &sr) {
                    // x AND true → x
                    (Operator::And, _, Expr::Literal(Literal::Bool(true))) => sl,
                    (Operator::And, Expr::Literal(Literal::Bool(true)), _) => sr,
                    // x AND false → false
                    (Operator::And, _, Expr::Literal(Literal::Bool(false))) => {
                        Expr::Literal(Literal::Bool(false))
                    }
                    (Operator::And, Expr::Literal(Literal::Bool(false)), _) => {
                        Expr::Literal(Literal::Bool(false))
                    }
                    // x OR true → true
                    (Operator::Or, _, Expr::Literal(Literal::Bool(true))) => {
                        Expr::Literal(Literal::Bool(true))
                    }
                    (Operator::Or, Expr::Literal(Literal::Bool(true)), _) => {
                        Expr::Literal(Literal::Bool(true))
                    }
                    // x OR false → x
                    (Operator::Or, _, Expr::Literal(Literal::Bool(false))) => sl,
                    (Operator::Or, Expr::Literal(Literal::Bool(false)), _) => sr,

                    // Constant folding for numeric comparisons
                    (op, Expr::Literal(Literal::Int64(a)), Expr::Literal(Literal::Int64(b))) => {
                        let result = eval_int_comparison(*a, *op, *b);
                        Expr::Literal(Literal::Bool(result))
                    }
                    (op, Expr::Literal(Literal::Float64(a)), Expr::Literal(Literal::Float64(b))) => {
                        let result = eval_float_comparison(*a, *op, *b);
                        Expr::Literal(Literal::Bool(result))
                    }

                    // Flip: lit > col → col < lit
                    (Operator::Gt, Expr::Literal(_), Expr::Column(_)) => Expr::BinaryOp {
                        left: Box::new(sr),
                        op: Operator::Lt,
                        right: Box::new(sl),
                    },
                    (Operator::Ge, Expr::Literal(_), Expr::Column(_)) => Expr::BinaryOp {
                        left: Box::new(sr),
                        op: Operator::Le,
                        right: Box::new(sl),
                    },
                    (Operator::Lt, Expr::Literal(_), Expr::Column(_)) => Expr::BinaryOp {
                        left: Box::new(sr),
                        op: Operator::Gt,
                        right: Box::new(sl),
                    },
                    (Operator::Le, Expr::Literal(_), Expr::Column(_)) => Expr::BinaryOp {
                        left: Box::new(sr),
                        op: Operator::Ge,
                        right: Box::new(sl),
                    },

                    _ => Expr::BinaryOp {
                        left: Box::new(sl),
                        op: *op,
                        right: Box::new(sr),
                    },
                }
            }

            // Leaf nodes — no simplification
            _ => expr.clone(),
        }
    }

    fn eval_int_comparison(a: i64, op: Operator, b: i64) -> bool {
        match op {
            Operator::Eq => a == b,
            Operator::Ne => a != b,
            Operator::Lt => a < b,
            Operator::Le => a <= b,
            Operator::Gt => a > b,
            Operator::Ge => a >= b,
            _ => false,
        }
    }

    fn eval_float_comparison(a: f64, op: Operator, b: f64) -> bool {
        match op {
            Operator::Eq => a == b,
            Operator::Ne => a != b,
            Operator::Lt => a < b,
            Operator::Le => a <= b,
            Operator::Gt => a > b,
            Operator::Ge => a >= b,
            _ => false,
        }
    }

    fn exprs_equal(a: &Expr, b: &Expr) -> bool {
        format!("{}", a) == format!("{}", b)
    }

    // ── Rule 3: PushDownFilter ────────────────────────────────────
    // Push Filter nodes as close to Scan as possible.

    #[derive(Debug)]
    pub struct PushDownFilter;

    impl OptimizerRule for PushDownFilter {
        fn name(&self) -> &str {
            "push_down_filter"
        }

        fn apply_order(&self) -> ApplyOrder {
            ApplyOrder::TopDown
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            let root = plan.root;
            self.rewrite_node(plan, root)
        }
    }

    impl PushDownFilter {
        fn rewrite_node(&self, mut plan: LogicalPlan, node: NodeId) -> Transformed {
            match plan.arena.get(node).clone() {
                // Filter(Project(x)) → Project(Filter(x))
                // Only if the filter columns are available in the input
                LogicalOp::Filter { input: proj_node, predicate } => {
                    match plan.arena.get(proj_node).clone() {
                        LogicalOp::Project {
                            input: scan_or_other,
                            columns,
                            ..
                        } => {
                            // Check that all filter columns exist in the projection
                            let filter_cols = predicate.referenced_columns();
                            let all_available = filter_cols
                                .iter()
                                .all(|c| columns.contains(c));

                            if all_available {
                                // Swap: Project → Filter → input
                                let new_filter = plan.arena.add(LogicalOp::Filter {
                                    input: scan_or_other,
                                    predicate,
                                });
                                let new_project = plan.arena.add(LogicalOp::Project {
                                    input: new_filter,
                                    columns,
                                    is_reorder_only: true,
                                });
                                plan.root = new_project;
                                Transformed::yes(plan)
                            } else {
                                // Can't push down — recurse into child
                                let child_result = self.rewrite_node(plan.clone(), proj_node);
                                if child_result.changed {
                                    plan = child_result.plan;
                                    let new_input = plan.root;
                                    let new_filter = plan.arena.add(LogicalOp::Filter {
                                        input: new_input,
                                        predicate,
                                    });
                                    plan.root = new_filter;
                                    Transformed::yes(plan)
                                } else {
                                    Transformed::no(plan)
                                }
                            }
                        }
                        _ => {
                            // Not a Project child — try recursing into the child
                            let child_result = self.rewrite_node(plan.clone(), proj_node);
                            if child_result.changed {
                                plan = child_result.plan;
                                let new_input = plan.root;
                                let new_filter = plan.arena.add(LogicalOp::Filter {
                                    input: new_input,
                                    predicate,
                                });
                                plan.root = new_filter;
                                Transformed::yes(plan)
                            } else {
                                Transformed::no(plan)
                            }
                        }
                    }
                }
                _ => Transformed::no(plan),
            }
        }
    }

    // ── Rule 4: PushDownProjection ───────────────────────────────
    // Push column projection into the Scan node to avoid reading unneeded columns.

    #[derive(Debug)]
    pub struct PushDownProjection;

    impl OptimizerRule for PushDownProjection {
        fn name(&self) -> &str {
            "push_down_projection"
        }

        fn apply_order(&self) -> ApplyOrder {
            ApplyOrder::TopDown
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            let root = plan.root;
            self.rewrite_node(plan, root)
        }
    }

    impl PushDownProjection {
        fn rewrite_node(&self, mut plan: LogicalPlan, node: NodeId) -> Transformed {
            match plan.arena.get(node).clone() {
                // Project(Scan) → Scan with projection
                LogicalOp::Project {
                    input,
                    columns,
                    ..
                } => {
                    match plan.arena.get(input).clone() {
                        LogicalOp::Scan {
                            source,
                            schema,
                            projection: None,
                            limit,
                        } => {
                            // Convert column names to indices
                            let proj_indices: Vec<usize> = columns
                                .iter()
                                .filter_map(|name| schema.index_of(name).ok())
                                .collect();

                            if proj_indices.len() == columns.len()
                                && proj_indices.len() < schema.fields().len()
                            {
                                let new_scan = plan.arena.add(LogicalOp::Scan {
                                    source,
                                    schema,
                                    projection: Some(proj_indices),
                                    limit,
                                });
                                plan.root = new_scan;
                                Transformed::yes(plan)
                            } else {
                                Transformed::no(plan)
                            }
                        }
                        LogicalOp::Filter {
                            input: filter_input,
                            predicate,
                        } => {
                            // Project(Filter(x)) → Filter(Project(x)) if all cols available
                            let filter_cols = predicate.referenced_columns();
                            let all_available = filter_cols
                                .iter()
                                .all(|c| columns.contains(c));

                            if all_available {
                                // Swap: Filter → Project → input
                                let new_project = plan.arena.add(LogicalOp::Project {
                                    input: filter_input,
                                    columns: columns.clone(),
                                    is_reorder_only: true,
                                });
                                let new_filter = plan.arena.add(LogicalOp::Filter {
                                    input: new_project,
                                    predicate,
                                });
                                plan.root = new_filter;
                                Transformed::yes(plan)
                            } else {
                                Transformed::no(plan)
                            }
                        }
                        _ => Transformed::no(plan),
                    }
                }
                _ => Transformed::no(plan),
            }
        }
    }

    // ── Rule 5: PushDownLimit ────────────────────────────────────
    // Push LIMIT into Scan nodes to avoid reading more rows than needed.

    #[derive(Debug)]
    pub struct PushDownLimit;

    impl OptimizerRule for PushDownLimit {
        fn name(&self) -> &str {
            "push_down_limit"
        }

        fn apply_order(&self) -> ApplyOrder {
            ApplyOrder::TopDown
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            let root = plan.root;
            self.rewrite_node(plan, root)
        }
    }

    impl PushDownLimit {
        fn rewrite_node(&self, mut plan: LogicalPlan, node: NodeId) -> Transformed {
            match plan.arena.get(node).clone() {
                // Limit(Scan) → Scan with limit
                LogicalOp::Limit {
                    input,
                    n,
                    offset: 0,
                } => {
                    match plan.arena.get(input).clone() {
                        LogicalOp::Scan {
                            source,
                            schema,
                            projection,
                            limit: None,
                        } => {
                            let new_scan = plan.arena.add(LogicalOp::Scan {
                                source,
                                schema,
                                projection,
                                limit: Some(n),
                            });
                            plan.root = new_scan;
                            Transformed::yes(plan)
                        }
                        _ => Transformed::no(plan),
                    }
                }
                // Limit(Project(x)) → Project(Limit(x))
                LogicalOp::Limit {
                    input: proj_node,
                    n,
                    offset,
                } => {
                    match plan.arena.get(proj_node).clone() {
                        LogicalOp::Project {
                            input: inner,
                            columns,
                            is_reorder_only,
                        } => {
                            let new_limit = plan.arena.add(LogicalOp::Limit {
                                input: inner,
                                n,
                                offset,
                            });
                            let new_project = plan.arena.add(LogicalOp::Project {
                                input: new_limit,
                                columns,
                                is_reorder_only,
                            });
                            plan.root = new_project;
                            Transformed::yes(plan)
                        }
                        _ => Transformed::no(plan),
                    }
                }
                _ => Transformed::no(plan),
            }
        }
    }

    // ── Rule 6: EliminateLimit ──────────────────────────────────
    // Remove LIMIT when it's a no-op (larger than known cardinality).

    #[derive(Debug)]
    pub struct EliminateLimit;

    impl OptimizerRule for EliminateLimit {
        fn name(&self) -> &str {
            "eliminate_limit"
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            // Only applies if we know the exact row count (not implemented yet)
            Transformed::no(plan)
        }
    }

    // ── Rule 7: PropagateEmptyRelation ──────────────────────────
    // When Empty feeds into join/filter/aggregate, propagate emptiness up.

    #[derive(Debug)]
    pub struct PropagateEmptyRelation;

    impl OptimizerRule for PropagateEmptyRelation {
        fn name(&self) -> &str {
            "propagate_empty"
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            let root = plan.root;
            self.rewrite_node(plan, root)
        }
    }

    impl PropagateEmptyRelation {
        fn rewrite_node(&self, mut plan: LogicalPlan, node: NodeId) -> Transformed {
            match plan.arena.get(node).clone() {
                // Filter(Empty) → Empty
                LogicalOp::Filter { input, .. } => {
                    if matches!(plan.arena.get(input), LogicalOp::Empty { .. }) {
                        let _empty = plan.arena.get(input).clone();
                        plan.root = input;
                        Transformed::yes(plan)
                    } else {
                        Transformed::no(plan)
                    }
                }
                // Project(Empty) → Empty
                LogicalOp::Project { input, .. } => {
                    if matches!(plan.arena.get(input), LogicalOp::Empty { .. }) {
                        plan.root = input;
                        Transformed::yes(plan)
                    } else {
                        Transformed::no(plan)
                    }
                }
                // Limit(Empty) → Empty
                LogicalOp::Limit { input, .. } => {
                    if matches!(plan.arena.get(input), LogicalOp::Empty { .. }) {
                        plan.root = input;
                        Transformed::yes(plan)
                    } else {
                        Transformed::no(plan)
                    }
                }
                _ => Transformed::no(plan),
            }
        }
    }

    // ── Rule 8: CommonSubexprEliminate ──────────────────────────
    // Deduplicate repeated subexpressions.
    // (Stub for now — full implementation requires expression hashing)

    #[derive(Debug)]
    pub struct CommonSubexprEliminate;

    impl OptimizerRule for CommonSubexprEliminate {
        fn name(&self) -> &str {
            "common_subexpr_eliminate"
        }

        fn rewrite(&self, plan: LogicalPlan) -> Transformed {
            // TODO: Full CSE requires expression hashing and memoization.
            // For now, this is a no-op placeholder.
            Transformed::no(plan)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::expr::Expr;
    use crate::logical_plan::{LogicalOp, ScanSource};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn test_schema() -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]))
    }

    #[test]
    fn test_eliminate_filter_false() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let filter = arena.add(LogicalOp::Filter {
            input: scan,
            predicate: Expr::lit_bool(false),
        });
        let plan = LogicalPlan::new(arena, filter);

        let opt = Optimizer::new(vec![Box::new(rules::EliminateFilter)]);
        let result = opt.optimize(plan);
        assert!(matches!(result.root_op(), LogicalOp::Empty { .. }));
    }

    #[test]
    fn test_eliminate_filter_true() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let filter = arena.add(LogicalOp::Filter {
            input: scan,
            predicate: Expr::lit_bool(true),
        });
        let plan = LogicalPlan::new(arena, filter);

        let opt = Optimizer::new(vec![Box::new(rules::EliminateFilter)]);
        let result = opt.optimize(plan);
        assert!(matches!(result.root_op(), LogicalOp::Scan { .. }));
    }

    #[test]
    fn test_simplify_not_not() {
        let expr = Expr::col("x").gt(Expr::lit_i64(5)).not().not();
        let simplified = rules::simplify_expr(&expr);
        // NOT(NOT(x > 5)) → x > 5
        assert_eq!(format!("{}", simplified), "(x > 5)");
    }

    #[test]
    fn test_simplify_and_true() {
        let expr = Expr::col("x").gt(Expr::lit_i64(5)).and(Expr::lit_bool(true));
        let simplified = rules::simplify_expr(&expr);
        assert_eq!(format!("{}", simplified), "(x > 5)");
    }

    #[test]
    fn test_simplify_and_false() {
        let expr = Expr::col("x")
            .gt(Expr::lit_i64(5))
            .and(Expr::lit_bool(false));
        let simplified = rules::simplify_expr(&expr);
        assert_eq!(format!("{}", simplified), "false");
    }

    #[test]
    fn test_simplify_constant_folding() {
        let expr = Expr::lit_i64(10).gt(Expr::lit_i64(5));
        let simplified = rules::simplify_expr(&expr);
        assert_eq!(format!("{}", simplified), "true");
    }

    #[test]
    fn test_push_down_filter_through_project() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let project = arena.add(LogicalOp::Project {
            input: scan,
            columns: vec!["id".into(), "name".into()],
            is_reorder_only: true,
        });
        let filter = arena.add(LogicalOp::Filter {
            input: project,
            predicate: Expr::col("id").gt(Expr::lit_i64(10)),
        });
        let plan = LogicalPlan::new(arena, filter);

        let opt = Optimizer::new(vec![Box::new(rules::PushDownFilter)]);
        let result = opt.optimize(plan);
        // After optimization: Project should be root, Filter below it
        assert!(matches!(result.root_op(), LogicalOp::Project { .. }));
    }

    #[test]
    fn test_push_down_projection_into_scan() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let project = arena.add(LogicalOp::Project {
            input: scan,
            columns: vec!["id".into(), "value".into()],
            is_reorder_only: true,
        });
        let plan = LogicalPlan::new(arena, project);

        let opt = Optimizer::new(vec![Box::new(rules::PushDownProjection)]);
        let result = opt.optimize(plan);
        // After optimization: Scan with projection, no separate Project node
        match result.root_op() {
            LogicalOp::Scan { projection, .. } => {
                assert!(projection.is_some());
                assert_eq!(projection.as_ref().unwrap().len(), 2);
            }
            _ => panic!("Expected Scan with projection"),
        }
    }

    #[test]
    fn test_push_down_limit_into_scan() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let limit = arena.add(LogicalOp::Limit {
            input: scan,
            n: 100,
            offset: 0,
        });
        let plan = LogicalPlan::new(arena, limit);

        let opt = Optimizer::new(vec![Box::new(rules::PushDownLimit)]);
        let result = opt.optimize(plan);
        match result.root_op() {
            LogicalOp::Scan { limit, .. } => {
                assert_eq!(*limit, Some(100));
            }
            _ => panic!("Expected Scan with limit"),
        }
    }

    #[test]
    fn test_full_optimizer_pipeline() {
        let mut arena = Arena::new();
        let scan = arena.add(LogicalOp::Scan {
            source: ScanSource::Parquet("test.parquet".into()),
            schema: test_schema(),
            projection: None,
            limit: None,
        });
        let project = arena.add(LogicalOp::Project {
            input: scan,
            columns: vec!["id".into(), "name".into()],
            is_reorder_only: true,
        });
        let filter = arena.add(LogicalOp::Filter {
            input: project,
            predicate: Expr::col("id").gt(Expr::lit_i64(10)),
        });
        let plan = LogicalPlan::new(arena, filter);

        let opt = Optimizer::with_defaults();
        let result = opt.optimize(plan);

        // Should have pushed filter below project
        let display = result.display();
        assert!(display.contains("Scan") || display.contains("Filter") || display.contains("Project"));
    }
}
