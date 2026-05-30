//! Expression AST and core types.

use std::fmt;

/// A literal value that can be used in expressions.
#[derive(Debug, Clone)]
pub enum Literal {
    Int32(i32),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Str(String),
    Null,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int32(v) => write!(f, "{}", v),
            Literal::Int64(v) => write!(f, "{}", v),
            Literal::Float64(v) => write!(f, "{}", v),
            Literal::Bool(v) => write!(f, "{}", v),
            Literal::Str(v) => write!(f, "'{}'", v),
            Literal::Null => write!(f, "NULL"),
        }
    }
}

/// Comparison and logical operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Eq => write!(f, "=="),
            Operator::Ne => write!(f, "!="),
            Operator::Lt => write!(f, "<"),
            Operator::Le => write!(f, "<="),
            Operator::Gt => write!(f, ">"),
            Operator::Ge => write!(f, ">="),
            Operator::And => write!(f, "AND"),
            Operator::Or => write!(f, "OR"),
        }
    }
}

/// Expression tree node.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Reference to a column by name.
    Column(String),
    /// A literal value.
    Literal(Literal),
    /// Binary operation: left op right.
    BinaryOp {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    /// NOT expression.
    Not(Box<Expr>),
}

impl Expr {
    /// Create a column reference.
    pub fn col(name: &str) -> Self {
        Expr::Column(name.to_string())
    }

    /// Create a literal expression.
    pub fn lit_i32(v: i32) -> Self {
        Expr::Literal(Literal::Int32(v))
    }

    pub fn lit_i64(v: i64) -> Self {
        Expr::Literal(Literal::Int64(v))
    }

    pub fn lit_f64(v: f64) -> Self {
        Expr::Literal(Literal::Float64(v))
    }

    pub fn lit_bool(v: bool) -> Self {
        Expr::Literal(Literal::Bool(v))
    }

    pub fn lit_str(v: &str) -> Self {
        Expr::Literal(Literal::Str(v.to_string()))
    }

    /// Shorthand for comparison operations.
    pub fn eq(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Eq,
            right: Box::new(other),
        }
    }

    pub fn ne(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Ne,
            right: Box::new(other),
        }
    }

    pub fn lt(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Lt,
            right: Box::new(other),
        }
    }

    pub fn le(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Le,
            right: Box::new(other),
        }
    }

    pub fn gt(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Gt,
            right: Box::new(other),
        }
    }

    pub fn ge(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Ge,
            right: Box::new(other),
        }
    }

    pub fn and(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::And,
            right: Box::new(other),
        }
    }

    pub fn or(self, other: Expr) -> Self {
        Expr::BinaryOp {
            left: Box::new(self),
            op: Operator::Or,
            right: Box::new(other),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Expr::Not(Box::new(self))
    }

    /// Collect all column names referenced in this expression.
    pub fn referenced_columns(&self) -> Vec<String> {
        let mut cols = Vec::new();
        self.collect_columns(&mut cols);
        cols.sort();
        cols.dedup();
        cols
    }

    fn collect_columns(&self, cols: &mut Vec<String>) {
        match self {
            Expr::Column(name) => cols.push(name.clone()),
            Expr::Literal(_) => {}
            Expr::BinaryOp { left, right, .. } => {
                left.collect_columns(cols);
                right.collect_columns(cols);
            }
            Expr::Not(inner) => inner.collect_columns(cols),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "{}", name),
            Expr::Literal(lit) => write!(f, "{}", lit),
            Expr::BinaryOp { left, op, right } => write!(f, "({} {} {})", left, op, right),
            Expr::Not(inner) => write!(f, "NOT ({})", inner),
        }
    }
}
