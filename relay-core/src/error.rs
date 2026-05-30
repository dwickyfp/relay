//! Error types for Relay engine.

use thiserror::Error;

/// Primary error type for all Relay operations.
#[derive(Error, Debug)]
pub enum RelayError {
    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Type error: expected {expected}, got {actual}")]
    Type { expected: String, actual: String },

    #[error("Index out of bounds: index {index} but len is {len}")]
    OutOfBounds { index: usize, len: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Expression error: {0}")]
    Expr(String),
}

/// Result type alias using RelayError.
pub type Result<T> = std::result::Result<T, RelayError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_error_display() {
        let err = RelayError::Schema("missing column x".to_string());
        assert_eq!(format!("{}", err), "Schema error: missing column x");
    }

    #[test]
    fn test_type_error_display() {
        let err = RelayError::Type {
            expected: "Int64".to_string(),
            actual: "Utf8".to_string(),
        };
        assert!(format!("{}", err).contains("expected Int64"));
        assert!(format!("{}", err).contains("got Utf8"));
    }

    #[test]
    fn test_oob_error_display() {
        let err = RelayError::OutOfBounds { index: 5, len: 3 };
        assert!(format!("{}", err).contains("index 5"));
        assert!(format!("{}", err).contains("len is 3"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let relay_err: RelayError = io_err.into();
        assert!(matches!(relay_err, RelayError::Io(_)));
    }

    #[test]
    fn test_not_implemented_error() {
        let err = RelayError::NotImplemented("GPU execution".to_string());
        assert!(format!("{}", err).contains("GPU execution"));
    }

    #[test]
    fn test_invalid_argument_error() {
        let err = RelayError::InvalidArgument("negative batch size".to_string());
        assert!(format!("{}", err).contains("negative batch size"));
    }

    #[test]
    fn test_execution_error() {
        let err = RelayError::Execution("join failed".to_string());
        assert!(format!("{}", err).contains("join failed"));
    }

    #[test]
    fn test_memory_error() {
        let err = RelayError::Memory("allocation failed".to_string());
        assert!(format!("{}", err).contains("allocation failed"));
    }
}
