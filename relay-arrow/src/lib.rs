//! # Relay Arrow
//!
//! Apache Arrow integration for Relay — array creation, manipulation,
//! zero-copy operations, RecordBatch, and FFI bridge.

pub mod array;
pub mod builder;
pub mod ffi;
pub mod ops;
pub mod recordbatch;

pub use array::RelayArray;
pub use builder::ArrayBuilder;
pub use recordbatch::RelayRecordBatch;

// Re-export core error types
pub use relay_core::{RelayError, Result};
