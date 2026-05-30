//! # Relay Arrow
//!
//! Apache Arrow integration for Relay — array creation, manipulation,
//! zero-copy operations, and FFI bridge.

pub mod array;
pub mod builder;
pub mod ops;

pub use array::RelayArray;
pub use builder::ArrayBuilder;

// Re-export core error types
pub use relay_core::{RelayError, Result};
