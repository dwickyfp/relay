//! # Relay Core
//!
//! Core types, schema definitions, error handling, and shared utilities
//! for the Relay zero-copy data engine.

pub mod config;
pub mod error;
pub mod schema;
pub mod types;

pub use config::RelayConfig;
pub use error::{RelayError, Result};
pub use schema::{RelayField, RelaySchema};
pub use types::RelayType;
