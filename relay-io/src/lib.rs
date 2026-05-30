//! Storage adapters: Arrow IPC reader with mmap for zero-copy access.

pub mod csv;
pub mod csv_simd;
pub mod ipc;
pub mod madvise;
pub mod mmap;
pub mod parquet;

pub use csv::{CsvReadOptions, CsvReader};
pub use ipc::{write_ipc, write_single_batch, IPCWriteOptions};
pub use madvise::AccessPattern;
pub use mmap::MmapIPCReader;
pub use parquet::ParquetReader;
