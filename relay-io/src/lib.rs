//! Storage adapters: Arrow IPC reader with mmap for zero-copy access.

pub mod mmap;
pub mod madvise;
pub mod ipc;

pub use mmap::MmapIPCReader;
pub use madvise::AccessPattern;
pub use ipc::{write_ipc, write_single_batch, IPCWriteOptions};
