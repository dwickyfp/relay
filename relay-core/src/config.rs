//! Configuration for Relay engine.

/// Global configuration for the Relay engine.
#[derive(Debug, Clone)]
pub struct RelayConfig {
    pub num_workers: usize,
    pub morsel_size: usize,
    pub memory_budget: usize,
    pub buffer_block_size: usize,
    pub enable_spill: bool,
    pub spill_dir: Option<String>,
    pub mmap_strategy: MmapStrategy,
    pub enable_simd: bool,
    pub enable_optimizer: bool,
    pub enable_mem_trace: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapStrategy {
    Eager,
    Lazy,
    BufferPool,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            num_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            morsel_size: 2048,
            memory_budget: 0,              // auto-detect at runtime
            buffer_block_size: 256 * 1024, // 256KB
            enable_spill: true,
            spill_dir: None,
            mmap_strategy: MmapStrategy::Lazy,
            enable_simd: true,
            enable_optimizer: true,
            enable_mem_trace: false,
        }
    }
}

impl RelayConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }
    pub fn with_morsel_size(mut self, size: usize) -> Self {
        self.morsel_size = size;
        self
    }
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = bytes;
        self
    }
    pub fn with_mem_trace(mut self, enable: bool) -> Self {
        self.enable_mem_trace = enable;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RelayConfig::default();
        assert!(config.num_workers > 0);
        assert_eq!(config.morsel_size, 2048);
        assert_eq!(config.buffer_block_size, 256 * 1024);
        assert!(config.enable_spill);
        assert!(config.enable_simd);
        assert!(config.enable_optimizer);
        assert!(!config.enable_mem_trace);
    }

    #[test]
    fn test_builder_pattern() {
        let config = RelayConfig::new()
            .with_workers(8)
            .with_morsel_size(4096)
            .with_memory_budget(8 * 1024 * 1024 * 1024)
            .with_mem_trace(true);
        assert_eq!(config.num_workers, 8);
        assert_eq!(config.morsel_size, 4096);
        assert_eq!(config.memory_budget, 8 * 1024 * 1024 * 1024);
        assert!(config.enable_mem_trace);
    }

    #[test]
    fn test_mmap_strategy() {
        let config = RelayConfig::default();
        assert_eq!(config.mmap_strategy, MmapStrategy::Lazy);
    }

    #[test]
    fn test_config_clone() {
        let config = RelayConfig::new().with_workers(16);
        let cloned = config.clone();
        assert_eq!(config.num_workers, cloned.num_workers);
    }
}
