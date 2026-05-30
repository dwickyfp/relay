//! Memory-mapped file access pattern hints.
//!
//! Applies madvise(2) hints to the OS for better page caching behavior.

use memmap2::Mmap;

/// Access pattern hint for memory-mapped files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential scan — aggressive readahead (MADV_SEQUENTIAL).
    Sequential,
    /// Random access — disable readahead (MADV_RANDOM).
    Random,
    /// Prefetch specific pages (MADV_WILLNEED).
    Prefetch,
    /// Release pages when done (MADV_DONTNEED).
    Release,
    /// No specific hint — let the OS decide (default).
    Normal,
}

#[allow(clippy::derivable_impls)]
impl Default for AccessPattern {
    fn default() -> Self {
        Self::Sequential
    }
}

/// Apply madvise hints to the mmap region.
///
/// On macOS/Linux, this hints to the kernel how the application will access
/// the mapped region, allowing it to optimize page caching and readahead.
pub fn apply_madvise(mmap: &Mmap, pattern: AccessPattern) {
    let advice = match pattern {
        AccessPattern::Sequential => libc::MADV_SEQUENTIAL,
        AccessPattern::Random => libc::MADV_RANDOM,
        AccessPattern::Normal => libc::MADV_NORMAL,
        // WILLNEED/DONTNEED are applied on specific regions, not the whole mmap
        _ => return,
    };

    unsafe {
        libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), advice);
    }
}

/// Prefetch specific pages in the mmap region.
#[allow(dead_code)]
pub fn prefetch_pages(mmap: &Mmap, offset: usize, length: usize) {
    if offset + length > mmap.len() {
        return;
    }
    unsafe {
        libc::madvise(
            mmap.as_ptr().add(offset) as *mut libc::c_void,
            length,
            libc::MADV_WILLNEED,
        );
    }
}

/// Release specific pages from the mmap region.
#[allow(dead_code)]
pub fn release_pages(mmap: &Mmap, offset: usize, length: usize) {
    if offset + length > mmap.len() {
        return;
    }
    unsafe {
        libc::madvise(
            mmap.as_ptr().add(offset) as *mut libc::c_void,
            length,
            libc::MADV_DONTNEED,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapOptions;
    use std::fs::File;
    use tempfile::NamedTempFile;

    fn create_temp_mmap(data: &[u8]) -> (NamedTempFile, Mmap) {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), data).unwrap();
        let file = File::open(tmp.path()).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        (tmp, mmap)
    }

    #[test]
    fn test_madvise_sequential() {
        let data = vec![42u8; 4096];
        let (_tmp, mmap) = create_temp_mmap(&data);
        // Should not panic
        apply_madvise(&mmap, AccessPattern::Sequential);
    }

    #[test]
    fn test_madvise_random() {
        let data = vec![42u8; 4096];
        let (_tmp, mmap) = create_temp_mmap(&data);
        apply_madvise(&mmap, AccessPattern::Random);
    }

    #[test]
    fn test_madvise_normal() {
        let data = vec![42u8; 4096];
        let (_tmp, mmap) = create_temp_mmap(&data);
        apply_madvise(&mmap, AccessPattern::Normal);
    }

    #[test]
    fn test_prefetch_within_bounds() {
        let data = vec![42u8; 8192];
        let (_tmp, mmap) = create_temp_mmap(&data);
        // Should not panic for valid range
        prefetch_pages(&mmap, 0, 4096);
    }

    #[test]
    fn test_prefetch_out_of_bounds() {
        let data = vec![42u8; 4096];
        let (_tmp, mmap) = create_temp_mmap(&data);
        // Should silently skip for out-of-bounds
        prefetch_pages(&mmap, 0, 999999);
    }

    #[test]
    fn test_release_pages() {
        let data = vec![42u8; 8192];
        let (_tmp, mmap) = create_temp_mmap(&data);
        release_pages(&mmap, 0, 4096);
    }

    #[test]
    fn test_default_pattern() {
        assert_eq!(AccessPattern::default(), AccessPattern::Sequential);
    }
}
