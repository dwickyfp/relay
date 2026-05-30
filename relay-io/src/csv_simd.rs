//! SWAR (SIMD Within A Register) accelerated CSV structural scanning.
//!
//! Processes 8 bytes at a time using u64 word operations to find:
//! - Delimiter positions (commas)
//! - Quote positions (double quotes)
//! - Newline positions (LF)
//!
//! Then uses prefix-XOR (quote-parity monoid) to determine which
//! structural characters are inside quoted fields vs outside.
//!
//! Reference: "Biscuit: A Framework for Producing and Consuming Data-Parallel
//! Bit Streams" — Marmalade / DuckDB CSV parser approach.

/// Broadcast a single byte to fill a u64 word.
#[inline(always)]
const fn broadcast(b: u8) -> u64 {
    0x0101_0101_0101_0101u64 * (b as u64)
}

/// High-bit mask for each byte lane: 0x80 in every byte position.
const HI: u64 = 0x8080_8080_8080_8080u64;

/// Low-bit mask for each byte lane: 0x01 in every byte position.
const LO: u64 = 0x0101_0101_0101_0101u64;

/// Detect bytes equal to `target` in a u64 word.
/// Returns a u64 where the high bit of each byte lane is set if that byte
/// matches `target`.
#[inline(always)]
fn eq_bytes(word: u64, target: u8) -> u64 {
    let xored = word ^ broadcast(target);
    // Bytes that are zero after XOR match the target.
    // Standard SWAR zero-byte detection:
    !xored & (xored.wrapping_sub(LO)) & HI
}

/// Compress high bits from a u64 into a contiguous bitmask.
/// Each byte lane's high bit maps to one bit in the output.
/// Output: bit 0 = byte 0's high bit, bit 7 = byte 7's high bit.
#[inline(always)]
fn compress_hi(mask: u64) -> u8 {
    let mut result: u8 = 0;
    for i in 0..8u8 {
        result |= (((mask >> (i as u64 * 8 + 7)) & 1) as u8) << i;
    }
    result
}

/// Find all positions of a target byte within a data slice.
/// Returns a Vec of byte positions where `data[pos] == target`.
pub fn find_byte_positions(data: &[u8], target: u8) -> Vec<usize> {
    let mut positions = Vec::with_capacity(data.len() / 16); // rough estimate
    let mut i = 0;

    // Process 8 bytes at a time
    while i + 8 <= data.len() {
        let word = u64::from_ne_bytes(data[i..i + 8].try_into().unwrap());
        let mask = eq_bytes(word, target);
        if mask != 0 {
            let bits = compress_hi(mask);
            for bit in 0..8u8 {
                if bits & (1 << bit) != 0 {
                    positions.push(i + bit as usize);
                }
            }
        }
        i += 8;
    }

    // Handle remaining bytes
    while i < data.len() {
        if data[i] == target {
            positions.push(i);
        }
        i += 1;
    }

    positions
}

/// Compute quote-parity mask using prefix XOR.
///
/// For each structural character position, determines whether it's inside
/// a quoted field (odd number of preceding quotes) or outside (even).
///
/// This is the "quote-parity monoid" from the Marmalade paper:
/// - Scan quotes left to right
/// - Maintain a running parity (XOR)
/// - Structural chars at odd parity are inside quotes → skip them
///
/// `quote_positions`: sorted positions of all quote characters in the data
/// `total_len`: total length of the data
///
/// Returns a closure that, given a position, returns true if that position
/// is inside a quoted field.
pub struct QuoteParity {
    /// Cumulative quote count at each byte position.
    /// quote_count[i] = number of quotes in data[0..i].
    /// Position i is inside quotes iff quote_count[i] is odd.
    prefix_counts: Vec<u32>,
}

impl QuoteParity {
    /// Build prefix quote counts from quote positions.
    pub fn new(quote_positions: &[usize], total_len: usize) -> Self {
        let mut prefix_counts = vec![0u32; total_len + 1];
        for &pos in quote_positions {
            if pos < total_len {
                prefix_counts[pos + 1] = 1;
            }
        }
        // Prefix sum
        for i in 1..prefix_counts.len() {
            prefix_counts[i] += prefix_counts[i - 1];
        }
        Self { prefix_counts }
    }

    /// Returns true if the given position is inside a quoted field.
    #[inline(always)]
    pub fn is_quoted(&self, pos: usize) -> bool {
        self.prefix_counts[pos] & 1 == 1
    }
}

/// Find record boundaries (newline positions outside quotes) using SWAR.
///
/// Single-pass, allocation-free: processes 8 bytes at a time, tracks quote
/// parity on the fly, and emits only unquoted newlines.
pub fn find_record_boundaries_simd(data: &[u8], quote: u8) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(data.len() / 32); // rough estimate
    let mut in_quotes = false;
    let mut i = 0;

    // Process 8 bytes at a time
    while i + 8 <= data.len() {
        let word = u64::from_ne_bytes(data[i..i + 8].try_into().unwrap());

        let quote_mask = eq_bytes(word, quote);
        let newline_mask = eq_bytes(word, b'\n');

        // Fast path: no quotes and no newlines in this word
        if quote_mask == 0 && newline_mask == 0 {
            i += 8;
            continue;
        }

        // If there are no newlines, just update quote parity
        if newline_mask == 0 {
            if quote_mask != 0 {
                in_quotes ^= (compress_hi(quote_mask).count_ones() & 1) == 1;
            }
            i += 8;
            continue;
        }

        // Process byte-by-byte within this word (rare path)
        for j in 0..8 {
            let byte = data[i + j];
            if byte == quote {
                in_quotes = !in_quotes;
            } else if byte == b'\n' && !in_quotes {
                boundaries.push(i + j);
            }
        }
        i += 8;
    }

    // Handle remaining bytes
    while i < data.len() {
        if data[i] == quote {
            in_quotes = !in_quotes;
        } else if data[i] == b'\n' && !in_quotes {
            boundaries.push(i);
        }
        i += 1;
    }

    // Handle case where file doesn't end with newline
    if !data.is_empty() && *data.last().unwrap() != b'\n' {
        boundaries.push(data.len());
    }

    boundaries
}

/// Find delimiter positions within a single line (no newlines), respecting quotes.
///
/// Returns positions of delimiters that are outside quoted fields.
pub fn find_field_delimiters(data: &[u8], start: usize, end: usize, delimiter: u8, quote: u8) -> Vec<usize> {
    let line = &data[start..end];

    // Find all delimiter and quote positions in this line
    let delim_positions = find_byte_positions(line, delimiter);
    let quote_positions = find_byte_positions(line, quote);

    let qp = QuoteParity::new(&quote_positions, line.len());

    delim_positions
        .into_iter()
        .filter(|&pos| !qp.is_quoted(pos))
        .map(|pos| start + pos)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast() {
        assert_eq!(broadcast(b','), 0x2C2C_2C2C_2C2C_2C2Cu64);
    }

    #[test]
    fn test_eq_bytes_basic() {
        let word = u64::from_ne_bytes(*b"hello,wo");
        let mask = eq_bytes(word, b',');
        // Comma is at position 5
        let bits = compress_hi(mask);
        assert_eq!(bits & (1 << 5), 1 << 5);
        // No comma at position 0
        assert_eq!(bits & 1, 0);
    }

    #[test]
    fn test_find_byte_positions() {
        let data = b"a,b,c\n1,2,3\n";
        let commas = find_byte_positions(data, b',');
        assert_eq!(commas, vec![1, 3, 7, 9]);

        let newlines = find_byte_positions(data, b'\n');
        assert_eq!(newlines, vec![5, 11]);
    }

    #[test]
    fn test_find_byte_positions_no_target() {
        let data = b"abcdef";
        let positions = find_byte_positions(data, b'z');
        assert!(positions.is_empty());
    }

    #[test]
    fn test_quote_parity() {
        // Data: `"hello",world`
        // Quotes at positions 0 and 6
        let qp = QuoteParity::new(&[0, 6], 13);

        // Before first quote: not quoted
        assert!(!qp.is_quoted(0));
        // Inside quotes (positions 1-5): quoted
        assert!(qp.is_quoted(1));
        assert!(qp.is_quoted(5));
        // After second quote (position 7+): not quoted
        assert!(!qp.is_quoted(7));
        assert!(!qp.is_quoted(12));
    }

    #[test]
    fn test_find_record_boundaries_simd_basic() {
        let data = b"a,b\n1,2\n3,4\n";
        let bounds = find_record_boundaries_simd(data, b'"');
        assert_eq!(bounds, vec![3, 7, 11]);
    }

    #[test]
    fn test_find_record_boundaries_simd_quoted_newlines() {
        let data = b"a,b\n\"hello\nworld\",2\n";
        let bounds = find_record_boundaries_simd(data, b'"');
        assert_eq!(bounds, vec![3, 19]);
    }

    #[test]
    fn test_find_record_boundaries_simd_no_trailing_newline() {
        let data = b"a,b\n1,2";
        let bounds = find_record_boundaries_simd(data, b'"');
        assert_eq!(bounds, vec![3, 7]);
    }

    #[test]
    fn test_find_field_delimiters() {
        let data = b"a,b,c,d";
        let delims = find_field_delimiters(data, 0, 7, b',', b'"');
        assert_eq!(delims, vec![1, 3, 5]);
    }

    #[test]
    fn test_find_field_delimiters_quoted() {
        let data = b"\"a,b\",c,d";
        let delims = find_field_delimiters(data, 0, 9, b',', b'"');
        // The comma inside quotes should be skipped
        assert_eq!(delims, vec![5, 7]);
    }

    #[test]
    fn test_swar_large_data() {
        // Test with data larger than 8 bytes to exercise SWAR loop
        let mut data = Vec::new();
        for i in 0..100 {
            data.extend_from_slice(format!("{},{}\n", i, i * 2).as_bytes());
        }
        let bounds = find_record_boundaries_simd(&data, b'"');
        // Should have 100 record boundaries
        assert_eq!(bounds.len(), 100);
    }

    /// Benchmark: scalar vs SWAR record boundary detection.
    /// Run with: cargo test -p relay-io -- csv_simd::tests::bench_scalar_vs_simd --nocapture
    #[test]
    fn bench_scalar_vs_simd() {
        // Generate a ~1MB CSV file
        let mut data = Vec::with_capacity(1024 * 1024);
        data.extend_from_slice(b"id,name,value,category,description\n");
        for i in 0..10_000 {
            let line = format!(
                "{},item_{},{:.2},cat_{}\"with,quotes\",\"A description with, commas and \\\"quotes\\\"\"\n",
                i, i, (i as f64) * 1.5, i % 10
            );
            data.extend_from_slice(line.as_bytes());
        }

        let iterations = 100;

        // Scalar
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = super::super::csv::find_record_boundaries(&data, b'"');
        }
        let scalar_elapsed = start.elapsed();

        // SWAR
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = find_record_boundaries_simd(&data, b'"');
        }
        let simd_elapsed = start.elapsed();

        let speedup = scalar_elapsed.as_nanos() as f64 / simd_elapsed.as_nanos() as f64;
        println!(
            "\n  CSV boundary detection ({} iters, {:.1} KB):",
            iterations,
            data.len() as f64 / 1024.0
        );
        println!("    Scalar: {:>8.2} ms", scalar_elapsed.as_secs_f64() * 1000.0);
        println!("    SWAR:   {:>8.2} ms", simd_elapsed.as_secs_f64() * 1000.0);
        println!("    Speedup: {:.2}x", speedup);
    }
}
