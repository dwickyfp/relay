//! Relay I/O benchmark suite
//!
//! Tests:
//! - CSV: SWAR vs scalar boundary detection
//! - CSV: full read throughput at different sizes
//! - CSV: projection pushdown
//! - NDJSON: full read throughput
//! - Parquet: full read + late materialization filter

use std::io::Write;
use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use relay_io::csv::{CsvReadOptions, CsvReader};
use relay_io::csv_simd::find_record_boundaries_simd;
use relay_io::json::{JsonReadOptions, JsonReader};

// ─── Data Generators ──────────────────────────────────────────────

fn create_csv_file(n: usize, ncols: usize) -> PathBuf {
    let dir = tempfile::tempdir().unwrap().into_path();
    let path = dir.join("bench.csv");
    let mut f = std::fs::File::create(&path).unwrap();

    // Header
    let header: Vec<String> = (0..ncols).map(|i| format!("col_{}", i)).collect();
    writeln!(f, "{}", header.join(",")).unwrap();

    // Data rows
    for row in 0..n {
        let vals: Vec<String> = (0..ncols)
            .map(|c| format!("{}", row as i64 + c as i64))
            .collect();
        writeln!(f, "{}", vals.join(",")).unwrap();
    }

    path
}

fn create_csv_with_quotes(n: usize) -> PathBuf {
    let dir = tempfile::tempdir().unwrap().into_path();
    let path = dir.join("bench_quoted.csv");
    let mut f = std::fs::File::create(&path).unwrap();

    writeln!(f, "id,name,description,value").unwrap();
    for i in 0..n {
        writeln!(
            f,
            "{},\"item_{}\",\"A description with, commas and \\\"quotes\\\"\",{:.2}",
            i,
            i,
            (i as f64) * 1.5
        )
        .unwrap();
    }

    path
}

fn create_ndjson_file(n: usize, ncols: usize) -> PathBuf {
    let dir = tempfile::tempdir().unwrap().into_path();
    let path = dir.join("bench.ndjson");
    let mut f = std::fs::File::create(&path).unwrap();

    for row in 0..n {
        write!(f, "{{");
        for c in 0..ncols {
            if c > 0 {
                write!(f, ",");
            }
            write!(f, "\"col_{}\":{}", c, row as i64 + c as i64).unwrap();
        }
        writeln!(f, "}}").unwrap();
    }

    path
}

// ─── Benchmarks ───────────────────────────────────────────────────

fn bench_csv_swar_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSV: Boundary Detection (SWAR vs Scalar)");

    for size in [10_000, 100_000, 1_000_000] {
        let path = create_csv_file(size, 10);
        let data = std::fs::read(&path).unwrap();
        let data_len = data.len();

        group.throughput(Throughput::Bytes(data_len as u64));

        group.bench_with_input(
            BenchmarkId::new("SWAR", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let bounds = find_record_boundaries_simd(black_box(data), b'"');
                    black_box(bounds);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Scalar", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let bounds =
                        relay_io::csv::find_record_boundaries(black_box(data), b'"');
                    black_box(bounds);
                });
            },
        );

        let _ = std::fs::remove_file(&path);
    }

    group.finish();
}

fn bench_csv_read_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSV: Full Read Throughput");

    for size in [10_000, 100_000, 1_000_000] {
        let path = create_csv_file(size, 10);
        let file_size = std::fs::metadata(&path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("Relay", size),
            &path,
            |b, path| {
                b.iter(|| {
                    let reader = CsvReader::open(path, CsvReadOptions::default()).unwrap();
                    let batch = reader.read_all().unwrap();
                    black_box(batch);
                });
            },
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    group.finish();
}

fn bench_csv_quoted(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSV: Quoted Fields");

    for size in [10_000, 100_000] {
        let path = create_csv_with_quotes(size);
        let file_size = std::fs::metadata(&path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("Relay", size),
            &path,
            |b, path| {
                b.iter(|| {
                    let reader = CsvReader::open(path, CsvReadOptions::default()).unwrap();
                    let batch = reader.read_all().unwrap();
                    black_box(batch);
                });
            },
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    group.finish();
}

fn bench_csv_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSV: Projection (2 of 10 cols)");

    for size in [100_000, 1_000_000] {
        let path = create_csv_file(size, 10);
        let file_size = std::fs::metadata(&path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("Relay", size),
            &path,
            |b, path| {
                b.iter(|| {
                    let reader = CsvReader::open(path, CsvReadOptions::default()).unwrap();
                    let batch = reader.read_columns(&["col_0", "col_5"]).unwrap();
                    black_box(batch);
                });
            },
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    group.finish();
}

fn bench_ndjson_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("NDJSON: Full Read Throughput");

    for size in [10_000, 100_000, 1_000_000] {
        let path = create_ndjson_file(size, 10);
        let file_size = std::fs::metadata(&path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("Relay", size),
            &path,
            |b, path| {
                b.iter(|| {
                    let reader = JsonReader::open(path, JsonReadOptions::default()).unwrap();
                    let batch = reader.read_all().unwrap();
                    black_box(batch);
                });
            },
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    group.finish();
}

fn bench_ndjson_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("NDJSON: Projection (2 of 10 cols)");

    for size in [100_000, 1_000_000] {
        let path = create_ndjson_file(size, 10);
        let file_size = std::fs::metadata(&path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("Relay", size),
            &path,
            |b, path| {
                b.iter(|| {
                    let reader = JsonReader::open(path, JsonReadOptions::default()).unwrap();
                    let batch = reader.read_columns(&["col_0", "col_5"]).unwrap();
                    black_box(batch);
                });
            },
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_csv_swar_vs_scalar,
    bench_csv_read_throughput,
    bench_csv_quoted,
    bench_csv_projection,
    bench_ndjson_read,
    bench_ndjson_projection,
);
criterion_main!(benches);
