use criterion::{black_box, criterion_group, criterion_main, Criterion};
use delta::your_function;

fn benchmark_function(c: &mut Criterion) {
    c.bench_function("your_function", |b| b.iter(|| your_function(black_box(42))));
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);