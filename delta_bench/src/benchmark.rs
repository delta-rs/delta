use criterion::{criterion_group, criterion_main, Criterion};
// use deltaml::;

fn _benchmark_function(_c: &mut Criterion) {

}

criterion_group!(benches, _benchmark_function);
criterion_main!(benches);