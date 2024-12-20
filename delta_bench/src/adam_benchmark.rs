use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deltaml::common::{IxDyn, Shape, Tensor};
use deltaml::optimizers::{Adam, Optimizer};

fn benchmark_adam_optimizer(c: &mut Criterion) {
    c.bench_function("adam_optimizer", |b| {
        b.iter(|| {
            let mut optimizer = Adam::new(black_box(0.001));
            let mut weights = Tensor::new(black_box(vec![1.0, 2.0, 3.0]), Shape::from(IxDyn(&[3, 1])));
            let gradients = Tensor::new(black_box(vec![0.1, 0.2, 0.3]), Shape::from(IxDyn(&[3, 1])));
            optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        })
    });
}

criterion_group!(benches, benchmark_adam_optimizer);
criterion_main!(benches);