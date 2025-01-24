use criterion::{Criterion, black_box, criterion_group, criterion_main};
use deltaml::{
    deep_learning::{
        optimizers::{Optimizer, RMSProp},
        tensor_ops::Tensor,
    },
    ndarray::{Dimension, IxDyn, Shape},
};
use rand::Rng;

#[allow(dead_code)]
fn benchmark_rmsprop_optimizer_small(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let dims = IxDyn(&[10, 10]);

    let weights_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let gradients_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let weights = Tensor::new(black_box(weights_data.clone()), Shape::from(dims.clone()));
    let gradients = Tensor::new(black_box(gradients_data.clone()), Shape::from(dims.clone()));

    c.bench_function("rmsprop_optimizer_small", |b| {
        b.iter(|| {
            let mut optimizer = RMSProp::new(black_box(0.01), black_box(0.9), black_box(1e-8))
                .expect("Failed to create optimizer");
            let mut weights_clone = weights.clone();
            let gradients_clone = gradients.clone();
            optimizer.step(&mut weights_clone, &gradients_clone).expect("Failed to perform step");
        })
    });
}

#[allow(dead_code)]
fn benchmark_rmsprop_optimizer_large(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let dims = IxDyn(&[1000, 1000]);

    let weights_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let gradients_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let weights = Tensor::new(black_box(weights_data.clone()), Shape::from(dims.clone()));
    let gradients = Tensor::new(black_box(gradients_data.clone()), Shape::from(dims.clone()));

    let mut group = c.benchmark_group("RMSPropOptimizer");
    group.measurement_time(std::time::Duration::new(10, 0));
    group.sample_size(40);

    group.bench_function("rmsprop_optimizer_large", |b| {
        b.iter(|| {
            let mut optimizer = RMSProp::new(black_box(0.01), black_box(0.9), black_box(1e-8))
                .expect("Failed to create optimizer");
            let mut weights_clone = weights.clone();
            let gradients_clone = gradients.clone();

            for _ in 0..10 {
                optimizer
                    .step(&mut weights_clone, &gradients_clone)
                    .expect("Failed to perform step");
            }
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_rmsprop_optimizer_small, benchmark_rmsprop_optimizer_large);
criterion_main!(benches);
