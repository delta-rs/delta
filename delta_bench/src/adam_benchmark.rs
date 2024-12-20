use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deltaml::common::{Dimension, IxDyn, Shape, Tensor};
use deltaml::optimizers::{Adam, Optimizer};
use rand::Rng;

#[allow(dead_code)]
fn benchmark_adam_optimizer_simple(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let dims = IxDyn(&[10, 10]);

    let weights_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let gradients_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let weights = Tensor::new(black_box(weights_data.clone()), Shape::from(dims.clone()));
    let gradients = Tensor::new(black_box(gradients_data.clone()), Shape::from(dims.clone()));

    c.bench_function("adam_optimizer_simple", |b| {
        b.iter(|| {
            let mut optimizer = Adam::new(black_box(0.001));
            let mut weights_clone = weights.clone();
            let gradients_clone = gradients.clone();
            optimizer.step(&mut weights_clone, &gradients_clone).expect("Failed to perform step");
        })
    });
}

// TODO: We have an issue here that it's unable to complete all the 100 samples.
// Need to increase the target time to more than 12s or reduce sample count to 40
#[allow(dead_code)]
fn benchmark_adam_optimizer_large(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let dims = IxDyn(&[1000, 1000]);

    let weights_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let gradients_data: Vec<f32> = (0..dims.size()).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let weights = Tensor::new(black_box(weights_data.clone()), Shape::from(dims.clone()));
    let gradients = Tensor::new(black_box(gradients_data.clone()), Shape::from(dims.clone()));

    c.bench_function("adam_optimizer_large", |b| {
        b.iter(|| {
            let mut optimizer = Adam::new(black_box(0.001));
            let mut weights_clone = weights.clone();
            let gradients_clone = gradients.clone();

            for _ in 0..10 {
                optimizer.step(&mut weights_clone, &gradients_clone).expect("Failed to perform step");
            }
        })
    });
}

criterion_group!(benches, benchmark_adam_optimizer_simple, benchmark_adam_optimizer_large);
criterion_main!(benches);