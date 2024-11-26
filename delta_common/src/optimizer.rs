use crate::tensor_ops::Tensor;

pub trait Optimizer {
    fn step(&mut self, gradients: &mut [Tensor]);
}

pub struct OptimizerConfig {
    pub learning_rate: f32,
}