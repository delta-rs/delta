use std::fmt::Debug;
use crate::tensor_ops::Tensor;

pub trait Optimizer: Debug {
    fn step(&mut self, gradients: &mut [Tensor]);
}

#[derive(Debug)]
pub struct OptimizerConfig {
    pub learning_rate: f32,
}