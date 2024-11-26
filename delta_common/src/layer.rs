use std::fmt::Debug;
use crate::tensor_ops::Tensor;

pub trait Layer: Debug {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad: &Tensor) -> Tensor;
}

#[derive(Debug)]
pub struct LayerOutput {
    pub output: Tensor,
    pub gradients: Tensor,
}