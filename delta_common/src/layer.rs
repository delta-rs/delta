use crate::tensor_ops::Tensor;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad: &Tensor) -> Tensor;
}

pub struct LayerOutput {
    pub output: Tensor,
    pub gradients: Tensor,
}