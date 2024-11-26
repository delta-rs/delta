use delta_core::Layer;
use delta_core::tensor_ops::Tensor;

pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Relu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        todo!()
    }
}