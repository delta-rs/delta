use delta_common::{Layer, Shape};
use delta_common::tensor_ops::Tensor;

#[derive(Debug)]
pub struct Dense {
    weights: Tensor,
    bias: Tensor,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Tensor::random(&Shape::from((input_size, output_size))),
            bias: Tensor::zeros(&Shape::new(vec![output_size])),
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights).add(&self.bias)
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        todo!()
    }
}

#[derive(Debug)]
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