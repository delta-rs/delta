use crate::common::tensor_ops::Tensor;
use crate::common::layer::Layer;
use crate::common::shape::Shape;

#[derive(Debug)]
pub struct MaxPooling2D {
    pool_size: usize,
    stride: usize,
    input_shape: Option<Shape>,
}

impl MaxPooling2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size, stride, input_shape: None }
    }
}

impl Layer for MaxPooling2D {
    fn build(&mut self, input_shape: Shape) {
        self.input_shape = Some(input_shape);
    }

    fn forward(&mut self, input: &Tensor) -> Tensor {
        todo!("Implement MaxPooling2D forward")
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        todo!("Implement MaxPooling2D backward")
    }

    fn output_shape(&self) -> Shape {
        todo!("Implement MaxPooling2D output_shape")
    }

    fn param_count(&self) -> (usize, usize) {
        (0, 0)
    }

    fn name(&self) -> &str {
        "MaxPooling2D"
    }

    fn update_weights(&mut self, _optimizer: &mut Box<dyn crate::common::optimizer::Optimizer>) {
        todo!("Implement MaxPooling2D update_weights")
    }
}