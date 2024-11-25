use crate::shape::Shape;

pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Shape,
}

impl Tensor {
    pub fn zeros(shape: &Shape) -> Self {
        todo!("Create a tensor filled with zeros")
    }

    pub fn random(shape: &Shape) -> Self {
        todo!("Create a tensor filled with random values")
    }
}