use crate::tensor_ops::Tensor;

pub trait DatasetOps {
    fn load_train() -> Self;
    fn load_test() -> Self;
    fn normalize(&mut self, min: f32, max: f32);
    fn add_noise(&mut self, noise_level: f32);
}

pub struct Dataset {
    pub inputs: Tensor,
    pub labels: Tensor,
}

impl Dataset {
    pub fn new(inputs: Tensor, labels: Tensor) -> Self {
        Dataset { inputs, labels }
    }
}