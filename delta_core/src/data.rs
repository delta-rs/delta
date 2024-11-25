use crate::tensor_ops::Tensor;

pub struct Dataset {
    pub inputs: Tensor,
    pub labels: Tensor,
}

impl Dataset {
    pub fn new(inputs: Tensor, labels: Tensor) -> Self {
        Dataset { inputs, labels }
    }
}