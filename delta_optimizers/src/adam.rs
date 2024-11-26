use std::fmt;
use std::fmt::Debug;
use delta_common::Optimizer;
use delta_common::tensor_ops::Tensor;

struct DebuggableScheduler(Box<dyn Fn(usize) -> f32>);

impl Debug for DebuggableScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DebuggableScheduler")
    }
}

#[derive(Debug)]
pub struct Adam {
    learning_rate: f32,
    scheduler: Option<DebuggableScheduler>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate, scheduler: None }
    }

    pub fn set_scheduler<F>(&mut self, scheduler: F)
    where
        F: Fn(usize) -> f32 + 'static,
    {
        self.scheduler = Some(DebuggableScheduler(Box::new(scheduler)));
    }
}

impl Optimizer for Adam {
    fn step(&mut self, gradients: &mut [Tensor]) {
        todo!()
    }
}