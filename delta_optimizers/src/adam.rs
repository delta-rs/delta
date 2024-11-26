pub struct Adam {
    learning_rate: f32,
    scheduler: Option<Box<dyn Fn(usize) -> f32>>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate, scheduler: None }
    }

    pub fn set_scheduler<F>(&mut self, scheduler: F)
    where
        F: Fn(usize) -> f32 + 'static,
    {
        self.scheduler = Some(Box::new(scheduler));
    }
}