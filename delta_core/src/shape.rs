#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Shape(dimensions)
    }

    pub fn len(&self) -> usize {
        self.0.iter().product()
    }
}