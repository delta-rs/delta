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

impl From<(usize, usize)> for Shape {
    fn from(dimensions: (usize, usize)) -> Self {
        Shape(vec![dimensions.0, dimensions.1])
    }
}