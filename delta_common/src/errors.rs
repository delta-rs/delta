#[derive(Debug)]
pub enum CoreError {
    InvalidShape,
    GradientMismatch,
    Other(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;