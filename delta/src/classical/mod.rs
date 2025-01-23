pub mod classification;
pub mod clustering;
pub mod dimensionality_reduction;
pub mod regression;

pub use classification::LinearRegression;

use ndarray::{Array1, Array2};

// Define a trait for all Classical models
pub trait Classical {
    /// Create a new instance of the model
    fn new() -> Self
    where
        Self: Sized;

    /// Fit the model to the data
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize);

    /// Use the model to make predictions
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
}

/// Calculate Mean Squared Error loss
pub fn calculate_loss(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    let m = predictions.len() as f64;
    let diff = predictions - actuals;
    (diff.mapv(|x| x.powi(2)).sum()) / m
}

/// Perform gradient descent and return gradients for weights and bias
fn gradient_descent(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    bias: f64,
) -> (Array1<f64>, f64) {
    let predictions = x.dot(weights) + bias;
    let m = x.shape()[0] as f64;

    let grad_weights = x.t().dot(&(predictions.clone() - y)) / m;
    let grad_bias = (predictions - y).sum() / m;

    (grad_weights, grad_bias)
}
