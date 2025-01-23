pub mod classification;
pub mod clustering;
pub mod dimensionality_reduction;
pub mod regression;

use ndarray::{Array1, Array2};

/// Calculate Mean Squared Error loss
pub fn calculate_loss(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    let m = predictions.len() as f64;
    let diff = predictions - actuals;
    (diff.mapv(|x| x.powi(2)).sum()) / m
}

/// Perform gradient descent and return gradients for weights and bias
pub fn gradient_descent(
    x: &Array2<f64>,
    y: &Array1<f64>,
    predictions: &Array1<f64>,
    weights: &Array1<f64>,
) -> (Array1<f64>, f64) {
    let m = x.shape()[0] as f64;

    let grad_weights = x.t().dot(&(predictions - y)) / m;
    let grad_bias = (predictions - y).sum() / m;

    (grad_weights, grad_bias)
}
