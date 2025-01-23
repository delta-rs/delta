pub mod classification;
pub mod clustering;
pub mod dimensionality_reduction;
pub mod regression;

pub use classification::LinearRegression;

use ndarray::{Array1, Array2};

/// Defines a common interface for classical machine learning models.
///
/// This trait outlines the basic methods that all classical ML models should implement,
/// providing a uniform way to instantiate, train, and use models for predictions.
pub trait Classical {
    /// Creates and returns a new instance of the model.
    ///
    /// This method should initialize the model with default parameters or learnable parameters
    /// set to initial values. The `Sized` constraint ensures that `Self` has a known size at compile time.
    fn new() -> Self
    where
        Self: Sized;

    /// Trains the model using the provided data.
    ///
    /// This method adjusts the model's parameters based on the input data to minimize the error
    /// between predictions and actual values. The training process involves multiple iterations
    /// or epochs, with each iteration potentially updating the model's parameters.
    ///
    /// # Arguments
    ///
    /// * `x` - An `Array2<f64>` representing the input features where each row is a sample
    ///   and each column is a feature.
    /// * `y` - An `Array1<f64>` representing the target values or labels for each sample.
    /// * `learning_rate` - A `f64` specifying how much to adjust the model's parameters with
    ///   each iteration.
    /// * `epochs` - A `usize` indicating the number of training iterations.
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize);

    /// Makes predictions using the trained model.
    ///
    /// This method applies the model's learned parameters to new or test data to generate predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - An `Array2<f64>` representing the input features for which predictions are to be made.
    ///
    /// # Returns
    ///
    /// Returns an `Array1<f64>` where each element is the model's prediction for the corresponding
    /// input sample.
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
}

/// Calculates the Mean Squared Error (MSE) loss between predictions and actual values.
///
/// This function computes the average of the squared differences between predicted
/// and actual values, which is a common measure of model performance in regression tasks.
///
/// # Arguments
///
/// * `predictions` - An `Array1<f64>` containing the predicted values from the model.
/// * `actuals` - An `Array1<f64>` containing the true or actual values.
///
/// # Returns
///
/// Returns a `f64` representing the Mean Squared Error loss.
pub fn calculate_loss(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    let m = predictions.len() as f64;
    let diff = predictions - actuals;
    (diff.mapv(|x| x.powi(2)).sum()) / m
}

/// Performs gradient descent to compute the gradients for weights and bias.
///
/// This function calculates the gradients for updating the model parameters in a linear regression
/// context. It computes the predictions based on the current weights and bias, then calculates
/// the gradients of the loss function with respect to these parameters.
///
/// # Arguments
///
/// * `x` - An `Array2<f64>` representing the input features where each row is a sample and each column
///   is a feature.
/// * `y` - An `Array1<f64>` representing the true labels or target values for each sample.
/// * `weights` - An `Array1<f64>` representing the current weights of the model.
/// * `bias` - A `f64` representing the current bias of the model.
///
/// # Returns
///
/// Returns a tuple where:
///   - The first element is an `Array1<f64>` representing the gradient for the weights.
///   - The second element is a `f64` representing the gradient for the bias.
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
