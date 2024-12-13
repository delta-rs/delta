use crate::common::tensor_ops::Tensor;
use crate::devices::Device;
use crate::losses::Loss;

#[derive(Debug)]
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MeanAbsoluteError {
    /// Calculates the mean absolute error (MAE) between two tensors.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The ground truth tensor.
    /// * `y_pred` - The predicted tensor.
    ///
    /// # Returns
    ///
    /// The mean absolute error between the two tensors.
    fn calculate_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Step 1: Ensure the shapes of y_true and y_pred match
        if y_true.data.shape() != y_pred.data.shape() {
            panic!(
                "Shape mismatch: y_true.shape = {:?}, y_pred.shape = {:?}",
                y_true.data.shape(),
                y_pred.data.shape()
            );
        }

        // Step 2: Check for NaN values in y_true and y_pred
        if y_true.data.iter().any(|&x| x.is_nan()) || y_pred.data.iter().any(|&x| x.is_nan()) {
            panic!("NaN value found in inputs");
        }

        // Step 3: Compute the absolute differences
        let abs_diff = (&y_true.data - &y_pred.data).mapv(|x| x.abs());

        // Step 4: Calculate the mean of the absolute differences
        if abs_diff.is_empty() {
            panic!("Cannot calculate loss: no dataset in input tensors");
        }

        let mean_absolute_error = abs_diff.mean().expect("Mean computation failed unexpectedly");

        mean_absolute_error
    }

    /// Calculates the gradient of the loss with respect to the output tensor.
    ///
    /// # Arguments
    ///
    /// * `output` - The output tensor from the model.
    /// * `target` - The target tensor.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the gradient of the loss with respect to the output tensor.
    fn calculate_loss_grad(&self, output: &Tensor, target: &Tensor) -> Tensor {
        // Ensure shapes match
        if output.data.shape() != target.data.shape() {
            panic!(
                "Shape mismatch: output.shape = {:?}, target.shape = {:?}",
                output.data.shape(),
                target.data.shape()
            );
        }

        // Compute the gradient
        let diff = &output.data - &target.data;
        let gradient = diff.mapv(|x| if x > 0.0 { 1.0 } else { -1.0 });

        Tensor { data: gradient, device: Device::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Tensor;
    use ndarray::{IxDyn, Shape};

    #[test]
    fn test_mean_absolute_error() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(IxDyn(&[2, 3])));
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(IxDyn(&[2, 3])));
        let loss = MeanAbsoluteError::new();
        let result = loss.calculate_loss(&y_true, &y_pred);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mean_absolute_error_with_mismatch() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(IxDyn(&[2, 3])));
        let loss = MeanAbsoluteError::new();

        let result = std::panic::catch_unwind(|| {
            loss.calculate_loss(&y_true, &y_pred);
        });

        assert!(result.is_err(), "Expected a panic due to shape mismatch, but no panic occurred.");
    }

    #[test]
    fn test_mean_absolute_error_with_nan() {
        let y_true = Tensor::new(vec![1.0, 2.0, f32::NAN, 4.0], Shape::from(IxDyn(&[2, 2])));
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let loss = MeanAbsoluteError::new();

        let result = std::panic::catch_unwind(|| {
            loss.calculate_loss(&y_true, &y_pred);
        });

        assert!(result.is_err(), "Expected a panic due to NaN in inputs, but no panic occurred.");
    }

    #[test]
    fn test_mean_absolute_error_with_actual_values() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let y_pred = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], Shape::from(IxDyn(&[2, 2])));
        let loss = MeanAbsoluteError::new();
        let result = loss.calculate_loss(&y_true, &y_pred);

        assert!(
            (result - 1.0).abs() < 1e-6,
            "Expected mean absolute error to be 1.0, got {}",
            result
        );
    }
}
