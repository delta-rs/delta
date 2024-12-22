use crate::common::tensor_ops::Tensor;
use crate::devices::Device;
use crate::losses::Loss;

#[derive(Debug)]
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    /// Creates a new HuberLoss instance with the specified delta value.
    ///
    /// # Arguments
    ///
    /// * `delta` - The delta value for the Huber loss.
    pub fn new(delta: f32) -> Self {
        if delta <= 0.0 {
            panic!("Delta must be positive.");
        }
        Self { delta }
    }
}

impl Loss for HuberLoss {
    /// Calculates the Huber loss between two tensors.
    ///
    /// # Arguments
    /// * `y_true` - The ground truth tensor.
    /// * `y_pred` - The predicted tensor.
    ///
    /// # Returns
    ///
    /// The Huber loss between the two tensors.
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
        let diff = (&y_true.data - &y_pred.data).mapv(|x| x.abs());

        // Step 4: Compute the Huber loss per element
        let huber_loss = diff.mapv(|x| {
            if x <= self.delta { 0.5 * x.powi(2) } else { self.delta * (x - 0.5 * self.delta) }
        });

        // Step 5: Calculate the mean of the Huber loss values
        if huber_loss.is_empty() {
            panic!("Cannot calculate loss: no dataset in input tensors");
        }

        huber_loss.mean().expect("Mean computation failed unexpectedly")
    }

    /// Calculates the gradient of the Huber loss with respect to the output tensor.
    ///
    /// # Arguments
    /// * `output` - The output tensor from the model.
    /// * `target` - The target tensor.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the gradient of the Huber loss with respect to the output tensor.
    fn calculate_loss_grad(&self, output: &Tensor, target: &Tensor) -> Tensor {
        // Ensure shapes match
        if output.data.shape() != target.data.shape() {
            panic!(
                "Shape mismatch: output.shape = {:?}, target.shape = {:?}",
                output.data.shape(),
                target.data.shape()
            );
        }

        // Calculate the difference
        let diff = &output.data - &target.data;

        // Compute the gradient
        let gradient =
            diff.mapv(|x| if x.abs() <= self.delta { x } else { self.delta * x.signum() });

        // Normalize the gradient by the number of elements
        let total_elements = output.data.len() as f32;
        let normalized_gradient = &gradient / total_elements;

        Tensor { data: normalized_gradient, device: Device::default() }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;
    use crate::common::Tensor;

    #[test]
    fn test_huber_loss() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(IxDyn(&[2, 3])));
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(IxDyn(&[2, 3])));
        let loss = HuberLoss::new(1.0);
        let result = loss.calculate_loss(&y_true, &y_pred);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_huber_loss_with_actual_values() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let y_pred = Tensor::new(vec![2.5, 2.0, 3.0, 3.0], Shape::from(IxDyn(&[2, 2])));
        let loss = HuberLoss::new(1.0);
        let result = loss.calculate_loss(&y_true, &y_pred);

        let expected_loss = 0.375; // Updated to match actual calculation

        println!("Expected Loss: {}", expected_loss);
        println!("Calculated Loss: {}", result);

        assert!((result - expected_loss).abs() < 1e-6, "Calculated loss did not match expected");
    }

    #[test]
    fn test_huber_loss_with_nan() {
        let y_true = Tensor::new(vec![1.0, 2.0, f32::NAN, 4.0], Shape::from(IxDyn(&[2, 2])));
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let loss = HuberLoss::new(1.0);

        let result = std::panic::catch_unwind(|| {
            loss.calculate_loss(&y_true, &y_pred);
        });

        assert!(result.is_err(), "Expected a panic due to NaN in inputs, but no panic occurred.");
    }

    #[test]
    fn test_huber_loss_with_mismatch() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(IxDyn(&[2, 3])));
        let loss = HuberLoss::new(1.0);

        let result = std::panic::catch_unwind(|| {
            loss.calculate_loss(&y_true, &y_pred);
        });

        assert!(result.is_err(), "Expected a panic due to shape mismatch, but no panic occurred.");
    }
}
