use ndarray::Dimension;

use crate::common::Tensor;
use crate::devices::Device;
use crate::optimizers::error::OptimizerError;
use crate::optimizers::Optimizer;

/// The SGD with Momentum optimizer struct.
#[derive(Debug)]
pub struct SGDWithMomentum {
    #[allow(dead_code)]
    learning_rate: f32,
    #[allow(dead_code)]
    momentum: f32,
    velocity: Option<Tensor>,
    device: Device,
}

impl SGDWithMomentum {
    /// Creates a new SGDWithMomentum optimizer with the given learning rate and momentum.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `momentum` - The momentum factor.
    ///
    /// # Returns
    ///
    /// A new instance of the SGDWithMomentum optimizer.
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: None,
            device: Device::default(),
        }
    }
}

impl Optimizer for SGDWithMomentum {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        // Initialize velocity if not already done
        if self.velocity.is_none()
            || self.velocity.as_ref().unwrap().shape().raw_dim() != weights.shape().raw_dim()
        {
            self.velocity = Some(Tensor::zeros(weights.shape().clone()));
            self.velocity = Some(
                self.velocity
                    .as_mut()
                    .unwrap()
                    .to_device(self.device.clone())
                    .unwrap(),
            );
        }

        let velocity = self.velocity.as_mut().unwrap();

        // Ensure gradients match the weights' shape
        let processed_gradients = if gradients.shape().raw_dim() == weights.shape().raw_dim() {
            gradients.clone()
        } else if gradients.shape().raw_dim().ndim() <= weights.shape().raw_dim().ndim()
            && gradients
                .shape()
                .raw_dim()
                .as_array_view()
                .iter()
                .rev()
                .zip(weights.shape().raw_dim().as_array_view().iter().rev())
                .all(|(g, w)| *g == *w || *g == 1)
        {
            gradients.broadcast(weights.shape())
        } else {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        };

        // Update velocity: v = momentum * v - learning_rate * gradients
        *velocity = velocity
            .mul_scalar(self.momentum)
            .sub(&processed_gradients.mul_scalar(self.learning_rate));

        // Update weights: w = w + v
        *weights += velocity.clone();

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn, Shape};

    fn assert_almost_equal(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
        let actual_slice = actual
            .as_slice()
            .expect("Failed to convert ArrayD to slice");
        for (a, e) in actual_slice.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < tolerance,
                "Expected: {:?}, Actual: {:?}",
                e,
                a
            );
        }
    }

    #[test]
    fn test_sgd_with_momentum_optimizer() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");
        let expected = vec![0.999, 1.998, 2.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_multiple_steps() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");
        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");

        let expected = vec![0.9971, 1.9971, 2.9971];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_zero_gradients() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));

        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_incompatible_shapes() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1])));
        let result = optimizer.step(&mut weights, &gradients);

        assert!(
            result.is_err(),
            "Expected an error due to incompatible shapes"
        );

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }
}
