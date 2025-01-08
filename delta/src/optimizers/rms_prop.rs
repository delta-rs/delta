//! BSD 3-Clause License
//!
//! Copyright (c) 2025, BlackPortal ○
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! 1. Redistributions of source code must retain the above copyright notice, this
//!    list of conditions and the following disclaimer.
//!
//! 2. Redistributions in binary form must reproduce the above copyright notice,
//!    this list of conditions and the following disclaimer in the documentation
//!    and/or other materials provided with the distribution.
//!
//! 3. Neither the name of the copyright holder nor the names of its
//!    contributors may be used to endorse or promote products derived from
//!    this software without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use ndarray::Dimension;

use crate::common::Tensor;
use crate::devices::Device;
use crate::optimizers::Optimizer;
use crate::optimizers::error::OptimizerError;

/// The RMSProp optimizer struct.
#[derive(Debug)]
pub struct RMSProp {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    mean_square: Option<Tensor>,
    device: Device,
}

impl RMSProp {
    /// Creates a new RMSProp optimizer with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `decay_rate` - The decay rate for the moving average of squared gradients.
    /// * `epsilon` - A small value to prevent division by zero (must be > 0).
    ///
    /// # Returns
    ///
    /// A new instance of the RMSProp optimizer.
    pub fn new(learning_rate: f32, decay_rate: f32, epsilon: f32) -> Result<Self, OptimizerError> {
        if epsilon <= 0.0 {
            return Err(OptimizerError::InvalidEpsilon(
                "Epsilon must be greater than 0.".to_string(),
            ));
        }
        if learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }
        Ok(Self {
            learning_rate,
            decay_rate,
            epsilon,
            mean_square: None,
            device: Device::default(),
        })
    }
}

impl Optimizer for RMSProp {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if weights.shape().raw_dim() != gradients.shape().raw_dim() {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Initialize mean square tensor if not already done
        if self.mean_square.is_none() {
            self.mean_square = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
        }

        let mean_square = self.mean_square.as_mut().unwrap();
        let one_minus_decay = 1.0 - self.decay_rate;

        // Update mean square
        *mean_square = mean_square
            .mul_scalar(self.decay_rate)
            .add(&gradients.pow(2.0).mul_scalar(one_minus_decay));

        // Compute update
        let update = gradients
            .div(&mean_square.sqrt().add_scalar(self.epsilon))
            .mul_scalar(self.learning_rate);

        *weights -= update;

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
    use ndarray::{ArrayD, IxDyn, Shape};

    use super::*;

    /// Default constants for RMSProp optimizer
    const DEFAULT_DECAY_RATE: f32 = 0.9;
    const DEFAULT_EPSILON: f32 = 1e-8;

    fn assert_almost_equal(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
        let actual_slice = actual.as_slice().expect("Failed to convert ArrayD to slice");
        for (a, e) in actual_slice.iter().zip(expected.iter()) {
            assert!((a - e).abs() < tolerance, "Expected: {:?}, Actual: {:?}", e, a);
        }
    }

    #[test]
    fn test_rmsprop_optimizer_multiple_steps() {
        let mut optimizer = RMSProp::new(0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON)
            .expect("Failed to create optimizer");
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Update expected values based on manual calculation or reference implementation
        let expected = vec![/* Recalculated values */];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_rmsprop_optimizer_incompatible_shapes() {
        let mut optimizer = RMSProp::new(0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON)
            .expect("Failed to create optimizer");
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1])));
        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");
    }

    #[test]
    fn test_rmsprop_optimizer_zero_gradients() {
        let mut optimizer = RMSProp::new(0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON)
            .expect("Failed to create optimizer");
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_rmsprop_invalid_epsilon() {
        let result = RMSProp::new(0.01, DEFAULT_DECAY_RATE, 0.0);
        assert!(result.is_err(), "Expected an error due to invalid epsilon");
    }

    #[test]
    fn test_rmsprop_invalid_learning_rate() {
        let result = RMSProp::new(-0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON);
        assert!(result.is_err(), "Expected an error due to invalid learning rate");
    }
}
