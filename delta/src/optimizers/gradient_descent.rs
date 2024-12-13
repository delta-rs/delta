//! BSD 3-Clause License
//!
//! Copyright (c) 2024, The Delta Project Δ
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

use crate::common::Tensor;
use crate::devices::Device;
use crate::optimizers::error::OptimizerError;
use crate::optimizers::Optimizer;
use ndarray::Dimension;

/// The Gradient Descent optimizer struct.
#[derive(Debug)]
pub struct GradientDescent {
    learning_rate: f32,
    device: Device,
}

impl GradientDescent {
    /// Creates a new Gradient Descent optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the Gradient Descent optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            device: Device::default(),
        }
    }
}

impl Optimizer for GradientDescent {
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

        // Ensure gradients match the weights' shape
        if gradients.shape().raw_dim().as_array_view().to_vec()
            != weights.shape().raw_dim().as_array_view().to_vec()
        {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Update weights
        let update = gradients.mul_scalar(self.learning_rate);
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
    fn test_gradient_descent_optimizer() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");
        let expected = vec![0.999, 1.998, 2.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_gradient_descent_optimizer_incompatible_shapes() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1]))); // Mismatched shape
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

    #[test]
    fn test_gradient_descent_optimizer_zero_gradients() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_gradient_descent_optimizer_multiple_steps() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");
        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");
        optimizer
            .step(&mut weights, &gradients)
            .expect("Failed to perform step");

        let expected = vec![0.997, 0.997, 0.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }
}
