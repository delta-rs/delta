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

/// The AdaDelta optimizer struct.
#[derive(Debug)]
pub struct AdaDelta {
    rho: f32,
    epsilon: f32,
    accumulated_gradients: Option<Tensor>,
    accumulated_updates: Option<Tensor>,
    device: Device,
}

impl AdaDelta {
    /// Creates a new AdaDelta optimizer with the given hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `rho` - Decay rate for the moving average of gradients.
    /// * `epsilon` - Small value to avoid division by zero.
    pub fn new(rho: f32, epsilon: f32) -> Self {
        Self {
            rho,
            epsilon,
            accumulated_gradients: None,
            accumulated_updates: None,
            device: Device::default(),
        }
    }
}

impl Optimizer for AdaDelta {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if gradients.shape().size() != weights.shape().size() {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Initialize accumulated gradients and updates if not already done
        if self.accumulated_gradients.is_none()
            || self.accumulated_gradients.as_ref().unwrap().shape().raw_dim()
                != weights.shape().raw_dim()
        {
            self.accumulated_gradients =
                Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
            self.accumulated_gradients = Some(
                self.accumulated_gradients
                    .as_mut()
                    .unwrap()
                    .to_device(self.device.clone())
                    .unwrap(),
            );
        }
        if self.accumulated_updates.is_none()
            || self.accumulated_updates.as_ref().unwrap().shape().raw_dim()
                != weights.shape().raw_dim()
        {
            self.accumulated_updates =
                Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
            self.accumulated_updates = Some(
                self.accumulated_updates.as_mut().unwrap().to_device(self.device.clone()).unwrap(),
            );
        }

        let accumulated_gradients = self.accumulated_gradients.as_mut().unwrap();
        let accumulated_updates = self.accumulated_updates.as_mut().unwrap();

        // Update accumulated gradients
        *accumulated_gradients = accumulated_gradients
            .mul_scalar(self.rho)
            .add(&gradients.pow(2.0).mul_scalar(1.0 - self.rho));

        // Compute the update value
        let rms_updates = accumulated_updates.sqrt().add_scalar(self.epsilon);
        let rms_gradients = accumulated_gradients.sqrt().add_scalar(self.epsilon);
        let update = Tensor {
            data: gradients.div(&rms_gradients).data * rms_updates.data,
            device: self.device.clone(),
        };

        // Update accumulated updates
        *accumulated_updates = accumulated_updates
            .mul_scalar(self.rho)
            .add(&update.pow(2.0).mul_scalar(1.0 - self.rho));

        // Apply the update to weights
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
    use ndarray::{IxDyn, Shape};
    use crate::optimizers::assert_almost_equal;
    use super::*;

    #[test]
    fn test_adadelta_optimizer() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.99999684, 1.999_996_8, 2.999_997];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }

    #[test]
    fn test_adadelta_optimizer_multiple_steps() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        for _ in 0..5 {
            optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        }

        let expected = vec![0.99997528, 1.999_975_3, 2.999_975_2];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }

    #[test]
    fn test_adadelta_optimizer_zero_gradients() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adadelta_optimizer_incompatible_shapes() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2])));

        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");
        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2]);
            assert_eq!(w_shape, vec![3]);
        } else {
            panic!("Unexpected error type");
        }
    }
}
