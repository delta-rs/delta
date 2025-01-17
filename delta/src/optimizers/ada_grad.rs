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

/// The AdaGrad optimizer struct.
#[derive(Debug)]
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
    g_sum: Option<Tensor>,
    timestep: usize,
    device: Device,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the given learning rate and epsilon.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `epsilon` - A small value to prevent division by zero.
    ///
    /// # Returns
    ///
    /// A new instance of the AdaGrad optimizer.
    pub fn new(learning_rate: f32, epsilon: f32) -> Self {
        Self { learning_rate, epsilon, g_sum: None, timestep: 0, device: Device::default() }
    }

    /// Resets the accumulated gradient sum (g_sum).
    pub fn reset(&mut self) {
        self.g_sum = None;
    }
}

impl Optimizer for AdaGrad {
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

        self.timestep += 1;

        // Initialize gradient sum if not already done
        if self.g_sum.is_none()
            || self.g_sum.as_ref().unwrap().shape().raw_dim() != weights.shape().raw_dim()
        {
            self.g_sum = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
        }

        let g_sum = self.g_sum.as_mut().unwrap();

        // Ensure gradients match the weights' shape
        let processed_gradients = if gradients.shape().raw_dim().as_array_view().to_vec()
            == weights.shape().raw_dim().as_array_view().to_vec()
        {
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

        // Update gradient sum
        *g_sum = g_sum.add(&processed_gradients.pow(2.0));

        // Compute update
        let update = processed_gradients
            .div(&(g_sum.sqrt().add_scalar(self.epsilon)))
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
    use ndarray::IxDyn;
    use crate::optimizers::assert_almost_equal;
    use super::*;

    #[test]
    fn test_adagrad_optimizer() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], IxDyn(&[3]).into());
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], IxDyn(&[3]).into());

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.90000004, 1.9, 2.9];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adagrad_optimizer_incompatible_shapes() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], IxDyn(&[3, 1]).into());
        let gradients = Tensor::new(vec![0.1, 0.2], IxDyn(&[2, 1]).into());

        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_adagrad_optimizer_reset() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], IxDyn(&[3]).into());
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], IxDyn(&[3]).into());

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        optimizer.reset();

        assert!(optimizer.g_sum.is_none(), "g_sum was not reset");
    }
}
