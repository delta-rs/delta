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

use std::fmt;
use std::fmt::Debug;

use ndarray::{Dimension, IxDyn, Shape};

use crate::common::Tensor;
use crate::devices::Device;
use crate::optimizers::Optimizer;
use crate::optimizers::error::OptimizerError;

/// A wrapper struct for a debuggable scheduler function.
#[allow(dead_code)]
struct DebuggableScheduler(Box<dyn Fn(usize) -> f32>);

impl Debug for DebuggableScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DebuggableScheduler")
    }
}

/// The Adam optimizer struct.
#[derive(Debug)]
pub struct Adam {
    #[allow(dead_code)]
    learning_rate: f32,
    scheduler: Option<DebuggableScheduler>,
    m: Option<Tensor>,
    v: Option<Tensor>,
    timestep: usize,
    device: Device,
}

impl Adam {
    /// Creates a new Adam optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the Adam optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            scheduler: None,
            m: None,
            v: None,
            timestep: 0,
            device: Device::default(),
        }
    }

    /// Sets the scheduler function for the Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `scheduler` - A function that takes an epoch number and returns a learning rate.
    pub fn set_scheduler<F>(&mut self, scheduler: F)
    where
        F: Fn(usize) -> f32 + 'static,
    {
        self.scheduler = Some(DebuggableScheduler(Box::new(scheduler)));
    }

    /// Initializes the moving averages for the Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the weights tensor.
    fn initialize_moving_averages(&mut self, shape: &Vec<usize>) {
        if self.m.is_none()
            || self.m.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec() != *shape
        {
            self.m = Some(Tensor::zeros(Shape::from(IxDyn(&shape)), self.device.clone()));
        }
        if self.v.is_none()
            || self.v.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec() != *shape
        {
            self.v = Some(Tensor::zeros(Shape::from(IxDyn(&shape)), self.device.clone()));
        }
    }
}

impl Optimizer for Adam {
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

        let weights_shape = weights.shape().raw_dim().as_array_view().to_vec();

        self.initialize_moving_averages(&weights_shape);

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

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

        // Update moving averages
        *m = m.mul_scalar(0.9).add(&processed_gradients.mul_scalar(0.1));
        *v = v.mul_scalar(0.999).add(&processed_gradients.pow(2.0).mul_scalar(0.001));

        // Bias correction
        let bias_correction_1 = 1.0 - 0.9f32.powi(self.timestep as i32);
        let bias_correction_2 = 1.0 - 0.999f32.powi(self.timestep as i32);

        let m_hat = m.div_scalar(bias_correction_1);
        let v_hat = v.div_scalar(bias_correction_2);

        // Get learning rate
        let lr = self.scheduler.as_ref().map_or(self.learning_rate, |s| s.0(self.timestep));

        // Compute scaling factor based on max gradient magnitude
        let max_gradient = processed_gradients.data.iter().map(|g| g.abs()).fold(0.0, f32::max);

        let scaling_factor = if max_gradient > 10.0 { 10.0 / max_gradient } else { 1.0 };

        // Apply scaled learning rate
        let epsilon = 1e-8;
        let scaled_lr = lr * scaling_factor;
        let update = m_hat.div(&v_hat.sqrt().add_scalar(epsilon)).mul_scalar(scaled_lr);

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

    fn assert_almost_equal(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
        let actual_slice = actual.as_slice().expect("Failed to convert ArrayD to slice");
        for (a, e) in actual_slice.iter().zip(expected.iter()) {
            assert!((a - e).abs() < tolerance, "Expected: {:?}, Actual: {:?}", e, a);
        }
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.999, 2.999];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_no_scheduler() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![-0.0009999934, -0.0009999934, -0.0009999934];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_with_scheduler() {
        let mut optimizer = Adam::new(0.001);
        optimizer.set_scheduler(|_epoch| 0.05); // Set a fixed learning rate for simplicity
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.95000035, 1.9500003, 2.9500003];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_gradients_broadcasting() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1], Shape::from(IxDyn(&[1, 1]))); // Broadcastable gradient
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.999, 2.999];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_incompatible_shapes() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1]))); // Mismatched shape
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
    fn test_adam_optimizer_step_multiple_times() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.99700004, 0.99700004, 0.99700004];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_bias_correction() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        // Step without bias correction
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Correct timestep
        assert_eq!(optimizer.timestep, 1);

        // Bias correction factors should influence the next step
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        assert_eq!(optimizer.timestep, 2);
    }

    #[test]
    fn test_adam_optimizer_zero_gradients() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_gradient_scaling() {
        let mut optimizer = Adam::new(0.001); // Initial learning rate
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![20.0, 20.0, 20.0], Shape::from(IxDyn(&[3, 1]))); // High gradient values

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Compute expected weights
        let scaling_factor: f32 = 10.0 / 20.0; // Scale learning rate
        let adjusted_lr = 0.001 * scaling_factor;
        let epsilon = 1e-8;
        let m = 0.0 + (1.0 - 0.9) * 20.0; // m after one step
        let v = 0.0 + (1.0 - 0.999) * (20.0 * 20.0); // v after one step
        let m_hat = m / (1.0 - 0.9); // Bias-corrected m
        let v_hat: f32 = v / (1.0 - 0.999); // Bias-corrected v
        let update = m_hat / (v_hat.sqrt() + epsilon) * adjusted_lr;

        let expected = vec![1.0 - update, 1.0 - update, 1.0 - update];

        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_small_gradients() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0_f32, 2.0_f32, 3.0_f32], Shape::from(IxDyn(&[3, 1])));
        let gradients =
            Tensor::new(vec![1e-7_f32, 1e-7_f32, 1e-7_f32], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Compute expected weights manually
        let learning_rate: f32 = 0.001;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let epsilon: f32 = 1e-8;

        let mut m: f32 = 0.0; // First moment estimate
        let mut v: f32 = 0.0; // Second moment estimate

        let g: f32 = 1e-7; // Gradient value
        m = beta1 * m + (1.0 - beta1) * g; // Update first moment
        v = beta2 * v + (1.0 - beta2) * (g * g); // Update second moment

        let m_hat: f32 = m / (1.0 - beta1.powi(1)); // Bias-corrected first moment
        let v_hat: f32 = v / (1.0 - beta2.powi(1)); // Bias-corrected second moment

        let update: f32 = m_hat / (v_hat.sqrt() + epsilon) * learning_rate;

        let expected = vec![1.0 - update, 2.0 - update, 3.0 - update];

        assert_almost_equal(&weights.data, &expected, 1e-6);
    }
}
