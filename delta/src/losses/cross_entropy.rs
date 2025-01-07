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
use crate::losses::Loss;

/// A struct representing the Cross-Entropy Loss function.
#[derive(Debug)]
pub struct CrossEntropyLoss;

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropyLoss {
    /// Creates a new CrossEntropyLoss instance.
    pub fn new() -> Self {
        Self
    }

    /// Clips the tensor to avoid issues with log(0) or division by zero.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clip.
    /// * `epsilon` - The epsilon value to use for clipping.
    ///
    /// # Returns
    ///
    /// The clipped tensor.
    fn clip_tensor(&self, tensor: &Tensor, epsilon: f32) -> Tensor {
        tensor.map(|x| x.clamp(epsilon, 1.0 - epsilon))
    }
}

impl Loss for CrossEntropyLoss {
    /// Calculates the cross-entropy loss between the true and predicted values.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values.
    /// * `y_pred` - The predicted values.
    ///
    /// # Returns
    ///
    /// The cross-entropy loss.
    fn calculate_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        if y_true.shape().raw_dim() != y_pred.shape().raw_dim() {
            panic!(
                "Shape mismatch: y_true.shape = {:?}, y_pred.shape = {:?}",
                y_true.shape().raw_dim(),
                y_pred.shape().raw_dim()
            );
        }

        // Clip predictions to avoid log(0)
        let epsilon = 1e-12;
        let clipped_pred = self.clip_tensor(y_pred, epsilon);

        // Compute element-wise multiplication
        let cross_entropy =
            y_true.data.iter().zip(clipped_pred.data.iter()).map(|(t, p)| t * p.ln()).sum::<f32>();

        // Return cross-entropy loss
        -cross_entropy
    }

    /// Calculates the gradient of the cross-entropy loss with respect to the predicted values.
    ///
    /// # Arguments
    ///
    /// * `output` - The predicted values.
    /// * `target` - The true values.
    ///
    /// # Returns
    ///
    /// The gradient of the cross-entropy loss with respect to the predicted values.
    fn calculate_loss_grad(&self, output: &Tensor, target: &Tensor) -> Tensor {
        // Ensure shapes match
        if output.shape().raw_dim() != target.shape().raw_dim() {
            panic!(
                "Shape mismatch: output.shape = {:?}, target.shape = {:?}",
                output.shape(),
                target.shape()
            );
        }

        // Clip predictions to avoid division by zero
        let epsilon = 1e-12;
        let clipped_pred = self.clip_tensor(output, epsilon);

        // Compute gradient: (predictions - targets) / batch_size
        let grad_data: Vec<f32> = clipped_pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| (p - t) / target.data.len() as f32)
            .collect();

        // Create Tensor from grad_data
        let grad_shape = clipped_pred.shape().raw_dim().as_array_view().to_vec(); // Ensure correct shape
        Tensor {
            data: ndarray::Array::from_shape_vec(ndarray::IxDyn(&grad_shape), grad_data)
                .expect("Failed to create gradient tensor"),
            device: Device::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use ndarray::{IxDyn, Shape};

    use super::*;
    use crate::common::Tensor;

    #[test]
    fn test_cross_entropy_loss() {
        let y_true = Tensor::new(vec![1.0, 0.0, 0.0], Shape::from(IxDyn(&[1, 3])));
        let y_pred = Tensor::new(vec![0.7, 0.2, 0.1], Shape::from(IxDyn(&[1, 3])));

        let loss = CrossEntropyLoss::new();
        let calculated_loss = loss.calculate_loss(&y_true, &y_pred);

        let expected_loss = -(1.0 * (0.7f32.ln())) / 1.0;

        assert!(
            (calculated_loss - expected_loss).abs() < 1e-6,
            "Expected loss to be {:.6}, got {:.6}",
            expected_loss,
            calculated_loss
        );
    }

    #[test]
    fn test_cross_entropy_loss_grad() {
        let y_true = Tensor::new(vec![1.0, 0.0, 0.0], Shape::from(IxDyn(&[1, 3])));
        let y_pred = Tensor::new(vec![0.7, 0.2, 0.1], Shape::from(IxDyn(&[1, 3])));

        let loss = CrossEntropyLoss::new();
        let grad = loss.calculate_loss_grad(&y_pred, &y_true);

        let expected_grad = vec![-0.1, 0.06666667, 0.03333334]; // Adjusted to correct gradient
        assert!(
            grad.data.iter().zip(expected_grad.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Expected gradient: {:?}, got: {:?}",
            expected_grad,
            grad.data
        );
    }
}
