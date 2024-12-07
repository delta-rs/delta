//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin, Chase Willden
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

use crate::common::loss::Loss;
use crate::common::tensor_ops::Tensor;

#[derive(Debug)]
pub struct MeanSquaredLoss;

impl MeanSquaredLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MeanSquaredLoss {
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

        // Step 3: Compute the squared differences
        let squared_diff = (&y_true.data - &y_pred.data).mapv(|x| x.powi(2));

        // Step 4: Calculate the mean of the squared differences
        if squared_diff.is_empty() {
            panic!("Cannot calculate loss: no dataset in input tensors");
        }

        let mean_squared_error = squared_diff
            .mean()
            .expect("Mean computation failed unexpectedly");

        mean_squared_error
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

        // Calculate the total number of elements in the tensor
        let total_elements = output.data.len() as f32;

        // Compute the gradient
        let diff = &output.data - &target.data;
        let gradient = &diff * 2.0 / total_elements;

        Tensor { data: gradient }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::tensor_ops::Tensor;

    #[test]
    fn test_mean_squared_loss() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let loss = MeanSquaredLoss::new();
        let result = loss.calculate_loss(&y_true, &y_pred);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mean_squared_loss_with_mismatch() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let loss = MeanSquaredLoss::new();

        let result = std::panic::catch_unwind(|| {
            loss.calculate_loss(&y_true, &y_pred);
        });

        assert!(
            result.is_err(),
            "Expected a panic due to shape mismatch, but no panic occurred."
        );
    }

    #[test]
    fn test_mean_squared_loss_with_nan() {
        let y_true = Tensor::new(vec![1.0, 2.0, f32::NAN, 4.0], vec![2, 2]);
        let y_pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let loss = MeanSquaredLoss::new();

        let result = std::panic::catch_unwind(|| {
            loss.calculate_loss(&y_true, &y_pred);
        });

        assert!(
            result.is_err(),
            "Expected a panic due to NaN in inputs, but no panic occurred."
        );
    }

    #[test]
    fn test_mean_squared_loss_with_actual_values() {
        let y_true = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y_pred = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
        let loss = MeanSquaredLoss::new();
        let result = loss.calculate_loss(&y_true, &y_pred);

        assert!(
            (result - 1.0).abs() < 1e-6,
            "Expected mean squared loss to be 1.0, got {}",
            result
        );
    }
}
