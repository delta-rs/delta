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

use ndarray::{s, IxDyn, Shape};

use crate::activations::Activation;
use crate::common::Tensor;

/// A struct representing the Softmax activation function.
#[derive(Debug)]
pub struct SoftmaxActivation;

impl SoftmaxActivation {
    /// Creates a new instance of `SoftmaxActivation`.
    pub fn new() -> Self {
        Self
    }
}

impl Activation for SoftmaxActivation {
    /// Applies the Softmax activation function to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the Softmax activation function.
    fn activate(&self, input: &Tensor) -> Tensor {
        // Find the maximum value in the input tensor
        let max_value = input.max();

        // Subtract the maximum value from each element in the input tensor
        let stabilized_input = input.map(|x| x - max_value);

        // Compute the exponentials
        let exps = stabilized_input.map(|x| x.exp());

        // Compute the sum of the exponentials
        let sum: f32 = exps.data.iter().sum();

        // Normalize to get the softmax probabilities
        exps.map(|x| x / sum)
    }

    /// Computes the Jacobian of the Softmax function.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the Jacobian matrix of the Softmax function.
    fn derivative(&self, input: &Tensor) -> Tensor {
        // Step 1: Compute the softmax output
        let softmax_output = self.activate(input);

        // Step 2: Get the shape of the input tensor
        let input_shape = softmax_output.data.shape();
        let batch_size = input_shape[0];
        let num_classes = input_shape[1];

        // Step 3: Compute the Jacobian matrix for each batch
        let mut jacobian_data = vec![0.0; batch_size * num_classes * num_classes];

        for b in 0..batch_size {
            for i in 0..num_classes {
                for j in 0..num_classes {
                    let y_i = *softmax_output.data.slice(s![b, i]).into_scalar(); // Extract y_i as a scalar
                    let y_j = *softmax_output.data.slice(s![b, j]).into_scalar(); // Extract y_j as a scalar

                    if i == j {
                        jacobian_data[b * num_classes * num_classes + i * num_classes + j] =
                            y_i * (1.0 - y_i);
                    } else {
                        jacobian_data[b * num_classes * num_classes + i * num_classes + j] =
                            -y_i * y_j;
                    }
                }
            }
        }

        // Step 4: Return the Jacobian tensor
        Tensor::new(
            jacobian_data,
            Shape::from(IxDyn(&[batch_size, num_classes, num_classes])),
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, IxDyn, Shape};

    use super::*;

    #[test]
    fn test_softmax_activation() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        let softmax = SoftmaxActivation::new();
        let output = softmax.activate(&input);

        let expected = Tensor::new(
            vec![0.09003057317038025, 0.24472847105479776, 0.6652409557758217],
            Shape::from(IxDyn(&[1, 3])),
        );

        assert_eq!(output.data, expected.data);
    }

    #[test]
    fn test_softmax_derivative() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        let softmax = SoftmaxActivation::new();
        let derivative = softmax.derivative(&input);

        // Assert that the Jacobian matrix has the correct shape
        assert_eq!(derivative.data.shape(), &[1, 3, 3]);

        // Verify values of the Jacobian matrix against a known correct output
        let expected_data = Array::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![
                0.08192507,
                -0.02204622,
                -0.05987885,
                -0.02204622,
                0.18483645,
                -0.16279022,
                -0.05987885,
                -0.16279022,
                0.22266908,
            ],
        )
        .unwrap();

        let tolerance = 1e-3;
        for (computed, expected) in derivative.data.iter().zip(expected_data.iter()) {
            assert!(
                (computed - expected).abs() < tolerance,
                "Values differ: computed = {}, expected = {}",
                computed,
                expected
            );
        }
    }
}
