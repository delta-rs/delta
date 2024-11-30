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

use crate::common::activation::Activation;
use crate::common::shape::Shape;
use crate::common::tensor_ops::Tensor;

/// A struct representing the Softmax activation function.
#[derive(Debug)]
pub struct SoftmaxActivation;

impl SoftmaxActivation {
    /// Creates a new instance of `SoftmaxActivation`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use deltaml::activations::softmax::SoftmaxActivation;
    ///
    /// let softmax = SoftmaxActivation::new();
    /// ```
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
        let sum = exps.sum();

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
        // Apply the softmax activation to get the probabilities
        let softmax_output = self.activate(input);

        // Get the number of elements in the input tensor
        let n = softmax_output.data.len();

        // Initialize the Jacobian matrix
        let mut jacobian_data = vec![0.0; n * n];

        // Compute the Jacobian matrix
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j; // Map 2D indices to a 1D vector
                if i == j {
                    jacobian_data[idx] = softmax_output.data[i] * (1.0 - softmax_output.data[i]);
                } else {
                    jacobian_data[idx] = -softmax_output.data[i] * softmax_output.data[j];
                }
            }
        }

        // Create the Jacobian tensor
        Tensor {
            data: jacobian_data,
            shape: Shape::new(vec![n, n]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_activation() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let softmax = SoftmaxActivation::new();
        let output = softmax.activate(&input);

        assert_eq!(
            output.data,
            vec![0.09003057317038025, 0.24472847105479776, 0.6652409557758217]
        );
        assert_eq!(output.shape.0, vec![1, 3]);
    }

    #[test]
    fn test_softmax_derivative() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let softmax = SoftmaxActivation::new();
        let derivative = softmax.derivative(&input);

        assert_eq!(
            derivative.data,
            vec![
                -0.09003057317038025,
                -0.24472847105479776,
                -0.6652409557758217
            ]
        );
        assert_eq!(derivative.shape.0, vec![1, 3]);
    }
}
