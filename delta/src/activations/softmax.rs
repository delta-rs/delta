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

use ndarray::{IxDyn, Shape, s, Ix};

use crate::activations::Activation;
use crate::common::Tensor;

/// A struct representing the Softmax activation function.
#[derive(Debug)]
pub struct SoftmaxActivation;

impl Default for SoftmaxActivation {
    fn default() -> Self {
        Self::new()
    }
}

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
        let shape = &input.data.shape();
        let batch_size = shape[0];
        let num_classes = shape[1];
        
        // We'll build a new Vec for the output data
        let mut output_data = Vec::with_capacity(batch_size * num_classes);
        
        for b in 0..batch_size {
            // Extract the row
            let row = input.data.slice(s![b, ..]);
        
            // Find the row-wise max for numerical stability
            let max_in_row = row.fold(std::f32::MIN, |acc, &x| acc.max(x));
        
            // Exponentiate each element minus that row-wise max
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_in_row).exp()).collect();
        
            // Sum the exponentials
            let sum_of_exps: f32 = exps.iter().sum();
        
            // Normalize each element by the sum of exponentials
            for val in exps.iter() {
                output_data.push(val / sum_of_exps);
            }
        }
        
        // Return a new Tensor with the same shape
        Tensor::new(output_data, Shape::from(IxDyn(&[batch_size, num_classes])))
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
        Tensor::new(jacobian_data, Shape::from(IxDyn(&[batch_size, num_classes, num_classes])))
    }

    /// Initializes the weights for the Softmax activation function.
    ///
    /// # Arguments
    ///
    /// * `input_units` - The number of input units.
    ///
    /// # Returns
    ///
    /// The initial weight value for the Softmax activation function.
    fn initialize(&self, input_units: Ix) -> f32 {
        (1.0 / input_units as f32).sqrt()   
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
            vec![0.090_030_57, 0.244_728_48, 0.665_240_94],
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
        let expected_data = Array::from_shape_vec(IxDyn(&[1, 3, 3]), vec![
            0.08192507,
            -0.02204622,
            -0.05987885,
            -0.02204622,
            0.18483645,
            -0.16279022,
            -0.05987885,
            -0.16279022,
            0.22266908,
        ])
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
    
    #[test]
    fn test_softmax_initialize() {
        let softmax = SoftmaxActivation::new();
        assert_eq!(softmax.initialize(3), 0.57735026);
    }
}
