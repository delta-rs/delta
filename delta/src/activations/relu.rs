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

use crate::activations::Activation;
use crate::common::Tensor;

/// A struct representing the Rectified Linear Unit (ReLU) activation function.
#[derive(Debug)]
pub struct ReluActivation;

impl Default for ReluActivation {
    fn default() -> Self {
        Self::new()
    }
}

impl ReluActivation {
    /// Creates a new instance of `ReluActivation`.
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }
}

impl Activation for ReluActivation {
    /// Applies ReLU activation to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying ReLU activation.
    #[inline(always)]
    fn activate(&self, input: &Tensor) -> Tensor {
        input.map_max(0.0)
    }

    /// Computes the derivative of ReLU activation for the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the derivative of ReLU activation.
    #[inline(always)]
    fn derivative(&self, input: &Tensor) -> Tensor {
        input.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;

    #[test]
    fn test_relu_activation() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let relu = ReluActivation::new();
        let output = relu.activate(&input);

        assert_eq!(output.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, 0.0, 3.0, 0.0]);
        assert_eq!(output.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_relu_derivative() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let relu = ReluActivation::new();
        let derivative = relu.derivative(&input);

        assert_eq!(derivative.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!(derivative.data.shape().to_vec(), vec![2, 2]);
    }
}
