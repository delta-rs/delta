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

use std::f64::consts::PI;

use libm::erff;
use libm::expf;
use libm::sqrt;

use crate::activations::Activation;
use crate::common::Tensor;

/// A struct representing the Gaussian Error Linear Unit (GeLU) activation function.
#[derive(Debug)]
pub struct GeluActivation;

impl GeluActivation {
    /// Creates a new instance of `GeluActivation`.
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    /// Applies the GELU formula.
    ///
    /// # Arguments
    ///
    /// * `x` - The input value.
    ///
    /// # Returns
    ///
    /// The output value after applying the GELU formula.
    fn gelu(x: f32) -> f32 {
        let sqrt_2: f32 = sqrt(2.0) as f32;
        x * 0.5 * (1.0 + erff(x / sqrt_2))
    }

    /// Computes the derivative of GELU.
    ///
    /// # Arguments
    ///
    /// * `x` - The input value.
    ///
    /// # Returns
    ///
    /// The derivative of GELU at the input value.
    fn gelu_derivative(x: f32) -> f32 {
        let sqrt_2: f32 = sqrt(2.0) as f32;
        let sqrt_2_pi: f32 = sqrt(2.0 * PI) as f32;
        let cdf = 0.5 * (1.0 + erff(x / sqrt_2));
        let pdf = sqrt_2_pi * expf(-0.5 * x * x);
        cdf + x * pdf
    }
}

impl Activation for GeluActivation {
    /// Applies GeLU activation to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying GeLU activation.
    #[inline(always)]
    fn activate(&self, input: &Tensor) -> Tensor {
        input.map(Self::gelu)
    }

    /// Computes the derivative of GeLU activation for the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the derivative of GeLU activation.
    #[inline(always)]
    fn derivative(&self, input: &Tensor) -> Tensor {
        input.map(Self::gelu_derivative)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;

    #[test]
    fn test_gelu_activate() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        // Compute the expected GELU outputs for the input tensor.
        let expected_values = vec![
            GeluActivation::gelu(1.0),
            GeluActivation::gelu(2.0),
            GeluActivation::gelu(3.0),
        ];
        let output = Tensor::new(expected_values, Shape::from(IxDyn(&[1, 3])));
        assert_eq!(GeluActivation::new().activate(&input), output);
    }

    #[test]
    fn test_gelu_derivative() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        // Compute the expected GELU derivatives for the input tensor.
        let expected_values = vec![
            GeluActivation::gelu_derivative(1.0),
            GeluActivation::gelu_derivative(2.0),
            GeluActivation::gelu_derivative(3.0),
        ];
        let output = Tensor::new(expected_values, Shape::from(IxDyn(&[1, 3])));
        assert_eq!(GeluActivation::new().derivative(&input), output);
    }
}
