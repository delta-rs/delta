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

use delta_common::{tensor_ops::Tensor, Activation};

#[derive(Debug)]
pub struct ReluActivation;

impl ReluActivation {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for ReluActivation {
    /// Applies the Rectified Linear Unit (ReLU) activation function to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the ReLU activation function.
    ///
    /// # Examples
    ///
    /// ```
    /// use delta_activations::relu::ReluActivation;
    /// use delta_common::tensor_ops::Tensor;
    ///
    /// let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::new(vec![2, 2]));
    /// let relu = ReluActivation::new();
    /// let output = relu.activate(&input);
    ///
    /// assert_eq!(output.data, vec![1.0, 0.0, 3.0, 0.0]);
    /// assert_eq!(output.shape.0, vec![2, 2]);
    /// ```
    fn activate(&self, input: &Tensor) -> Tensor {
        input.map(|x| x.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_common::{tensor_ops::Tensor, Shape};

    #[test]
    fn test_relu_activation() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::new(vec![2, 2]));
        let relu = ReluActivation::new();
        let output = relu.activate(&input);

        assert_eq!(output.data, vec![1.0, 0.0, 3.0, 0.0]);
        assert_eq!(output.shape.0, vec![2, 2]);
    }
}
