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

use std::f64::consts::PI;

use libm::{erff, expf, sqrt};
use ndarray::{Ix, IxDyn, Shape, s};

use std::fmt::Debug;

use super::tensor_ops::Tensor;

/// A trait representing an activation function.
pub trait Activation: Debug {
    /// Applies the activation function to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the activation function.
    fn activate(&self, input: &Tensor) -> Tensor;

    /// Computes the derivative of the activation function.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The derivative tensor of the activation function.
    fn derivative(&self, input: &Tensor) -> Tensor;

    /// Returns the name of the activation function.
    ///
    /// # Returns
    ///
    /// A string slice containing the name of the activation function.
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    /// Initializes the activation function with the given input units.
    ///
    /// # Arguments
    ///
    /// * `input_units` - The number of input units.
    ///
    /// # Returns
    ///
    /// The initialization value for the activation function.
    fn initialize(&self, input_units: Ix) -> f32;
}

/// A struct representing the Gaussian Error Linear Unit (GeLU) activation function.
#[derive(Debug)]
pub struct GeluActivation;

impl Default for GeluActivation {
    fn default() -> Self {
        Self::new()
    }
}

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

/// A struct representing the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
#[derive(Debug)]
pub struct LeakyReluActivation {
    alpha: f32,
}

impl LeakyReluActivation {
    /// Creates a new instance of `LeakyReluActivation` with the given alpha value.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The slope for negative input values.
    #[inline(always)]
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Activation for LeakyReluActivation {
    /// Applies Leaky ReLU activation to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying Leaky ReLU activation.
    #[inline(always)]
    fn activate(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Computes the derivative of Leaky ReLU activation for the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the derivative of Leaky ReLU activation.
    #[inline(always)]
    fn derivative(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(|x| if x > 0.0 { 1.0 } else { alpha })
    }

    /// Initializes the activation function with the given input units.
    ///
    /// # Arguments
    ///
    /// * `input_units` - The number of input units.
    ///
    /// # Returns
    ///
    /// The standard deviation to use for weight initialization.
    fn initialize(&self, input_units: Ix) -> f32 {
        (2.0 / input_units as f32).sqrt()
    }
}

/// A struct representing the Parametric Rectified Linear Unit (PReLU) activation function.
#[derive(Debug)]
pub struct PreluActivation {
    /// The learnable parameter `alpha` for the PReLU activation function.
    pub alpha: f32,
}

impl PreluActivation {
    /// Creates a new instance of `PreluActivation` with the specified `alpha` parameter.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The parameter for the negative slope.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Activation for PreluActivation {
    /// Applies PReLU activation to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying PReLU activation.
    fn activate(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Computes the derivative of PReLU activation for the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the derivative of PReLU activation.
    fn derivative(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(|x| if x > 0.0 { 1.0 } else { alpha })
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

    /// Initializes the activation function with the given input units.
    ///
    /// # Arguments
    ///
    /// * `input_units` - The number of input units.
    ///
    /// # Returns
    ///
    /// The standard deviation to use for weight initialization.
    fn initialize(&self, input_units: Ix) -> f32 {
        (2.0 / input_units as f32).sqrt()
    }
}

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
            let max_in_row = row.fold(f32::MIN, |acc, &x| acc.max(x));

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
    fn test_gelu_activate() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        // Compute the expected GELU outputs for the input tensor.
        let expected_values =
            vec![GeluActivation::gelu(1.0), GeluActivation::gelu(2.0), GeluActivation::gelu(3.0)];
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

    #[test]
    fn test_gelu_initialize() {
        assert_eq!(GeluActivation::new().initialize(10), 0.31622776);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let leaky_relu = LeakyReluActivation::new(0.01);
        let output = leaky_relu.activate(&input);

        assert_eq!(output.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, -0.02, 3.0, -0.04]);
        assert_eq!(output.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let leaky_relu = LeakyReluActivation::new(0.01);
        let derivative = leaky_relu.derivative(&input);

        assert_eq!(derivative.data.iter().cloned().collect::<Vec<f32>>(), vec![
            1.0, 0.01, 1.0, 0.01
        ]);
        assert_eq!(derivative.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_leaky_relu_initialize() {
        let leaky_relu = LeakyReluActivation::new(0.01);
        assert_eq!(leaky_relu.initialize(2), 1.0);
    }

    #[test]
    fn test_prelu_activation() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let prelu = PreluActivation::new(0.1);
        let output = prelu.activate(&input);

        assert_eq!(output.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, -0.2, 3.0, -0.4]);
        assert_eq!(output.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_prelu_derivative() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let prelu = PreluActivation::new(0.1);
        let derivative = prelu.derivative(&input);

        assert_eq!(derivative.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, 0.1, 1.0, 0.1]);
        assert_eq!(derivative.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_prelu_initialize() {
        let prelu = PreluActivation::new(0.1);
        assert_eq!(prelu.initialize(10), 0.31622776);
    }

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

    #[test]
    fn test_relu_initialize() {
        let relu = ReluActivation::new();
        assert_eq!(relu.initialize(10), 0.4472136);
    }

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
