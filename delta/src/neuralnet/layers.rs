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
use crate::common::layer::Layer;
use crate::common::shape::Shape;
use crate::common::tensor_ops::Tensor;

/// A dense (fully connected) layer.
#[derive(Debug)]
pub struct Dense {
    name: String,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    units: usize,
    activation: Box<dyn Activation>,
    trainable: bool,
    weights_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    input: Option<Tensor>,
}

impl Dense {
    /// Creates a new dense layer.
    ///
    /// # Arguments
    ///
    /// * `units` - The number of units in the layer.
    /// * `activation` - The activation function to use.
    ///
    /// # Returns
    ///
    /// A new instance of the dense layer.
    pub fn new<A: Activation + 'static>(units: usize, activation: A, trainable: bool) -> Self {
        Self {
            name: format!("Dense_{}", units),
            weights: None,
            bias: None,
            units,
            activation: Box::new(activation),
            trainable,
            weights_grad: None,
            bias_grad: None,
            input: None,
        }
    }
}

impl Layer for Dense {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape) {
        self.weights = Some(Tensor::random(&Shape::from((
            input_shape.len(),
            self.units,
        ))));

        self.bias = Some(Tensor::zeros(&Shape::new(vec![self.units])));
    }

    /// Performs a forward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn forward(&mut self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape.0.last().unwrap(),
            &self.weights.as_ref().unwrap().shape.0[0],
            "Input shape does not match Dense layer weight dimensions"
        );

        // Store the input tensor for use in backward pass
        self.input = Some(input.clone());

        let z = input.matmul(&self.weights.as_ref().unwrap()).zip_map(
            &self
                .bias
                .as_ref()
                .unwrap()
                .reshape(Shape::new(vec![1, self.units])),
            |a, b| a + b,
        );

        self.activation.activate(&z)
    }

    /// Performs a backward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    ///
    /// # Returns
    ///
    /// The gradient tensor with respect to the input.
    fn backward(&mut self, grad: &Tensor) -> Tensor {
        // Ensure the layer was built and the weights/bias exist
        assert!(
            self.weights.is_some() && self.bias.is_some(),
            "Dense layer must be built before performing backward pass"
        );

        // Extract the weights and bias tensors
        let weights = self.weights.as_ref().unwrap();

        // 1. Gradient of the activation function applied to the pre-activation output
        let activation_grad = self.activation.derivative(&grad);

        // 2. Compute the gradient for weights: input.T dot activation_grad
        let input_transposed = self.input.as_ref().unwrap().transpose();
        let weight_grad = input_transposed.matmul(&activation_grad);

        // 3. Compute the gradient for bias: sum of activation_grad along batch dimension
        let bias_grad = Tensor::new(vec![activation_grad.sum()], Shape::new(vec![self.units]));

        // 4. Compute the gradient for the input to propagate to the previous layer
        let input_grad = activation_grad.matmul(&weights.transpose());

        // Update weights and bias gradients if trainable
        if self.trainable {
            self.weights_grad = Some(weight_grad);
            self.bias_grad = Some(bias_grad);
        }

        input_grad
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Shape {
        Shape::new(vec![self.units])
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of parameters in the layer.
    fn param_count(&self) -> (usize, usize) {
        let weights_count = self.weights.as_ref().map_or(0, |w| w.data.len());
        let bias_count = self.bias.as_ref().map_or(0, |b| b.data.len());
        let total_params = weights_count + bias_count;
        if self.trainable {
            (total_params, 0)
        } else {
            (0, total_params)
        }
    }

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of units in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of units in the layer.
    fn units(&self) -> usize {
        self.units
    }
}

/// A flatten layer that reshapes the input tensor to a 1D vector.
#[derive(Debug)]
pub struct Flatten {
    name: String,
    input_shape: Shape,
}

impl Flatten {
    /// Creates a new flatten layer.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    ///
    /// # Returns
    ///
    /// A new instance of the flatten layer.
    pub fn new(input_shape: Shape) -> Self {
        Self {
            name: "Flatten".to_string(),
            input_shape,
        }
    }
}

impl Layer for Flatten {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape) {
        self.input_shape = input_shape;
    }

    /// Performs a forward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let batch_size = input.shape.0[0];
        let flattened_size = input.shape.0.iter().skip(1).product::<usize>();
        Tensor::new(
            input.data.clone(),
            Shape::new(vec![batch_size, flattened_size]),
        )
    }

    /// Performs a backward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    ///
    /// # Returns
    ///
    /// The gradient tensor with respect to the input.
    fn backward(&mut self, grad: &Tensor) -> Tensor {
        // Reshape the gradient back to the original input shape
        grad.reshape(self.input_shape.clone())
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Shape {
        Shape::new(vec![self.input_shape.0.iter().product()])
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of parameters in the layer.
    fn param_count(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::relu::ReluActivation;
    use super::*;

    #[test]
    fn test_dense_layer() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let mut dense_layer = Dense::new(2, ReluActivation::new(), true);
        dense_layer.build(Shape::new(vec![1, 3]));

        let output = dense_layer.forward(&input);

        assert_eq!(output.shape.0, vec![1, 2]);
        // Assert values are within expected range; exact values depend on random weights.
        assert!(output.data.len() == 2);
    }

    #[test]
    fn test_dense_layer_forward_pass() {
        let input = Tensor::new(
            (0..784).map(|x| x as f32).collect(),
            Shape::new(vec![1, 784]),
        );
        let mut dense_layer = Dense::new(128, ReluActivation::new(), true);
        dense_layer.build(Shape::new(vec![1, 784]));

        let output = dense_layer.forward(&input);

        assert_eq!(output.shape.0, vec![1, 128]);
    }
}
