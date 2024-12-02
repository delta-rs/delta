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

use crate::common::Activation;
use crate::common::Layer;
use crate::common::Shape;
use crate::common::Tensor;

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
        println!(
            "Building Dense layer with input shape: {:?} and units: {}",
            input_shape, self.units
        );
        let input_units = input_shape.0.last().expect("Input shape must not be empty");
        self.weights = Some(Tensor::random(vec![*input_units, self.units]));
        self.bias = Some(Tensor::zeros(vec![self.units]));
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
        let weights = self.weights.as_ref().expect("Weights must be initialized");
        let bias = self.bias.as_ref().expect("Bias must be initialized");

        self.input = Some(input.clone());

        // Perform forward pass: Z = input · weights + bias
        let z = input.matmul(weights).add(bias);
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
        // Ensure weights and input are initialized
        let weights = self.weights.as_ref().expect("Weights must be initialized");
        let input = self.input.as_ref().expect("Input must be initialized");

        // Calculate the gradient with respect to weights and bias
        let weights_grad = input.transpose().matmul(grad);
        let bias_grad = grad.sum_along_axis(0);

        // Store the gradients
        if self.trainable {
            self.weights_grad = Some(weights_grad);
            self.bias_grad = Some(bias_grad);
        }

        // Calculate the gradient with respect to the input
        let input_grad = grad.matmul(&weights.transpose());

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
        (weights_count, bias_count)
    }

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str {
        &self.name
    }

    /// Updates the weights of the layer using the given gradient and optimizer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn crate::common::optimizer::Optimizer>) {
        if !self.trainable {
            return;
        }

        // Update weights
        if let Some(ref weights_grad) = self.weights_grad {
            optimizer.step(self.weights.as_mut().unwrap(), weights_grad);
        }

        // Update bias
        if let Some(ref bias_grad) = self.bias_grad {
            optimizer.step(self.bias.as_mut().unwrap(), bias_grad);
        }

        // Clear gradients after update
        self.weights_grad = None;
        self.bias_grad = None;
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
        let batch_size = input.data.shape()[0];
        let flattened_size = input.data.len() / batch_size;
        input.reshape(vec![batch_size, flattened_size])
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
        let batch_size = grad.shape()[0];
        let new_shape = [batch_size]
            .iter()
            .chain(self.input_shape.0.iter())
            .cloned()
            .collect::<Vec<_>>();
        grad.reshape(new_shape)
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

    /// Updates the weights of the layer using the given gradient and optimizer.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn crate::common::optimizer::Optimizer>) {
        let _ = optimizer;
        // Do nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::relu::ReluActivation;

    #[test]
    fn test_dense_layer() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let mut dense_layer = Dense::new(2, ReluActivation::new(), true);
        dense_layer.build(Shape::new(vec![1, 3]));

        let output = dense_layer.forward(&input);

        assert_eq!(output.data.shape(), &[1, 2]);
        assert_eq!(output.data.len(), 2);
    }

    #[test]
    fn test_dense_layer_forward_pass() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let mut dense_layer = Dense::new(2, ReluActivation::new(), true);
        dense_layer.build(Shape::new(vec![1, 3]));

        let output = dense_layer.forward(&input);

        assert_eq!(output.data.shape(), &[1, 2]);
        assert_eq!(output.data.len(), 2);
    }

    #[test]
    fn test_dense_layer_backward_pass() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let mut dense_layer = Dense::new(2, ReluActivation::new(), true);
        dense_layer.input = Some(input.clone());
        dense_layer.build(Shape::new(vec![1, 3]));

        let grad = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let output = dense_layer.backward(&grad);

        assert_eq!(output.data.shape(), &[1, 3]);
        assert_eq!(output.data.len(), 3);
    }
}
