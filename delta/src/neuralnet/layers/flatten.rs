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

use crate::common::{Shape, Tensor};
use crate::neuralnet::layers::error::LayerError;
use crate::neuralnet::layers::Layer;

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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::neuralnet::layers::flatten::Flatten;
    /// use deltaml::common::Shape;
    ///
    /// let flatten_layer = Flatten::new(Shape::new(vec![28, 28]));
    /// ```
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
    fn build(&mut self, input_shape: Shape) -> Result<(), LayerError> {
        self.input_shape = input_shape;
        Ok(())
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
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = input.data.shape()[0];
        let flattened_size = input.data.len() / batch_size;
        let reshaped = input.reshape(vec![batch_size, flattened_size]);
        Ok(reshaped)
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
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = grad.shape()[0];
        let new_shape = [batch_size]
            .iter()
            .chain(self.input_shape.0.iter())
            .cloned()
            .collect::<Vec<_>>();
        Ok(grad.reshape(new_shape))
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape, LayerError> {
        let shape = Shape::new(vec![self.input_shape.0.iter().product()]);
        Ok(shape)
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of parameters in the layer.
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        Ok((0, 0))
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
    fn update_weights(
        &mut self,
        optimizer: &mut Box<dyn crate::optimizers::Optimizer>,
    ) -> Result<(), LayerError> {
        let _ = optimizer;
        // Do nothing
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Shape;

    #[test]
    fn test_flatten_new() {
        let input_shape = Shape::new(vec![28, 28]);
        let flatten_layer = Flatten::new(input_shape.clone());

        assert_eq!(flatten_layer.name(), "Flatten");
        assert_eq!(flatten_layer.input_shape, input_shape);
    }

    #[test]
    fn test_flatten_output_shape() {
        let input_shape = Shape::new(vec![3, 4]);
        let flatten_layer = Flatten::new(input_shape.clone());

        let output_shape = flatten_layer.output_shape().unwrap();
        assert_eq!(output_shape.0, vec![12]);
    }

    #[test]
    fn test_flatten_param_count() {
        let flatten_layer = Flatten::new(Shape::new(vec![10, 10]));
        let (trainable, non_trainable) = flatten_layer.param_count().unwrap();

        assert_eq!(trainable, 0);
        assert_eq!(non_trainable, 0);
    }
}
