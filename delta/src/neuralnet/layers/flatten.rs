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

use ndarray::{Dimension, IxDyn, Shape};

use crate::common::Tensor;
use crate::neuralnet::layers::Layer;
use crate::neuralnet::layers::error::LayerError;

/// A flatten layer that reshapes the input tensor to a 1D vector.
#[derive(Debug)]
pub struct Flatten {
    name: String,
    input_shape: Shape<IxDyn>,
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
    pub fn new(input_shape: &[usize]) -> Self {
        Self { name: "Flatten".to_string(), input_shape: Shape::from(IxDyn(&input_shape)) }
    }
}

impl Layer for Flatten {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
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
        let reshaped = input.reshape(IxDyn(&[batch_size, flattened_size]));
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
        let batch_size = grad.shape().raw_dim().as_array_view()[0];
        let new_shape = [batch_size]
            .iter()
            .chain(self.input_shape.raw_dim().as_array_view().iter())
            .cloned()
            .collect::<Vec<_>>();

        Ok(grad.reshape(IxDyn(&new_shape)))
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let shape =
            Shape::from(IxDyn(&[self.input_shape.raw_dim().as_array_view().iter().product()]));
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

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, _device: &crate::devices::Device) {
        // Do nothing
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;

    #[test]
    fn test_flatten_new() {
        let input_shape = Shape::from(IxDyn(&[28, 28]));
        let flatten_layer = Flatten::new(&[28, 28]);

        assert_eq!(flatten_layer.name(), "Flatten");
        assert_eq!(flatten_layer.input_shape.raw_dim(), input_shape.raw_dim());
    }

    #[test]
    fn test_flatten_output_shape() {
        let flatten_layer = Flatten::new(&[3, 4]);

        let output_shape = flatten_layer.output_shape().unwrap();
        assert_eq!(output_shape.raw_dim().as_array_view().to_vec(), vec![12]);
    }

    #[test]
    fn test_flatten_param_count() {
        let flatten_layer = Flatten::new(&[10, 10]);
        let (trainable, non_trainable) = flatten_layer.param_count().unwrap();

        assert_eq!(trainable, 0);
        assert_eq!(non_trainable, 0);
    }
}
