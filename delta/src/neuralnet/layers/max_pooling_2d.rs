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

use crate::common::tensor_ops::Tensor;
use crate::neuralnet::layers::error::LayerError;
use crate::neuralnet::layers::Layer;
use crate::optimizers::Optimizer;
use ndarray::{IxDyn, Shape};

/// A struct representing a 2D max pooling layer.
#[derive(Debug)]
pub struct MaxPooling2D {
    /// The size of the pooling window.
    #[allow(dead_code)]
    pool_size: usize,
    /// The stride of the pooling operation.
    #[allow(dead_code)]
    stride: usize,
    /// The shape of the input tensor.
    input_shape: Option<Shape<IxDyn>>,
}

impl MaxPooling2D {
    /// Creates a new instance of `MaxPooling2D`.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - The size of the pooling window.
    /// * `stride` - The stride of the pooling operation.
    ///
    /// # Returns
    ///
    /// A new `MaxPooling2D` instance.
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self {
            pool_size,
            stride,
            input_shape: None,
        }
    }
}

impl Layer for MaxPooling2D {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        self.input_shape = Some(input_shape);
        Ok(())
    }

    /// Performs the forward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `_input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the output tensor or an error.
    fn forward(&mut self, _input: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!()
    }

    /// Performs the backward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `_grad` - The gradient tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the gradient with respect to the input or an error.
    fn backward(&mut self, _grad: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!()
    }

    /// Returns the shape of the output tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the shape of the output tensor or an error.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        unimplemented!()
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `Result` containing a tuple with the number of trainable and non-trainable parameters.
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        Ok((0, 0))
    }

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A string slice containing the name of the layer.
    fn name(&self) -> &str {
        "MaxPooling2D"
    }

    /// Updates the weights of the layer using the given optimizer.
    ///
    /// # Arguments
    ///
    /// * `_optimizer` - The optimizer to use for updating the weights.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    fn update_weights(&mut self, _optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        unimplemented!()
    }

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `_device` - The device to set for the layer.
    fn set_device(&mut self, _device: &crate::devices::Device) {
        unimplemented!()
    }
}
