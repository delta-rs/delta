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

pub mod conv_1d;
pub mod dense;
pub mod error;
pub mod flatten;
pub mod max_pooling_2d;

use std::fmt::Debug;

pub use conv_1d::Conv1D;
pub use dense::Dense;
pub use flatten::Flatten;
pub use max_pooling_2d::MaxPooling2D;
use ndarray::{IxDyn, Shape};

use crate::common::Tensor;
use crate::devices::Device;
use crate::neuralnet::layers::error::LayerError;
use crate::optimizers::Optimizer;

// A trait representing a neural network layer.
pub trait Layer: Debug {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError>;

    /// Performs the forward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the layer.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError>;

    /// Performs the backward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor from the next layer.
    ///
    /// # Returns
    ///
    /// The gradient tensor to be passed to the previous layer.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError>;

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError>;

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A tuple `(usize, usize)` representing the number of trainable and non-trainable parameters in the layer.
    fn param_count(&self) -> Result<(usize, usize), LayerError>;

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str;

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, _device: &Device);

    /// Returns the number of units in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of units in the layer. Default is 0.
    fn units(&self) -> usize {
        0
    }

    /// Updates the weights of the layer.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError>;

    /// Returns the weights of the layer as a serializable format.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` containing the layer's weights.
    fn get_weights(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    /// Returns the layer's configuration as a serializable format.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` containing the layer's configuration.
    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    /// Returns the type name of the layer.
    ///
    /// # Returns
    ///
    /// A `String` representing the type name of the layer.
    fn type_name(&self) -> String {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown").to_string()
    }
}

/// A struct representing the output of a layer.
#[derive(Debug)]
pub struct LayerOutput {
    /// The output tensor of the layer.
    pub output: Tensor,
    /// The gradients tensor of the layer.
    pub gradients: Tensor,
}
