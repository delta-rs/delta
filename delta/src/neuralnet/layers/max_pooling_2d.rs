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

use ndarray::{
    Array, Array4, Axis, Dimension, Ix4, IxDyn, Shape, Zip, s
};
use crate::common::tensor_ops::Tensor;
use crate::neuralnet::layers::Layer;
use crate::neuralnet::layers::error::LayerError;
use crate::optimizers::Optimizer;

/// A struct representing a 2D max pooling layer.
#[derive(Debug)]
pub struct MaxPooling2D {
    /// The size of the pooling window.
    pool_size: usize,
    /// The stride of the pooling operation.
    stride: usize,
    /// The shape of the input tensor (expected to be 4D: N x C x H x W).
    input_shape: Option<Shape<IxDyn>>,
    /// Stores the positions of maximum values from the forward pass.
    /// The array is shaped `[N, C, outH, outW]`, each element holding `(row, col)` in the original window.
    max_indices: Option<Array<(usize, usize), Ix4>>,
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
            max_indices: None,
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
        // Expect 4D input: [N, C, H, W].
        if input_shape.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        self.input_shape = Some(input_shape);
        Ok(())
    }

    /// Performs the forward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the output tensor or an error.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let shape = match &self.input_shape {
            Some(s) => s,
            None => return Err(LayerError::MissingInput),
        };

        // Shape should be [N, C, H, W].
        let dims = shape.dims();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Convert the input's data to a 4D array.
        // If it fails to reshape, return an error.
        let input_4d = input
            .data
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;

        // Compute output height and width.
        // For valid pooling, assume the user ensures that (H - pool_size) % stride == 0, etc.
        if self.pool_size > h || self.pool_size > w {
            return Err(LayerError::InvalidInputShape);
        }
        let out_h = (h - self.pool_size) / self.stride + 1;
        let out_w = (w - self.pool_size) / self.stride + 1;

        // Prepare output and index arrays.
        let mut output = Array::<f32, Ix4>::zeros((n, c, out_h, out_w));
        let mut max_positions = Array::<(usize, usize), Ix4>::zeros((n, c, out_h, out_w));

        // Perform max pooling.
        for batch in 0..n {
            for ch in 0..c {
                for out_i in 0..out_h {
                    for out_j in 0..out_w {
                        let start_i = out_i * self.stride;
                        let start_j = out_j * self.stride;
                        let end_i = start_i + self.pool_size;
                        let end_j = start_j + self.pool_size;

                        let window = input_4d.slice(s![batch, ch, start_i..end_i, start_j..end_j]);

                        let mut max_val = f32::MIN;
                        let mut max_pos = (0, 0);
                        for wi in 0..self.pool_size {
                            for wj in 0..self.pool_size {
                                let val = window[[wi, wj]];
                                if val > max_val {
                                    max_val = val;
                                    max_pos = (wi, wj);
                                }
                            }
                        }
                        output[[batch, ch, out_i, out_j]] = max_val;
                        max_positions[[batch, ch, out_i, out_j]] = (start_i + max_pos.0, start_j + max_pos.1);
                    }
                }
            }
        }

        self.max_indices = Some(max_positions);

        // Convert output back to a dynamic shape tensor.
        let output_tensor = Tensor {
            data: output.into_dyn(),
            device: input.device.clone(),
        };
        Ok(output_tensor)
    }

    /// Performs the backward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor from the next layer.
    ///
    /// # Returns
    ///
    /// A `Result` containing the gradient with respect to the input or an error.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let shape = match &self.input_shape {
            Some(s) => s,
            None => return Err(LayerError::MissingInput),
        };

        // Expect [N, C, H, W].
        let dims = shape.dims();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Convert grad to [N, C, outH, outW].
        let grad_4d = grad
            .data
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;

        // Check we have stored max_indices from forward().
        let max_indices = match &self.max_indices {
            Some(mi) => mi,
            None => return Err(LayerError::MissingInput),
        };

        // Prepare output gradient for the input (same shape as the original input).
        let mut dx = Array::<f32, Ix4>::zeros((n, c, h, w));

        // Scatter the gradients back to where the maxima were found.
        for batch in 0..n {
            for ch in 0..c {
                let out_h = grad_4d.shape()[2];
                let out_w = grad_4d.shape()[3];
                for out_i in 0..out_h {
                    for out_j in 0..out_w {
                        let (src_i, src_j) = max_indices[[batch, ch, out_i, out_j]];
                        dx[[batch, ch, src_i, src_j]] += grad_4d[[batch, ch, out_i, out_j]];
                    }
                }
            }
        }

        // Return the gradient w.r.t. the input as a Tensor.
        Ok(Tensor {
            data: dx.into_dyn(),
            device: grad.device.clone(),
        })
    }

    /// Returns the shape of the output tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the shape of the output tensor or an error.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let shape = match &self.input_shape {
            Some(s) => s,
            None => return Err(LayerError::MissingInput),
        };

        if shape.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        let dims = shape.dims();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        if self.pool_size > h || self.pool_size > w {
            return Err(LayerError::InvalidInputShape);
        }

        let out_h = (h - self.pool_size) / self.stride + 1;
        let out_w = (w - self.pool_size) / self.stride + 1;

        let out_shape = Shape::from(IxDyn(&[n, c, out_h, out_w]));
        Ok(out_shape)
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `Result` containing a tuple with the number of trainable and non-trainable parameters.
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        // Max pooling has no trainable parameters.
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
    /// * `optimizer` - The optimizer to use for updating the weights.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    fn update_weights(&mut self, _optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        // No weights in a max pooling layer.
        Ok(())
    }

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, _device: &crate::devices::Device) {
        // No device-specific resources for pure CPU pooling
    }
}
