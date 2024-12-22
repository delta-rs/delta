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

use ndarray::{Dimension, IxDyn, Shape};

use crate::common::tensor_ops::Tensor;
use crate::devices::Device;
use crate::neuralnet::layers::Layer;
use crate::neuralnet::layers::error::LayerError;
use crate::optimizers::Optimizer;

/// A struct representing a 2D max pooling layer.
#[derive(Debug)]
pub struct MaxPooling2D {
    /// The size of the pooling window (e.g. 2, 3, etc.).
    pool_size: usize,
    /// The stride (vertical and horizontal) for the pooling window.
    stride: usize,
    /// The shape of the input tensor (once built).
    input_shape: Option<Shape<IxDyn>>,
    /// The mask storing argmax indices for backward pass (one integer per output element).
    argmax_mask: Option<Tensor>,
    /// The device this layer is on.
    device: Device,
}

impl MaxPooling2D {
    /// Creates a new instance of `MaxPooling2D`.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - The size of the pooling window.
    /// * `stride` - The stride of the pooling operation.
    /// * `device` - The initial device to place this layer on.
    pub fn new(pool_size: usize, stride: usize, device: Device) -> Self {
        Self { pool_size, stride, input_shape: None, argmax_mask: None, device }
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
        let dims = input_shape.raw_dim().as_array_view();
        if dims.len() != 4 {
            return Err(LayerError::Build(format!(
                "MaxPooling2D requires a 4D input. Got shape: {:?}",
                dims
            )));
        }

        if self.pool_size == 0 || self.stride == 0 {
            return Err(LayerError::Build("Pool size and stride must be > 0.".to_string()));
        }

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
    /// Forward pass: Performs the max pooling operation and saves an argmax mask.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let in_shape = self
            .input_shape
            .as_ref()
            .ok_or_else(|| LayerError::Build("Layer not built".to_string()))?;

        let dims = in_shape.raw_dim().as_array_view();
        let (batch_size, channels, in_height, in_width) = (dims[0], dims[1], dims[2], dims[3]);

        let out_height = (in_height - self.pool_size) / self.stride + 1;
        let out_width = (in_width - self.pool_size) / self.stride + 1;

        // Use Tensor::zeros instead of ArrayD::zeros
        let mut output =
            Tensor::zeros(Shape::from(IxDyn(&[batch_size, channels, out_height, out_width])));
        let mut mask =
            Tensor::zeros(Shape::from(IxDyn(&[batch_size, channels, out_height, out_width])));

        // Access data through Tensor methods
        let input_data = &input.data;
        let mut output_data = output.data.view_mut();
        let mut mask_data = mask.data.view_mut();

        // Rest of the pooling logic remains the same
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let start_h = oh * self.stride;
                        let start_w = ow * self.stride;

                        let mut max_val = input_data[[b, c, start_h, start_w]];
                        let mut max_idx = start_h * in_width + start_w;

                        for kh in 0..self.pool_size {
                            for kw in 0..self.pool_size {
                                let h = start_h + kh;
                                let w = start_w + kw;
                                let val = input_data[[b, c, h, w]];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = h * in_width + w;
                                }
                            }
                        }

                        output_data[[b, c, oh, ow]] = max_val;
                        mask_data[[b, c, oh, ow]] = max_idx as f32;
                    }
                }
            }
        }

        // Create output tensor using device from input
        let output_tensor = Tensor { data: output_data.to_owned(), device: self.device.clone() };

        // Store mask for backprop
        self.argmax_mask = Some(Tensor { data: mask_data.to_owned(), device: self.device.clone() });

        Ok(output_tensor)
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
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let in_shape = self
            .input_shape
            .as_ref()
            .ok_or_else(|| LayerError::Build("Layer not built".to_string()))?;

        let dims = in_shape.raw_dim().as_array_view();
        let (batch_size, channels, in_height, in_width) = (dims[0], dims[1], dims[2], dims[3]);

        let mask = self.argmax_mask.as_ref().ok_or_else(|| {
            LayerError::Build("Missing argmax mask from forward pass".to_string())
        })?;

        // Use Tensor::zeros instead of ArrayD::zeros
        let mut grad_input =
            Tensor::zeros(Shape::from(IxDyn(&[batch_size, channels, in_height, in_width])));

        let grad_data = &grad.data;
        let mask_data = &mask.data;
        let mut grad_input_data = grad_input.data.view_mut();

        let out_dims = grad_data.shape();
        let (out_height, out_width) = (out_dims[2], out_dims[3]);

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let grad_val = grad_data[[b, c, oh, ow]];
                        let linear_index = mask_data[[b, c, oh, ow]] as usize;
                        let row = linear_index / in_width;
                        let col = linear_index % in_width;
                        grad_input_data[[b, c, row, col]] += grad_val;
                    }
                }
            }
        }

        Ok(Tensor { data: grad_input_data.to_owned(), device: self.device.clone() })
    }

    /// Returns the shape of the output tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the shape of the output tensor or an error.
    /// Returns the shape of the output tensor.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let in_shape = self
            .input_shape
            .as_ref()
            .ok_or_else(|| LayerError::Build("Layer must be built first".to_string()))?;

        let dims = in_shape.raw_dim().as_array_view();
        if dims.len() != 4 {
            return Err(LayerError::Build(format!("Expected a 4D input, got {:?}", dims)));
        }

        let (batch_size, channels, in_height, in_width) = (dims[0], dims[1], dims[2], dims[3]);
        let out_height = (in_height - self.pool_size) / self.stride + 1;
        let out_width = (in_width - self.pool_size) / self.stride + 1;

        // Use IxDyn for dynamic dimensionality
        Ok(Shape::from(IxDyn(&[batch_size, channels, out_height, out_width])))
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
    /// Sets the device for the layer.
    fn set_device(&mut self, device: &Device) {
        // Update the internal device reference
        self.device = device.clone();
        // If you also wanted to move the mask’s data or other Tensors to the new device, do so here.
        if let Some(mask) = &mut self.argmax_mask {
            mask.device = device.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, IxDyn};

    use super::*;

    #[test]
    fn test_basic_max_pooling() {
        let mut layer = MaxPooling2D::new(2, 2, Device::Cpu);
        let input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), vec![
                1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
            ])
            .unwrap(),
            device: Device::Cpu,
        };

        layer.build(Shape::from(IxDyn(&[1, 1, 4, 4]))).unwrap();
        let output = layer.forward(&input).unwrap();

        let expected =
            Array::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![4.0, 6.0, 4.0, 2.0]).unwrap();

        assert_eq!(output.data, expected);
    }

    #[test]
    fn test_backward_pass() {
        let mut layer = MaxPooling2D::new(2, 2, Device::Cpu);

        // Forward pass first
        let input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), vec![
                1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
            ])
            .unwrap(),
            device: Device::Cpu,
        };

        layer.build(Shape::from(IxDyn(&[1, 1, 4, 4]))).unwrap();
        layer.forward(&input).unwrap();

        // Backward pass with gradient of 1.0 for each output
        let grad_output = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
            device: Device::Cpu,
        };

        let grad_input = layer.backward(&grad_output).unwrap();

        // The gradient should flow back to the positions where the max values were found
        // In the forward pass:
        // - 4.0 was at position (1,1)
        // - 6.0 was at position (1,3)
        // - 4.0 was at position (3,1)
        // - 2.0 was at position (3,3)
        let expected_grad = Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ])
        .unwrap();

        assert_eq!(grad_input.data, expected_grad);
    }

    #[test]
    fn test_edge_cases() {
        let mut layer = MaxPooling2D::new(2, 2, Device::Cpu);
        let min_input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            device: Device::Cpu,
        };

        assert!(layer.build(Shape::from(IxDyn(&[1, 1, 2, 2]))).is_ok());
        assert!(layer.forward(&min_input).is_ok());

        let invalid_input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0]).unwrap(),
            device: Device::Cpu,
        };

        assert!(layer.build(Shape::from(IxDyn(&[1, 1, 1]))).is_err());

        let zero_input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), vec![0.0; 16]).unwrap(),
            device: Device::Cpu,
        };

        layer.build(Shape::from(IxDyn(&[1, 1, 4, 4]))).unwrap();
        let output = layer.forward(&zero_input).unwrap();
        assert_eq!(output.data.sum(), 0.0);
    }

    #[test]
    fn test_multi_channel() {
        let mut layer = MaxPooling2D::new(2, 2, Device::Cpu);

        let input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 2, 4, 4]), vec![
                // Channel 1
                1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
                // Channel 2
                2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 6.0, 5.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0,
            ])
            .unwrap(),
            device: Device::Cpu,
        };

        layer.build(Shape::from(IxDyn(&[1, 2, 4, 4]))).unwrap();
        let output = layer.forward(&input).unwrap();

        let expected = Array::from_shape_vec(IxDyn(&[1, 2, 2, 2]), vec![
            4.0, 6.0, 4.0, 2.0, // Channel 1
            4.0, 6.0, 4.0, 2.0, // Channel 2
        ])
        .unwrap();

        assert_eq!(output.data, expected);
    }

    #[test]
    fn test_stride_variations() {
        let mut layer = MaxPooling2D::new(2, 1, Device::Cpu);

        let input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 3, 3]), vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ])
            .unwrap(),
            device: Device::Cpu,
        };

        layer.build(Shape::from(IxDyn(&[1, 1, 3, 3]))).unwrap();
        let output = layer.forward(&input).unwrap();

        let expected =
            Array::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![5.0, 6.0, 8.0, 9.0]).unwrap();

        assert_eq!(output.data, expected);
    }

    #[test]
    fn test_negative_values() {
        let mut layer = MaxPooling2D::new(2, 2, Device::Cpu);

        let input = Tensor {
            data: Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), vec![
                -1.0, -2.0, -1.0, -3.0, -2.0, -4.0, -5.0, -6.0, -1.0, -2.0, -1.0, -2.0, -3.0, -4.0,
                -1.0, -2.0,
            ])
            .unwrap(),
            device: Device::Cpu,
        };

        layer.build(Shape::from(IxDyn(&[1, 1, 4, 4]))).unwrap();
        let output = layer.forward(&input).unwrap();

        let expected =
            Array::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![-1.0, -1.0, -1.0, -1.0]).unwrap();

        assert_eq!(output.data, expected);
    }
}
