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

use log::debug;
use ndarray::{ Array, Dim, Dimension, Ix4, IxDyn, IxDynImpl, Shape, s };
use serde::Deserialize;
use serde_json;

use crate::activations::Activation;
use crate::common::Tensor;
use crate::devices::Device;
use crate::neuralnet::layers::Layer;
use crate::neuralnet::layers::error::LayerError;
use crate::optimizers::Optimizer;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, Deserialize)]
pub enum PaddingMode {
    Valid, // No padding
    Same, // Pad to keep same spatial dimensions
    Custom(usize, usize), // Custom padding for height and width
}

/// A 2D convolutional layer.
#[derive(Debug)]
pub struct Conv2D {
    name: String,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    kernel_units: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize), // Dilation rate for each dimension
    groups: usize, // Number of groups for grouped convolution
    include_bias: bool,
    activation: Option<Box<dyn Activation>>,
    trainable: bool,
    weights_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    input: Option<Tensor>,
    input_shape: Option<Shape<IxDyn>>,
    device: Device,
}

impl Conv2D {
    /// Creates a new Conv2D layer with advanced features.
    pub fn new<A: Activation + 'static>(
        kernel_units: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
        activation: Option<A>,
        trainable: bool,
        include_bias: bool
    ) -> Self {
        if groups > kernel_units {
            panic!("Number of groups can not be higher than number of kernel units");
        }
        Conv2D {
            name: format!("conv2d_{}", kernel_units),
            weights: None,
            bias: None,
            kernel_units,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            include_bias,
            activation: activation.map(|a| Box::new(a) as Box<dyn Activation>),
            trainable,
            weights_grad: None,
            bias_grad: None,
            input: None,
            input_shape: None,
            device: Device::default(),
        }
    }

    fn get_input_shape(&self) -> Result<Dim<IxDynImpl>, LayerError> {
        let shape = self.input_shape.as_ref().ok_or(LayerError::MissingInput)?;
        let raw = shape.raw_dim();
        if raw.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        Ok(raw.clone())
    }

    /// Calculate padding sizes based on padding mode
    fn get_padding_sizes(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        match self.padding {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => {
                let pad_height =
                    ((input_height - 1) * self.stride.0 +
                        (self.kernel_size.0 - 1) * self.dilation.0 +
                        1 -
                        input_height) /
                    2;
                let pad_width =
                    ((input_width - 1) * self.stride.1 +
                        (self.kernel_size.1 - 1) * self.dilation.1 +
                        1 -
                        input_width) /
                    2;
                (pad_height, pad_width)
            }
            PaddingMode::Custom(h, w) => (h, w),
        }
    }

    /// Apply padding to input tensor
    fn pad_input(&self, input: &Array<f32, Ix4>) -> Array<f32, Ix4> {
        let (batch_size, channels, height, width) = input.dim();
        let (pad_h, pad_w) = self.get_padding_sizes(height, width);

        if pad_h == 0 && pad_w == 0 {
            return input.clone();
        }

        let mut padded = Array::zeros((
            batch_size,
            channels,
            height + 2 * pad_h,
            width + 2 * pad_w,
        ));

        padded.slice_mut(s![.., .., pad_h..pad_h + height, pad_w..pad_w + width]).assign(input);

        padded
    }
}

/// 2D Convolution operation
///
/// # Arguments
///
/// * `input` - Input feature map [batch_size, channels, height, width]
/// * `kernel` - Convolution kernel [out_channels, in_channels, kernel_height, kernel_width]
/// * `stride` - (height_stride, width_stride)
///
/// # Returns
///
/// Output feature map
fn conv2d_raw(
    input: &Array<f32, Ix4>,
    kernel: &Array<f32, Ix4>,
    stride: (usize, usize),
    dilation: (usize, usize),
    groups: usize
) -> Array<f32, Ix4> {
    let (batch_size, in_channels, in_height, in_width) = input.dim();
    let (out_channels, _, kernel_height, kernel_width) = kernel.dim();

    let out_height = (in_height - (kernel_height - 1) * dilation.0 - 1) / stride.0 + 1;
    let out_width = (in_width - (kernel_width - 1) * dilation.1 - 1) / stride.1 + 1;

    if in_channels % groups != 0 {
        panic!("Input channels must be divisible by the number of groups");
    }

    if out_channels % groups != 0 {
        panic!("Output channels must be divisible by the number of groups");
    }

    let channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let mut output = Array::zeros((batch_size, out_channels, out_height, out_width));

    for g in 0..groups {
        let in_start = g * channels_per_group;
        let in_end = (g + 1) * channels_per_group;
        let out_start = g * out_channels_per_group;
        let out_end = (g + 1) * out_channels_per_group;

        let input_slice = input.slice(s![.., in_start..in_end, .., ..]);
        let kernel_slice = kernel.slice(s![out_start..out_end, .., .., ..]);

        for b in 0..batch_size {
            for oc in 0..out_channels_per_group {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;
                        for ic in 0..channels_per_group {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * stride.0 + kh * dilation.0;
                                    let iw = ow * stride.1 + kw * dilation.1;
                                    if ih < in_height && iw < in_width {
                                        sum +=
                                            input_slice[[b, ic, ih, iw]] *
                                            kernel_slice[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                        output[[b, out_start + oc, oh, ow]] = sum;
                    }
                }
            }
        }
    }

    output
}

impl Layer for Conv2D {
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        debug!(
            "Building Conv2D layer with input shape: {:?}, kernel_units: {}, kernel_size: {:?}, stride: {:?}, padding: {:?}",
            input_shape,
            self.kernel_units,
            self.kernel_size,
            self.stride,
            self.padding
        );

        self.input_shape = Some(input_shape);
        let raw = self.get_input_shape()?;
        let (batch_size, in_channels, height, width) = (raw[0], raw[1], raw[2], raw[3]);

        // Initialize weights using He or Xavier initialization
        let fan_in = in_channels * self.kernel_size.0 * self.kernel_size.1;
        let stddev = if let Some(ref activation) = self.activation {
            match activation.name() {
                "relu" | "leaky_relu" => (2.0 / (fan_in as f32)).sqrt(), // He
                _ => (1.0 / (fan_in as f32)).sqrt(), // Xavier
            }
        } else {
            (1.0 / (fan_in as f32)).sqrt()
        };

        if self.kernel_units % self.groups != 0 {
            return Err(LayerError::InvalidParameters);
        }
        if in_channels % self.groups != 0 {
            return Err(LayerError::InvalidParameters);
        }

        // Shape: [out_channels, in_channels/groups, kernel_height, kernel_width]
        self.weights = Some(
            Tensor::random_normal(
                Shape::from(
                    IxDyn(
                        &[
                            self.kernel_units,
                            in_channels / self.groups, // Correct shape for grouped convolutions
                            self.kernel_size.0,
                            self.kernel_size.1,
                        ]
                    )
                ),
                0.0,
                stddev
            )
        );

        self.weights.as_mut().unwrap().device = self.device.clone();

        // Initialize bias
        if self.include_bias {
            self.bias = Some(
                Tensor::zeros(
                    Shape::from(IxDyn(&[1, self.kernel_units, 1, 1])),
                    self.device.clone()
                )
            );
        }

        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.input = Some(input.clone());

        let weights = self.weights.as_ref().expect("Weights must be initialized");
        let bias = self.bias.as_ref();

        // Convert input to 4D
        let input_4d = input.data
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;

        // Apply padding to the input data
        let padded_input = self.pad_input(&input_4d);

        // Perform convolution
        let mut output = conv2d_raw(
            &padded_input,
            &weights.data
                .clone()
                .into_dimensionality::<Ix4>()
                .map_err(|_| LayerError::InvalidInputShape)?,
            self.stride,
            self.dilation,
            self.groups
        );

        // Add bias if present
        if let Some(bias) = bias {
            output = output + bias.data.clone().into_dimensionality::<Ix4>().unwrap();
        }

        // Convert to Tensor and apply activation
        let mut z = Tensor {
            data: output.into_dyn(),
            device: input.device.clone(),
        };

        if let Some(ref activation) = self.activation {
            z = activation.activate(&z);
        }

        Ok(z)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        if !self.trainable {
            // If not trainable, return zero gradient tensor
            return Ok(Tensor::zeros(self.input_shape.clone().unwrap(), self.device.clone()));
        }

        let input = self.input.as_ref().ok_or(LayerError::MissingInput)?;
        let weights = self.weights.as_ref().ok_or(LayerError::UninitializedWeights)?;

        // Convert tensors to 4D arrays
        let grad_4d = grad.data
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;
        let input_4d = input.data
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;
        let weights_4d = weights.data
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;

        let (batch_size, in_channels, in_height, in_width) = input_4d.dim();
        let (out_channels, _, kernel_height, kernel_width) = weights_4d.dim();
        let channels_per_group = in_channels / self.groups;

        // Initialize gradients
        let mut dx = Array::zeros(input_4d.dim());
        let mut dw = Array::zeros(weights_4d.dim());
        let mut db = if self.include_bias {
            Some(Array::zeros((1, out_channels, 1, 1)))
        } else {
            None
        };

        // Apply padding to input if needed
        let padded_input = self.pad_input(&input_4d);

        // Calculate bias gradient if needed
        if let Some(db_mut) = db.as_mut() {
            for c in 0..out_channels {
                db_mut[[0, c, 0, 0]] = grad_4d.slice(s![.., c, .., ..]).sum();
            }
        }

        // Process each group separately
        for g in 0..self.groups {
            let in_start = g * channels_per_group;
            let in_end = (g + 1) * channels_per_group;
            let out_start = g * (out_channels / self.groups);
            let out_end = (g + 1) * (out_channels / self.groups);

            // Extract slices for current group
            let input_slice = padded_input.slice(s![.., in_start..in_end, .., ..]);
            let grad_slice = grad_4d.slice(s![.., out_start..out_end, .., ..]);
            let weights_slice = weights_4d.slice(s![out_start..out_end, .., .., ..]);

            // Calculate weights gradient
            for b in 0..batch_size {
                for oc in 0..out_channels / self.groups {
                    for ic in 0..channels_per_group {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let mut grad_sum = 0.0;

                                // Sum over output spatial dimensions
                                for oh in 0..grad_slice.shape()[2] {
                                    for ow in 0..grad_slice.shape()[3] {
                                        let ih = oh * self.stride.0 + kh * self.dilation.0;
                                        let iw = ow * self.stride.1 + kw * self.dilation.1;

                                        if ih < in_height && iw < in_width {
                                            grad_sum +=
                                                grad_slice[[b, oc, oh, ow]] *
                                                input_slice[[b, ic, ih, iw]];
                                        }
                                    }
                                }

                                dw[[out_start + oc, ic, kh, kw]] += grad_sum;
                            }
                        }
                    }
                }
            }

            // Calculate input gradient
            for b in 0..batch_size {
                for ic in 0..channels_per_group {
                    for ih in 0..in_height {
                        for iw in 0..in_width {
                            let mut grad_sum = 0.0;

                            // Sum over output channels and kernel dimensions
                            for oc in 0..out_channels / self.groups {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let oh =
                                            ((ih as i32) - (kh as i32) * (self.dilation.0 as i32)) /
                                            (self.stride.0 as i32);
                                        let ow =
                                            ((iw as i32) - (kw as i32) * (self.dilation.1 as i32)) /
                                            (self.stride.1 as i32);

                                        if
                                            oh >= 0 &&
                                            ow >= 0 &&
                                            oh < (grad_slice.shape()[2] as i32) &&
                                            ow < (grad_slice.shape()[3] as i32)
                                        {
                                            grad_sum +=
                                                grad_slice[[b, oc, oh as usize, ow as usize]] *
                                                weights_slice[[oc, ic, kh, kw]];
                                        }
                                    }
                                }
                            }

                            dx[[b, in_start + ic, ih, iw]] = grad_sum;
                        }
                    }
                }
            }
        }

        // Store gradients
        self.weights_grad = Some(Tensor {
            data: dw.into_dyn(),
            device: self.device.clone(),
        });

        if let Some(db) = db {
            self.bias_grad = Some(Tensor {
                data: db.into_dyn(),
                device: self.device.clone(),
            });
        }

        // Return input gradient
        Ok(Tensor {
            data: dx.into_dyn(),
            device: self.device.clone(),
        })
    }

    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        if !self.trainable {
            return Ok(());
        }

        if let Some(ref weights_grad) = self.weights_grad {
            optimizer
                .step(self.weights.as_mut().unwrap(), weights_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        if let Some(ref bias_grad) = self.bias_grad {
            if let Some(ref mut bias) = self.bias {
                optimizer.step(bias, bias_grad).map_err(LayerError::OptimizerError)?;
            }
        }

        self.weights_grad = None;
        self.bias_grad = None;

        Ok(())
    }

    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let raw = self.get_input_shape()?;
        let (batch_size, _, height, width) = (raw[0], raw[1], raw[2], raw[3]);

        let (pad_h, pad_w) = self.get_padding_sizes(height, width);
        let out_height =
            (height + 2 * pad_h - (self.kernel_size.0 - 1) * self.dilation.0 - 1) / self.stride.0 +
            1;
        let out_width =
            (width + 2 * pad_w - (self.kernel_size.1 - 1) * self.dilation.1 - 1) / self.stride.1 +
            1;

        Ok(Shape::from(IxDyn(&[batch_size, self.kernel_units, out_height, out_width])))
    }

    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        let weights_count = self.weights.as_ref().map_or(0, |w| w.data.len());
        let bias_count = self.bias.as_ref().map_or(0, |b| b.data.len());
        Ok((weights_count, bias_count))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();

        if let Some(ref mut weights) = self.weights {
            weights.device = device.clone();
        }
        if let Some(ref mut bias) = self.bias {
            bias.device = device.clone();
        }
        if let Some(ref mut input) = self.input {
            input.device = device.clone();
        }
    }

    fn get_weights(&self) -> serde_json::Value {
        serde_json::json!({
            "weights": self.weights.as_ref().map(|w| w.to_vec()),
            "bias": self.bias.as_ref().map(|b| b.to_vec())
        })
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "kernel_units": self.kernel_units,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
             "padding": serde_json::to_value(self.padding.clone()).unwrap(),
            "dilation": self.dilation,
            "groups": self.groups,
            "trainable": self.trainable,
            "include_bias": self.include_bias,
            "activation": self.activation.as_ref().map(|a| a.name())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::ReluActivation;
    use approx::assert_abs_diff_eq; // Import approx for floating-point comparisons

    #[test]
    fn test_conv2d_forward_pass_with_precision() {
        // Initialize Conv2D layer
        let mut conv = Conv2D::new(
            1,
            (1, 1),
            (2, 2),
            PaddingMode::Valid,
            (1, 1),
            1,
            None::<ReluActivation>,
            true,
            false
        ); // No padding, stride 1

        // Set weights and bias manually
        conv.weights = Some(
            Tensor::new(vec![0.5, -0.5, 1.0, -1.0], Shape::from(IxDyn(&[1, 1, 2, 2])))
        );
        conv.bias = Some(Tensor::new(vec![0.1], Shape::from(IxDyn(&[1, 1, 1, 1]))));

        // Input tensor
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::from(IxDyn(&[1, 1, 3, 3]))
        );

        // Expected output (calculated manually or with a trusted library)
        let expected_output = vec![
            vec![
                vec![0.1, 0.2], // Example values with potential rounding
                vec![0.3, 0.4]
            ]
        ];

        // Perform forward pass
        let output = conv.forward(&input).expect("Forward pass failed");

        // Compare each value with tolerance
        for (out_row, exp_row) in output.data.outer_iter().zip(expected_output.iter()) {
            for (out_val, exp_val) in out_row.iter().zip(exp_row.iter()) {
                for (out_inner, exp_inner) in exp_val.iter().zip(exp_val.iter()) {
                    assert_abs_diff_eq!(out_inner, exp_inner, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_padding_with_precision() {
        // Initialize Conv2D with padding
        let mut conv = Conv2D::new(
            1,
            (1, 1),
            (2, 2),
            PaddingMode::Valid,
            (1, 1),
            1,
            None::<ReluActivation>,
            true,
            false
        );

        // Input tensor
        let input = vec![vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]];

        // Expected padded input
        let expected_padded = vec![
            vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 2.0, 3.0, 0.0],
                vec![0.0, 4.0, 5.0, 6.0, 0.0],
                vec![0.0, 7.0, 8.0, 9.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ];

        // Perform padding logic
        let input_array = Array::from_shape_vec(
            (1, 1, 3, 3),
            input.into_iter().flatten().flatten().collect()
        ).unwrap();
        let padded_input = conv.pad_input(&input_array);

        // Compare each value in the padded matrix
        for (padded_row, exp_row) in padded_input.iter().zip(expected_padded.iter()) {
            for (padded_val, exp_val) in exp_row.iter().zip(exp_row.iter()) {
                for (padded_inner, exp_inner) in padded_val.iter().zip(exp_val.iter()) {
                    assert_abs_diff_eq!(padded_inner, exp_inner, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_conv2d_with_stride() {
        let input_shape = Shape::from(IxDyn(&[1, 1, 5, 5]));
        let mut conv2d = Conv2D::new(
            1, // kernel_units
            (2, 2), // kernel_size
            (2, 2), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            false // no bias
        );

        conv2d.build(input_shape.clone()).expect("Failed to build layer");

        // Check output shape with stride
        let output_shape = conv2d.output_shape().expect("Failed to compute output shape");
        assert_eq!(output_shape.raw_dim()[2], 2); // (5-2)/2 + 1 = 2
        assert_eq!(output_shape.raw_dim()[3], 2);
    }

    #[test]
    fn test_conv2d_with_custom_padding() {
        let input_shape = Shape::from(IxDyn(&[1, 1, 3, 3]));
        let mut conv2d = Conv2D::new(
            1, // kernel_units
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Custom(1, 1), // padding
            (1, 1), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            false // no bias
        );
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            input_shape.clone()
        );

        conv2d.build(input_shape).expect("Failed to build layer");
        let output = conv2d.forward(&input).expect("Forward pass failed");
        let output_shape = conv2d.output_shape().expect("Failed to compute output shape");
        // Check output shape

        assert_eq!(output_shape.raw_dim()[2], 4);
        assert_eq!(output_shape.raw_dim()[3], 4);
    }

    #[test]
    fn test_conv2d_grouped() {
        let input_shape = Shape::from(IxDyn(&[1, 4, 4, 4])); // 4 input channels
        let mut conv2d = Conv2D::new(
            4, // 4 output channels
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            2, // 2 groups
            None::<ReluActivation>,
            true,
            false // no bias
        );
        conv2d.build(input_shape.clone()).expect("Failed to build layer");

        // Verify weight shape for grouped convolution
        assert_eq!(
            conv2d.weights.as_ref().unwrap().data.shape(),
            &[4, 2, 2, 2] // [out_channels, in_channels/groups, kernel_h, kernel_w]
        );
    }

    #[test]
    fn test_conv2d_grouped_with_bias() {
        let input_shape = Shape::from(IxDyn(&[1, 4, 4, 4])); // 4 input channels
        let mut conv2d = Conv2D::new(
            4, // 4 output channels
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            2, // 2 groups
            None::<ReluActivation>,
            true,
            true
        );
        conv2d.build(input_shape.clone()).expect("Failed to build layer");

        // Verify weight shape for grouped convolution
        assert_eq!(
            conv2d.weights.as_ref().unwrap().data.shape(),
            &[4, 2, 2, 2] // [out_channels, in_channels/groups, kernel_h, kernel_w]
        );
    }

    #[test]
    fn test_conv2d_backward() {
        let input_shape = Shape::from(IxDyn(&[1, 1, 3, 3]));
        let mut conv2d = Conv2D::new(
            1, // kernel_units
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            true // include bias
        );

        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            input_shape.clone()
        );
        conv2d.build(input_shape).expect("Failed to build layer");

        // Forward pass
        let output = conv2d.forward(&input).expect("Forward pass failed");

        // Create gradient tensor (assuming gradients of 1.0)
        let grad = Tensor::ones(output.data.dim().into(), Device::default());

        // Backward pass
        let input_grad = conv2d.backward(&grad).expect("Backward pass failed");

        // Verify gradients were computed
        assert!(conv2d.weights_grad.is_some());
        assert!(conv2d.bias_grad.is_some());
        assert_eq!(input_grad.data.shape(), input.data.shape());
    }
    #[test]
    fn test_conv2d_with_dilation() {
        let input_shape = Shape::from(IxDyn(&[1, 1, 5, 5]));
        let mut conv2d = Conv2D::new(
            1, // kernel_units
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (2, 2), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            false // no bias
        );

        conv2d.build(input_shape.clone()).expect("Failed to build layer");

        // Check output shape with dilation
        let output_shape = conv2d.output_shape().expect("Failed to compute output shape");
        assert_eq!(output_shape.raw_dim()[2], 3);
        assert_eq!(output_shape.raw_dim()[3], 3);
    }

    #[test]
    fn test_conv2d_update_weights() {
        let input_shape = Shape::from(IxDyn(&[1, 1, 3, 3]));
        let mut conv2d = Conv2D::new(
            1, // kernel_units
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            true // include bias
        );

        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            input_shape.clone()
        );

        conv2d.build(input_shape).expect("Failed to build layer");
        let initial_weights = conv2d.weights.clone().unwrap();

        // Forward pass
        let output = conv2d.forward(&input).expect("Forward pass failed");

        // Create gradient tensor (assuming gradients of 1.0)
        let grad = Tensor::ones(output.data.dim().into(), Device::default());

        // Backward pass
        conv2d.backward(&grad).expect("Backward pass failed");

        let mut optimizer: Box<dyn Optimizer> = Box::new(crate::optimizers::Adam::new(1.0));
        conv2d.update_weights(&mut optimizer).unwrap();

        let updated_weights = conv2d.weights.unwrap();

        assert_ne!(initial_weights.data, updated_weights.data, "Weights should be updated");
    }

    #[test]
    fn test_conv2d_bias_update() {
        let input_shape = Shape::from(IxDyn(&[1, 1, 3, 3]));
        let mut conv2d = Conv2D::new(
            1, // kernel_units
            (2, 2), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            true // include bias
        );

        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            input_shape.clone()
        );
        conv2d.build(input_shape).expect("Failed to build layer");
        let initial_bias = conv2d.bias.clone().unwrap();

        // Forward pass
        let output = conv2d.forward(&input).expect("Forward pass failed");

        // Create gradient tensor (assuming gradients of 1.0)
        let grad = Tensor::ones(output.data.dim().into(), Device::default());

        // Backward pass
        conv2d.backward(&grad).expect("Backward pass failed");

        let mut optimizer: Box<dyn Optimizer> = Box::new(crate::optimizers::Adam::new(1.0));
        conv2d.update_weights(&mut optimizer).unwrap();

        let updated_bias = conv2d.bias.unwrap();
        assert_ne!(initial_bias.data, updated_bias.data, "Bias should be updated");
    }
    #[test]
    fn test_conv2d_param_count() {
        let input_shape = Shape::from(IxDyn(&[1, 3, 28, 28]));
        let mut conv2d = Conv2D::new(
            2, // kernel_units
            (3, 3), // kernel_size
            (1, 1), // stride
            PaddingMode::Valid, // padding
            (1, 1), // dilation
            1, // groups
            None::<ReluActivation>,
            true,
            true
        );
        conv2d.build(input_shape).unwrap();
        let (weights_count, bias_count) = conv2d.param_count().unwrap();
        assert_eq!(weights_count, 2 * 3 * 3 * 3);
        assert_eq!(bias_count, 2 * 1);
    }
}
