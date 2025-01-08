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

use ndarray::{s, Array, ArrayView, Axis, Ix1, IxDyn, Shape};
use rand::Error;

use crate::common::{CoreError, Tensor};


/// Pads a 1D input tensor with zeros on both sides.
/// 
/// # Arguments
/// * `input` - Input tensor of shape (in_length)
/// * `padding` - Number of zeros to pad on each side
/// 
/// # Returns
/// * Padded tensor of shape (in_length + 2 * padding)
pub fn pad1d_raw(input: &ArrayView<f32, Ix1>, padding: usize) -> Array<f32, Ix1> {
    let input_len = input.len();
    let padded_len = input_len + 2 * padding;
    let mut padded_input = Array::zeros(padded_len);
    padded_input.slice_mut(s![padding..padding + input_len])
        .assign(&input);
    
    padded_input
}

/// Pads a 3D input tensor with zeros on both sides of a 1D axis
/// 
/// # Arguments
/// * `input` - Input tensor of shape (batch_size, in_channels, in_length)
/// * `padding` - Number of zeros to pad on each side
/// 
/// # Returns
/// Padded tensor of shape (batch_size, in_channels, in_length + 2 * padding)
/// 
pub fn pad_1d(input: &Tensor, padding: usize, axis: Axis) -> Result<Tensor, Error> {

    let shape = input.shape().raw_dim().clone();

    let mut padded_shape = shape.clone();
    match axis {
        Axis(0) => padded_shape[0] += 2 * padding,
        Axis(1) => padded_shape[1] += 2 * padding,
        Axis(2) => padded_shape[2] += 2 * padding,
        _ => panic!("Unsupported axis for padding"),
    }

    let mut expanded = Tensor::zeros(
        Shape::from(padded_shape),
        input.device.clone()
    );

    match axis {
        Axis(0) => expanded.data.slice_mut(s![padding..padding + shape[0], .., ..]).assign(&input.data),
        Axis(1) => expanded.data.slice_mut(s![.., padding..padding + shape[1], ..]).assign(&input.data),
        Axis(2) => expanded.data.slice_mut(s![.., .., padding..padding + shape[2]]).assign(&input.data),
        _ => panic!("Unsupported axis for padding"),
    }

    Ok(expanded)
}

/// 1D Convolutional for batched data with weights and biases.
/// 
/// # Arguments
/// 
/// * `inputs` - Input tensor of shape (batch_size, in_channels, in_length)
/// * `weight` - Weight tensor of shape (out_channels, in_channels, kernel_size)
/// * `bias` - Bias tensor of shape (out_channels)
/// * `stride` - Stride of the convolution
/// 
/// # Returns
/// 
/// * Output tensor of shape (batch_size, out_channels, out_length)
///  
pub fn conv1d(inputs: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize, dilation: usize) -> Result<Tensor, CoreError> {

    
    // Get input dimensions
    let raw_input_dim = inputs.shape().raw_dim().clone();
    let (batch_size, in_ch, in_len) = (raw_input_dim[0], raw_input_dim[1], raw_input_dim[2]);
    
    // Get weight dimensions
    let raw_weight_dim = weight.shape().raw_dim().clone();
    let (out_ch, in_ch_w, kernel_size) = (raw_weight_dim[0], raw_weight_dim[1], raw_weight_dim[2]);
    
    if in_ch != in_ch_w {
        return Err(CoreError::InvalidShape);
    } 
    
    let bias = match bias {
        Some(b) => b,
        None => &Tensor::zeros(Shape::from(IxDyn(&[out_ch])), inputs.device.clone()),
    };

    let dilated_kernel_len = (kernel_size - 1) * dilation + 1;
    let out_len = (in_len + 2 * padding - dilated_kernel_len) / stride + 1;

    let out_shape = Shape::from(IxDyn(&[batch_size, out_ch, out_len]));
    let mut output = Tensor::zeros(out_shape, inputs.device.clone());
    
    for batch in 0..batch_size {
        for o in 0..out_ch {
            output.data.slice_mut(s![batch, o, ..]).assign(
                {
                    let mut tmp = Array::zeros(
                        Shape::from(IxDyn(&[out_len])),
                    );
                    for i in 0..in_ch {
                        let kernel_slice = weight.data.slice(s![o, i, ..]);
                        let input_slice = inputs.data.slice(s![batch, i, ..]);
                        tmp += &conv1d_raw(&input_slice, &kernel_slice, stride, padding, dilation);
                    }
                        
                    &(tmp + bias.data[[o]]).clone()
                }
            );
        }
    }
    
    Ok(output)
}

/// Convolution operation for two 1D vectors.
/// 
/// # Arguments
/// 
/// * `input` - The input tensor.
/// * `kernel` - The kernel tensor.
/// * `stride` - The stride for the convolution operation.
/// 
/// # Returns
/// 
/// The output tensor.
/// 
pub fn conv1d_raw(input: &ArrayView<f32, Ix1>, kernel: &ArrayView<f32, Ix1>, stride: usize, padding: usize, dilation: usize) -> Array<f32, Ix1> {
    let input_len = input.len();
    let kernel_len = kernel.len();
    let dilated_kernel_len = (kernel_len - 1) * dilation + 1;
    
    // Calculate output length using the formula:
    let output_len = (input_len + 2 * padding - dilated_kernel_len) / stride + 1;
    
    // Create output array filled with zeros
    let mut output = Array::zeros(output_len);
    
    // Create padded input if necessary
    let padded_len = input_len + 2 * padding;
    let mut padded_input = Array::zeros(padded_len);
    padded_input.slice_mut(s![padding..padding + input_len])
        .assign(&input);
    
    // Perform convolution
    for out_idx in 0..output_len {
        let input_start_idx = out_idx * stride;
        let mut sum = 0.0;
        
        for (k_idx, &k_val) in kernel.iter().enumerate() {
            let in_idx = input_start_idx + k_idx * dilation;
            if in_idx < padded_len {
                sum += padded_input[in_idx] * k_val;
            }
        }
        
        output[out_idx] = sum;
    }
    
    output
}
    



#[cfg(test)]
mod tests {
    use std::f32;

    use crate::{common::Tensor, devices::Device, neuralnet::functional as F};
    use ndarray::{arr3, array, Array, ArrayD, Axis, IxDyn, Shape};

    #[test]
    fn test_conv1d_raw() {

        let input = Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let kernel = Array::from(vec![1.0, 0.5]);
        let stride = 1;
        let padding = 1;
        let dilation = 2;

        let output = F::conv1d_raw(&input.view(), &kernel.view(), stride, padding, dilation);
        assert_eq!(output, Array::from(vec![1.0, 2.5, 4.0, 5.5, 4.0]));
    }

    #[test]
    fn test_conv1d() {

        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::from(IxDyn(&[1, 1, 5])), // 1 batch, 1 channel, 5 length
        );
        let weight = Tensor::ones(
            Shape::from(IxDyn(&[3, 1, 2])), // 3 output channels, 1 input channel, 2 kernel size
            Device::default(),
        );
        let bias = Tensor::ones(
            Shape::from(IxDyn(&[3])), // 3 output channels
            Device::default(),
        );

        let stride = 1;
        let padding = 0;
        let dilation = 1;

        let output = F::conv1d(
            &input,
            &weight,
            Some(&bias),
            stride,
            padding,
            dilation,
        ).unwrap();

        assert_eq!(output, Tensor::new(
            vec![4.0, 6.0, 8.0, 10.0,
                 4.0, 6.0, 8.0, 10.0,
                 4.0, 6.0, 8.0, 10.0 ],
            Shape::from(IxDyn(&[1, 3, 4])), // 1 batch, 3 output channels, 4 length
        ));
        
    }

    #[test]
    fn test_pad1d_raw() {
        
        let input = Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let padding = 2;
        let padded_input = F::pad1d_raw(&input.view(), padding);
        assert_eq!(padded_input, Array::from(vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]));

    }

    #[test]
    fn test_pad_1d() {
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from(IxDyn(&[2, 2, 2])), // 1 batch, 1 channel, 5 length
        );
        let padding = 1;
        let axis = Axis(2);
        let expanded = F::pad_1d(&input, padding, axis).unwrap();
        // expanded.slice_mut(s![.., 1..5]).assign(&input.data);

        let a = array! [[[0.0, 1.0, 2.0, 0.0],
            [ 0.0, 3.0, 4.0, 0.0]],    
           [[ 0.0, 5.0, 6.0, 0.0],     
            [0.0, 7.0, 8.0, 0.0]]];
        let a_dynamic: ArrayD<f32> = a.into_dyn();

        assert_eq!(expanded.data, a_dynamic);
    }
}