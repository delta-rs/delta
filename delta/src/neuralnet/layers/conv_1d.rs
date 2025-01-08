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

use log::debug;
use ndarray::{s, Axis, Dim, Dimension, IxDyn, IxDynImpl, Shape};
use serde_json;

use crate::activations::Activation;
use crate::common::Tensor;
use crate::devices::Device;
use crate::neuralnet::layers::Layer;
use crate::neuralnet::layers::error::LayerError;
use crate::optimizers::Optimizer;

// used for operations on Tensors
use crate::neuralnet::functional as F;

// type ArrayView<'a> = ndarray::ArrayBase<ndarray::ViewRepr<&'a f32>, Dim<[usize; 1]>>;

/// A 1D convolutional layer.
#[derive(Debug)]
pub struct Conv1D {
    name: String,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    kernel_units: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    include_bias: bool,
    activation: Option<Box<dyn Activation>>,
    trainable: bool,
    weights_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    input: Option<Tensor>,
    input_shape: Option<Shape<IxDyn>>,
    device: Device,
}

impl Conv1D {

    /// Creates a new dense layer.
    ///
    /// # Arguments
    ///
    /// * `kernel_units` - The number of output kernels.
    /// * `kernel_size` - The shape of the output kernels.
    /// * `stride` - Controls the stride for cross-correlation operation.
    /// * `activation` - The activation function to use.
    /// * `trainable` - Whether the layer is trainable.
    pub fn new<A: Activation + 'static>( //IMP: since it's dynamic dispatch and hence the type A could be a reference as well hence compiler needs to make sure that the value has a static lifetime!, even though box own the value (cause the value could be a reference!)
        kernel_units: usize,
        kernel_size: usize,
        stride: usize,
        activation: Option<A>,
        trainable: bool,
        include_bias: bool,
    ) -> Self {
        Conv1D {
            name: format!("conv1d_{}", kernel_units),
            weights: None,
            bias: None,
            kernel_units,
            kernel_size,
            stride,     
            padding: 0_usize, // keeping the value 0 as usize for now!
            dilation: 1_usize, // keeping the value 1 as usize for now!
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

    /// Returns the input shape of the layer.
    /// 
    /// # Returns
    /// A `Dim<IxDynImpl>` representing the input shape of the layer.
    /// 
    /// # Errors
    /// Returns a `LayerError` if the input shape is not valid.
    fn get_input_shape(&self) -> Result<Dim<IxDynImpl>, LayerError> {

        let shape = self.input_shape.as_ref().ok_or(LayerError::MissingInput)?;
        let raw = shape.raw_dim(); // IxDyn
        if raw.ndim() != 3 {
            return Err(LayerError::InvalidInputShape);
        }
        Ok(raw.clone())

    }
}

impl Layer for Conv1D {

    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor, expected in `[in_channels, * ]`
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        debug!(
            "Building Conv1D layer with input shape: {:?}, kernel_units: {}, kernel_size: {} and stride: {}",
            input_shape, self.kernel_units, self.kernel_size, self.stride
        );

        // Store input shape for later use
        self.input_shape = Some(input_shape);

        // Get input dimensions
        let raw = self.get_input_shape()?;
        let (_, input_kernel_units, input_units) = (raw[0], raw[1], raw[2]);

        // Choose initialization strategy based on the activation function
        let stddev = if let Some(ref activation) = self.activation {
            match activation.name() {
                "relu" | "leaky_relu" => (2.0 / input_units as f32).sqrt(), // He initialization
                _ => (1.0 / input_units as f32).sqrt(), // Xavier initialization
            }
        } else {
            (1.0 / input_units as f32).sqrt() // Xavier initialization for no activation
        };

        // Initialize weights using random normal distribution
        self.weights = Some(Tensor::random_normal(
            Shape::from(IxDyn(&[self.kernel_units, input_kernel_units, self.kernel_size])), // [out_channels, in_channels, kernel_size]
            0.0,
            stddev,
        ));

        #[cfg(debug_assertions)] // for debugging purposes
        {
            self.weights = Some(Tensor::ones(
                Shape::from(IxDyn(&[self.kernel_units, input_kernel_units, self.kernel_size])),
                self.device.clone()
            ));
        }

        // Set device for weights
        self.weights.as_mut().expect("Weight not initialized properly!").device = self.device.clone();

        // Initialize bias to zeros
        self.bias = if self.include_bias { Some(Tensor::zeros(
            Shape::from(IxDyn(&[self.kernel_units])),
            self.device.clone())
        )} else { None };

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

        // Save input for backward pass
        self.input = Some(input.clone());
        
        // Apply convolution operation
        let z = F::conv1d(
            input,
            self.weights.as_ref().unwrap(),
            Some(self.bias.as_ref().unwrap()),
            self.stride,
            self.padding, // 0 for now!
            self.dilation // 1 for now!
        ).unwrap();

        // Apply activation if present
        let z =
            if let Some(ref activation) = self.activation { activation.activate(&z) } else { z };

        Ok(z)
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

        // Get weights shape
        // let raw = self.get_input_shape()?;
        // let (batch_size, c_out, c_in, kernel_size) = (raw[0], self.kernel_units, raw[1],self.kernel_size);

        // Get permuted weight
        // let w_perm = self.weights.as_ref().unwrap().data.view().permuted_axes(Dim(IxDyn(&[1, 0, 2])));
        let mut w_perm = self.weights.as_ref().unwrap().permute(vec![1, 0, 2]);
        // Flip the kernel
        w_perm.data.slice_mut(s![.., .., ..;-1]);
        debug!("w_perm: {:?}", w_perm);
        
        // Get permuted input
        let input_perm = self.input.as_ref().unwrap().permute(vec![1, 0, 2]);
        debug!("input_perm: {:?}", input_perm);

        // Get permuted gradient
        let grad_perm = grad.permute(vec![1, 0, 2]);
        debug!("grad_perm: {:?}", grad_perm);
        
        let grad_padded = F::pad_1d(grad, self.kernel_size-1, Axis(2));
        
        let input_grad = F::conv1d(
            grad_padded.as_ref().unwrap(),
            &w_perm,
            None,
            self.stride,
            self.padding, // 0 for now!
            self.dilation // 1 for now!
        ).unwrap();
        debug!("input_grad: {:?}", input_grad);

        let weight_grad = F::conv1d(
            &input_perm,
            &grad_perm,
            None,
            self.stride,
            self.padding, // 0 for now!
            self.dilation // 1 for now!
        ).unwrap();
        debug!("weight_grad: {:?}", weight_grad);

        // Store the gradient 
        if self.trainable {
            self.weights_grad = Some(weight_grad.permute(vec![1, 0, 2]));
            // self.bias_grad = 
        }

        Ok(grad.clone())
    }

    /// Updates the weights of the layer using the given gradient and optimizer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        if !self.trainable {
            return Ok(());
        }

        // Update weights
        if let Some(ref weights_grad) = self.weights_grad {
            optimizer
                .step(self.weights.as_mut().unwrap(), weights_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        if let Some(ref bias_grad) = self.bias_grad {
            optimizer
                .step(self.bias.as_mut().unwrap(), bias_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        // Clear gradients after update
        self.weights_grad = None;
        self.bias_grad = None;

        Ok(())
    }
    
    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let raw = self.get_input_shape()?;
        let (n, _, l) = (raw[0], raw[1], raw[2]);
        let output_units = (l - self.kernel_size) / self.stride + 1;
        let shape = Shape::from(IxDyn(&[n, self.kernel_units, output_units]));
        Ok(shape)
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of parameters in the layer.
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        let weights_count = self.weights.as_ref().map_or(0, |w| w.data.len());
        let bias_count = self.bias.as_ref().map_or(0, |b| b.data.len());
        Ok((weights_count, bias_count))
    }

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str {
        &self.name
    }

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
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

    // For saving the weights of the layer

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
            "trainable": self.trainable,
            "activation": self.activation.as_ref().map(|a| a.name())
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::ReluActivation;

    use super::*;

    #[test]
    fn test_conv1d_name() {
        let conv1d_layer = Conv1D::new(
            2,
            2,
            1,
            None::<ReluActivation>,
            false,
            false,
        );
        println!("{}", conv1d_layer.name());

        // checking the dim trait
        let input_shape = Shape::from(IxDyn(&[1, 3]));
        let raw_dim = input_shape.raw_dim();
        let array_view = raw_dim.as_array_view();
        let input_kernel_units = array_view.first().ok_or(LayerError::InvalidInputShape);

        println!("raw-dim: {:?}, array_view: {}, input_units: {:?}", raw_dim, array_view, input_kernel_units);
    }
    #[test]
    fn test_conv1d_build() {
        let input_shape = Shape::from(IxDyn(&[1, 2, 5]));
        let mut conv1d_layer = Conv1D::new(
            2,
            2,
            1,
            None::<ReluActivation>,
            false,
            false
        );
        conv1d_layer.build(input_shape).expect("Failed to build layer");
        println!("weights: {:?}", conv1d_layer.weights);
        println!("bias: {:?}", conv1d_layer.bias);
    }
    #[test]
    fn test_conv1d_param_count_with_bias() {
        let input_shape = Shape::from(IxDyn(&[1, 2, 5]));
        let mut conv1d_layer = Conv1D::new(
            2,
            2,
            1,
            None::<ReluActivation>,
            false,
            true
        );
        conv1d_layer.build(input_shape).expect("Failed to build layer");
        assert_eq!(conv1d_layer.param_count().unwrap(), (8, 2));
    }
    #[test]
    fn test_conv1d_param_count_without_bias() {
        let input_shape = Shape::from(IxDyn(&[1, 2, 5]));
        let mut conv1d_layer = Conv1D::new(
            2,
            2,
            1,
            None::<ReluActivation>,
            false,
            false
        );
        // let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], input_shape.clone());
        conv1d_layer.build(input_shape).expect("Failed to build layer");
        assert_eq!(conv1d_layer.param_count().unwrap(), (8, 0));
    }
    #[cfg(debug_assertions)]
    #[test]
    fn test_conv1d_forward_with_bias() {
        let input_shape = Shape::from(IxDyn(&[1, 2, 5]));
        let mut conv1d_layer = Conv1D::new(
            3,
            2,
            1,
            None::<ReluActivation>,
            false,
            true
        );
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], input_shape.clone());
        conv1d_layer.build(input_shape).expect("Failed to build layer");

        let out = conv1d_layer.forward(&input).unwrap();
        println!("Output: {:?}", out);
    }
    #[test]
    fn test_conv1d_backward() {
        let input_shape = Shape::from(IxDyn(&[1, 2, 5]));
        let mut conv1d_layer = Conv1D::new(
            3,
            2,
            1,
            None::<ReluActivation>, 
            false,
            true
        );
        let input = Tensor::new((1..=10).map(|x| x as f32).collect() , input_shape.clone());
        conv1d_layer.build(input_shape.clone()).expect("Failed to build layer");
        
        let out = conv1d_layer.forward(&input).unwrap();
        println!("Output: {:?}", out);
        let grad = Tensor::new((1..=12).map(|x| x as f32).collect(), out.shape());
        
        // running backward
        let next_grad = conv1d_layer.backward(&grad).unwrap();
        println!("Next grad: {:?}", next_grad);
    }


}


// Get input
// let input_3d = self.input.as_ref().expect("Input must be initialized")
//             .data
//             .clone()
//             .into_dimensionality::<IxDyn>()
//             .map_err(|_| LayerError::InvalidInputShape)?;

// initialize the weights gradient
// let mut w_grad = Array::<f32, Ix4>::zeros((batch_size, c_out, c_in, kernel_size));

// get the gradient of the weights
// for b in 0..batch_size {
//     for o in 0..c_out {
//         for i in 0..c_in {
//             let grad_slice = grad.data.slice(s![b, o, ..]);
//             let input_slice = input_3d.slice(s![b, i, ..]);
//             let w_grad_slice = conv1d_raw(input_slice, grad_slice, self.stride);

//             for x in 0..kernel_size {
//                 w_grad[[b, c_out, c_in, x]] = w_grad_slice[x];
//             }
//         }
//     }
// }