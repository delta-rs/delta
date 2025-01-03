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
    Array, Array4, Axis, Ix4, IxDyn, Shape, s, Dimension
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
    /// The shape of the input tensor (expected 4D: [N, C, H, W]).
    input_shape: Option<Shape<IxDyn>>,
    /// Stores the positions of the maximum values from the forward pass.
    /// Array shape is [N, C, outH, outW], each element is `(row, col)` of
    /// where the max was found within its pooling window.
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
    /// * `input_shape` - The shape of the input tensor (must be 4D).
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        // Convert to IxDyn
        let raw = input_shape.raw_dim(); // IxDyn
        if raw.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        if self.pool_size == 0 || self.stride == 0 {
            return Err(LayerError::InvalidInputShape);
        }
        Ok(())
    }

    /// Performs the forward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor, shape [N, C, H, W].
    ///
    /// # Returns
    ///
    /// A `Result` containing the output tensor or an error.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Make sure we have a recorded input shape.
        let shape = self.input_shape.as_ref().ok_or(LayerError::MissingInput)?;
        let raw = shape.raw_dim(); // IxDyn
        if raw.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }

        // Extract (N, C, H, W)
        let (n, c, h, w) = (raw[0], raw[1], raw[2], raw[3]);

        // Reshape input data to 4D array [N, C, H, W].
        let input_4d = input.data.clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;

        // Compute the output height/width.
        if self.pool_size > h || self.pool_size > w {
            return Err(LayerError::InvalidInputShape);
        }
        let out_h = (h - self.pool_size) / self.stride + 1;
        let out_w = (w - self.pool_size) / self.stride + 1;

        // Prepare output array [N, C, outH, outW].
        let mut output = Array::<f32, Ix4>::zeros((n, c, out_h, out_w));
        // Prepare max_indices array using default values ((0,0) is the default).
        let mut max_positions = Array::<(usize, usize), Ix4>::default((n, c, out_h, out_w));

        // Perform max pooling.
        for batch in 0..n {
            for ch in 0..c {
                for out_i in 0..out_h {
                    for out_j in 0..out_w {
                        let start_i = out_i * self.stride;
                        let start_j = out_j * self.stride;
                        let end_i = start_i + self.pool_size;
                        let end_j = start_j + self.pool_size;

                        // Slice the window [start_i..end_i, start_j..end_j].
                        let window = input_4d.slice(s![batch, ch, start_i..end_i, start_j..end_j]);

                        let mut max_val = f32::NEG_INFINITY;
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
                        // Store absolute position in the original input:
                        max_positions[[batch, ch, out_i, out_j]] = (start_i + max_pos.0, start_j + max_pos.1);
                    }
                }
            }
        }

        // Save indices for backward pass.
        self.max_indices = Some(max_positions);

        // Convert output to a dynamic shape tensor.
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
    /// * `grad` - The gradient tensor [N, C, outH, outW].
    ///
    /// # Returns
    ///
    /// A `Result` containing the gradient with respect to the input ([N, C, H, W]).
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let shape = self.input_shape.as_ref().ok_or(LayerError::MissingInput)?;
        let raw = shape.raw_dim(); // IxDyn
        if raw.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        let (n, c, h, w) = (raw[0], raw[1], raw[2], raw[3]);

        // Convert grad to 4D.
        let grad_4d = grad.data.clone()
            .into_dimensionality::<Ix4>()
            .map_err(|_| LayerError::InvalidInputShape)?;

        // Retrieve saved max indices.
        let max_indices = self.max_indices.as_ref().ok_or(LayerError::MissingInput)?;

        // Prepare dX with zeros.
        let mut dx = Array::<f32, Ix4>::zeros((n, c, h, w));

        // Scatter gradients back to the positions of maxima.
        let out_h = grad_4d.shape()[2];
        let out_w = grad_4d.shape()[3];
        for batch in 0..n {
            for ch in 0..c {
                for out_i in 0..out_h {
                    for out_j in 0..out_w {
                        let (src_i, src_j) = max_indices[[batch, ch, out_i, out_j]];
                        dx[[batch, ch, src_i, src_j]] += grad_4d[[batch, ch, out_i, out_j]];
                    }
                }
            }
        }

        // Return gradient w.r.t. the input.
        Ok(Tensor {
            data: dx.into_dyn(),
            device: grad.device.clone(),
        })
    }

    /// Returns the shape of the output tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing `[N, C, outH, outW]` or an error.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let shape = self.input_shape.as_ref().ok_or(LayerError::MissingInput)?;
        let raw = shape.raw_dim();
        if raw.ndim() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        let (n, c, h, w) = (raw[0], raw[1], raw[2], raw[3]);

        if self.pool_size > h || self.pool_size > w {
            return Err(LayerError::InvalidInputShape);
        }

        let out_h = (h - self.pool_size) / self.stride + 1;
        let out_w = (w - self.pool_size) / self.stride + 1;

        let out_shape = IxDyn(&[n, c, out_h, out_w]);
        Ok(Shape::from(out_shape))
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// `(0, 0)` for max pooling (no trainable parameters).
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        Ok((0, 0))
    }

    /// Returns the name of the layer.
    fn name(&self) -> &str {
        "MaxPooling2D"
    }

    /// Updates the weights of the layer using the given optimizer.
    ///
    /// # Returns
    ///
    /// Always `Ok(())` since max pooling has no weights.
    fn update_weights(&mut self, _optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        Ok(())
    }

    /// Sets the device for the layer. No-op in this case.
    fn set_device(&mut self, _device: &crate::devices::Device) {
        // No device-specific resources for a pure CPU pooling layer.
    }
}

// Below is the test module, which you can keep in the same file or move into a separate test file.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{IxDyn, Shape};
    use crate::common::tensor_ops::Tensor;
    use crate::neuralnet::layers::{Layer, error::LayerError};

    #[test]
    fn test_build_valid_shape() {
        let mut layer = MaxPooling2D::new(2, 2);
        // Valid 4D shape: [N, C, H, W].
        let shape = Shape::from(IxDyn(&[1, 3, 8, 8]));
        let result = layer.build(shape);
        assert!(result.is_ok(), "Building with a valid 4D shape should succeed.");
    }

    #[test]
    fn test_build_invalid_shape() {
        let mut layer = MaxPooling2D::new(2, 2);
        // Invalid shape (only 3D).
        let shape = Shape::from(IxDyn(&[1, 8, 8]));
        let result = layer.build(shape);
        assert!(result.is_err(), "Building with a 3D shape should fail.");
    }

    #[test]
    fn test_build_too_small_for_pool_size() {
        let mut layer = MaxPooling2D::new(5, 2); 
        let shape = Shape::from(IxDyn(&[1, 1, 4, 4])); // H=4, W=4, but pool_size=5.
        layer.build(shape).unwrap();
        
        // Querying output shape should fail because pool_size > input height/width.
        let out_shape_result = layer.output_shape();
        assert!(out_shape_result.is_err(), "Pool size exceeding input dims should produce error.");
    }

    #[test]
    fn test_param_count() {
        let mut layer = MaxPooling2D::new(2, 2);
        let shape = Shape::from(IxDyn(&[1, 1, 4, 4]));
        layer.build(shape).unwrap();

        let count = layer.param_count().unwrap();
        // Should return (0, 0) for no parameters.
        assert_eq!(count, (0, 0));
    }

    #[test]
    fn test_forward_basic() {
        let mut layer = MaxPooling2D::new(2, 2);
        let shape = Shape::from(IxDyn(&[1, 1, 4, 4])); // 1 batch, 1 channel, 4x4.
        layer.build(shape.clone()).unwrap();

        // Input data: 4x4 = 16 values from 1..16.
        let data = vec![
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0,  10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::new(data, shape);

        let output = layer.forward(&input).unwrap();
        // Expected a 2x2 output (pool_size=2, stride=2).
        // 2x2 windows => max values: [6, 8, 14, 16].
        let expected = vec![6.0, 8.0, 14.0, 16.0];

        assert_eq!(output.shape().raw_dim().slice(), &[1, 1, 2, 2]);
        assert_eq!(output.to_vec(), expected);
    }

    #[test]
    fn test_backward_basic() {
        let mut layer = MaxPooling2D::new(2, 2);
        let shape = Shape::from(IxDyn(&[1, 1, 4, 4]));
        layer.build(shape.clone()).unwrap();

        // Forward input:
        let forward_data = vec![
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0,  10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let forward_input = Tensor::new(forward_data, shape.clone());
        // Call forward so max_indices are stored.
        layer.forward(&forward_input).unwrap(); 

        // Suppose gradient from next layer is 1.0 in the 2x2 output.
        let grad_data = vec![1.0, 1.0, 1.0, 1.0];
        let grad_shape = Shape::from(IxDyn(&[1, 1, 2, 2]));
        let grad_tensor = Tensor::new(grad_data, grad_shape);

        // Backward
        let dx = layer.backward(&grad_tensor).unwrap();
        // dx should be [1,1,4,4].
        assert_eq!(dx.shape().raw_dim().slice(), &[1, 1, 4, 4]);

        // The maxima were at indices (1,1), (1,3), (3,1), (3,3) => each gets +1. Others get 0.
        let dx_data = dx.to_vec();
        let expected = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ];
        assert_eq!(dx_data, expected);
    }

    #[test]
    fn test_update_weights_no_op() {
        let mut layer = MaxPooling2D::new(2, 2);
        let shape = Shape::from(IxDyn(&[2, 3, 16, 16]));
        layer.build(shape).unwrap();

        // Pooling has no weights, so update_weights() should be Ok and do nothing.
        let mut dummy_optimizer: Box<dyn Optimizer> = Box::new(MockOptimizer {});
        let result = layer.update_weights(&mut dummy_optimizer);
        assert!(result.is_ok(), "update_weights should be a no-op for a pooling layer.");
    }

    // A mock optimizer for testing update_weights(). It does nothing.
    #[derive(Debug)]
    struct MockOptimizer;
    impl crate::optimizers::Optimizer for MockOptimizer {
        fn step(
            &mut self,
            _weights: &mut Tensor,
            _gradients: &Tensor
        ) -> Result<(), crate::optimizers::error::OptimizerError> {
            Ok(())
        }
        fn set_device(&mut self, _device: &crate::devices::Device) {}
    }
}
