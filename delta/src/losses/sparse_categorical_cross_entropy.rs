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

use crate::common::Tensor;
use crate::devices::Device;
use crate::losses::Loss;

/// A struct representing the Sparse Categorical Cross-Entropy Loss function.
#[derive(Debug)]
pub struct SparseCategoricalCrossEntropyLoss;

impl Default for SparseCategoricalCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseCategoricalCrossEntropyLoss {
    /// Creates a new SparseCategoricalCrossEntropyLoss instance.
    pub fn new() -> Self {
        Self
    }

    /// Clips the tensor to avoid issues with log(0) or division by zero.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clip.
    /// * `epsilon` - The epsilon value to use for clipping.
    ///
    /// # Returns
    ///
    /// The clipped tensor.
    fn clip_tensor(&self, tensor: &Tensor, epsilon: f32) -> Tensor {
        tensor.map(|x| x.clamp(epsilon, 1.0 - epsilon))
    }

    /// Converts one-hot encoded `y_true` to class indices.
    ///
    /// # Arguments
    ///
    /// * `one_hot` - The one-hot encoded tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing class indices.
    fn one_hot_to_indices(&self, one_hot: &Tensor) -> Tensor {
        if one_hot.shape().raw_dim().ndim() != 2 {
            panic!(
                "Expected a 2D tensor for one-hot encoding, but got shape: {:?}",
                one_hot.shape()
            );
        }

        let rows = one_hot.shape().raw_dim()[0];
        let indices: Vec<f32> = one_hot
            .data
            .outer_iter()
            .map(|row| {
                row.iter()
                    .position(|&x| x == 1.0)
                    .expect("One-hot encoding must have exactly one '1' per row")
                    as f32
            })
            .collect();

        let mut indices = Tensor::new(indices, Shape::from(IxDyn(&[rows])));
        indices.device = one_hot.device.clone();
        indices
    }

    /// Preprocesses one-hot encoded `y_true` for use with sparse categorical cross-entropy loss.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The one-hot encoded tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing class indices.
    fn preprocess_one_hot(&self, y_true: &Tensor) -> Tensor {
        let shape = y_true.shape();
        if shape.raw_dim().ndim() != 2 {
            panic!("Expected a 2D tensor for one-hot encoding, but got shape: {:?}", shape);
        }

        let rows = shape.raw_dim()[0];
        let cols = shape.raw_dim()[1];
        let data = y_true.data.as_slice().expect("Tensor dataset must be accessible as a slice");

        let mut processed_data = Vec::new();
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            let row = &data[start..end]; // Slice the row
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .expect("Row cannot be empty");
            processed_data.extend((0..cols).map(|j| if j == max_idx { 1.0 } else { 0.0 }));
        }

        let mut new_tensor = Tensor::new(processed_data, Shape::from(IxDyn(&[rows, cols])));
        new_tensor.device = y_true.device.clone();
        new_tensor
    }
}

impl Loss for SparseCategoricalCrossEntropyLoss {
    /// Calculates the sparse categorical cross-entropy loss between the true and predicted values.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values (indices of the correct classes or one-hot encoded).
    /// * `y_pred` - The predicted values (probability distributions).
    ///
    /// # Returns
    ///
    /// The sparse categorical cross-entropy loss.
    fn calculate_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        let y_true = if y_true.shape().raw_dim().ndim() == 2 {
            // Handle one-hot encoding
            if y_true.shape().raw_dim()[1] != y_pred.shape().raw_dim()[1] {
                panic!(
                    "If y_true is one-hot encoded, it must have the same number of classes as y_pred. \nGot y_true: {:?}\nGot y_pred: {:?}",
                    y_true.shape(),
                    y_pred.shape()
                );
            }
            let y_true = self.preprocess_one_hot(y_true);

            self.one_hot_to_indices(&y_true)
        } else {
            // Validate 1D tensor
            if y_true.shape().raw_dim().ndim() != 1 {
                panic!(
                    "Expected y_true to be a 1D tensor of class indices, but got shape: {:?}",
                    y_true.shape()
                );
            }
            y_true.clone()
        };

        // Clip predictions to avoid log(0)
        let epsilon = 1e-12;
        let clipped_pred = self.clip_tensor(y_pred, epsilon);

        // Compute cross-entropy loss
        let batch_size = y_true.shape().raw_dim()[0];
        let num_classes = y_pred.shape().raw_dim()[1];
        let mut loss = 0.0;

        for i in 0..batch_size {
            let true_class = y_true.data[i] as usize;
            if true_class >= num_classes {
                panic!(
                    "Invalid class index in y_true: {}, exceeds number of classes: {}",
                    true_class, num_classes
                );
            }
            loss -= clipped_pred.data[[i, true_class]].ln();
        }

        loss / batch_size as f32
    }

    /// Calculates the gradient of the sparse categorical cross-entropy loss with respect to the predicted values.
    ///
    /// # Arguments
    ///
    /// * `output` - The predicted values (probability distributions).
    /// * `target` - The true values (indices of the correct classes).
    ///
    /// # Returns
    ///
    /// The gradient of the sparse categorical cross-entropy loss with respect to the predicted values.
    fn calculate_loss_grad(&self, output: &Tensor, target: &Tensor) -> Tensor {
        let target = if target.shape().raw_dim().ndim() == 2 {
            // Handle one-hot encoding
            if target.shape().raw_dim()[1] != output.shape().raw_dim()[1] {
                panic!(
                    "If target is one-hot encoded, it must have the same number of classes as output. Got: {:?}",
                    target.shape()
                );
            }
            let target = self.preprocess_one_hot(target);

            self.one_hot_to_indices(&target)
        } else {
            // Validate 1D tensor
            if target.shape().raw_dim().ndim() != 1 {
                panic!(
                    "Expected target to be a 1D tensor of class indices, but got shape: {:?}",
                    target.shape()
                );
            }
            target.clone()
        };

        let batch_size = target.shape().raw_dim()[0];
        let num_classes = output.shape().raw_dim()[1];
        let epsilon = 1e-12;
        let clipped_pred = self.clip_tensor(output, epsilon);

        let mut grad_data = vec![0.0; clipped_pred.data.len()];
        for i in 0..batch_size {
            let true_class = target.data[i] as usize;
            if true_class >= num_classes {
                panic!(
                    "Invalid class index in target: {}, exceeds number of classes: {}",
                    true_class, num_classes
                );
            }
            for j in 0..num_classes {
                let pred = clipped_pred.data[[i, j]];
                grad_data[i * num_classes + j] = if j == true_class {
                    (pred - 1.0) / batch_size as f32
                } else {
                    pred / batch_size as f32
                };
            }
        }

        let grad_shape = clipped_pred.shape().raw_dim().as_array_view().to_vec();
        Tensor {
            data: ndarray::Array::from_shape_vec(ndarray::IxDyn(&grad_shape), grad_data)
                .expect("Failed to create gradient tensor"),
            device: Device::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;
    use crate::common::Tensor;

    #[test]
    fn test_sparse_categorical_cross_entropy_loss() {
        let y_true = Tensor::new(vec![0.0, 1.0, 2.0], Shape::from(IxDyn(&[3])));
        let y_pred = Tensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            Shape::from(IxDyn(&[3, 3])),
        );
        let loss = SparseCategoricalCrossEntropyLoss::new();
        let loss_value = loss.calculate_loss(&y_true, &y_pred);
        assert_eq!(loss_value, 1.0336976);
    }

    #[test]
    fn test_sparse_categorical_cross_entropy_loss_grad() {
        let y_true = Tensor::new(vec![0.0, 1.0, 2.0], Shape::from(IxDyn(&[3])));
        let y_pred = Tensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            Shape::from(IxDyn(&[3, 3])),
        );
        let loss = SparseCategoricalCrossEntropyLoss::new();
        let grad = loss.calculate_loss_grad(&y_pred, &y_true);
        assert_eq!(grad.data.shape(), &[3, 3]);
    }

    #[test]
    fn test_preprocess_one_hot() {
        let y_true = Tensor::new(vec![1.0, 0.0, 0.0], Shape::from(IxDyn(&[1, 3])));
        let loss = SparseCategoricalCrossEntropyLoss::new();
        let result = loss.preprocess_one_hot(&y_true);
        assert_eq!(result.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, 0.0, 0.0]);
    }
}
