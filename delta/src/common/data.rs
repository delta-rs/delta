//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin
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
use std::future::Future;

/// A trait representing operations that can be performed on a dataset.
pub trait DatasetOps {
    /// The type of future returned by the `load_train` and `load_test` methods.
    type LoadFuture: Future<Output = Self> + Send;

    /// Loads the training dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the training dataset.
    fn load_train() -> Self::LoadFuture;

    /// Loads the test dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the test dataset.
    fn load_test() -> Self::LoadFuture;

    /// Normalizes the dataset.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value for normalization.
    /// * `max` - The maximum value for normalization.
    fn normalize(&mut self, min: f32, max: f32);

    /// Adds noise to the dataset.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - The level of noise to add.
    fn add_noise(&mut self, noise_level: f32);

    /// Returns the number of samples in the dataset.
    ///
    /// # Returns
    ///
    /// The number of samples in the dataset.
    fn len(&self) -> usize;

    /// Gets a batch of data from the dataset.
    ///
    /// # Arguments
    ///
    /// * `batch_idx` - The index of the batch to retrieve.
    /// * `batch_size` - The size of the batch to retrieve.
    ///
    /// # Returns
    ///
    /// A tuple containing the input tensor and the target tensor for the batch.
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor);

    /// Calculates the loss between the predicted outputs and the true targets.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The predicted outputs from the model.
    /// * `targets` - The true target values.
    ///
    /// # Returns
    ///
    /// The calculated loss as a `f32` value.
    fn loss(&self, outputs: &Tensor, targets: &Tensor) -> f32;

    /// Calculates the gradient of the loss with respect to the predicted outputs.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The predicted outputs from the model.
    /// * `targets` - The true target values.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the gradients of the loss with respect to the outputs.
    fn loss_grad(&self, outputs: &Tensor, targets: &Tensor) -> Tensor;

    /// Shuffles the dataset.
    fn shuffle(&mut self);

    /// Clones the dataset.
    ///
    /// # Returns
    ///
    /// A new instance of the dataset.
    fn clone(&self) -> Self;
}

/// A struct representing a dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub inputs: Tensor,
    pub labels: Tensor,
}

impl Dataset {
    /// Creates a new instance of `Dataset`.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input tensor.
    /// * `labels` - The label tensor.
    ///
    /// # Returns
    ///
    /// A new `Dataset` instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use deltaml::common::Dataset;
    /// use deltaml::common::shape::Shape;
    /// use deltaml::common::tensor_ops::Tensor;
    ///
    /// let inputs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    /// let labels = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]);
    /// let dataset = Dataset::new(inputs, labels);
    /// ```
    pub fn new(inputs: Tensor, labels: Tensor) -> Self {
        Dataset { inputs, labels }
    }
}
