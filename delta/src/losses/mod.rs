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

pub mod cross_entropy;
pub mod huber;
pub mod mean_absolute_error;
pub mod mean_squared;
pub mod sparse_categorical_cross_entropy;

use std::fmt::Debug;

pub use cross_entropy::CrossEntropyLoss;
pub use huber::HuberLoss;
pub use mean_absolute_error::MeanAbsoluteError;
pub use mean_squared::MeanSquaredLoss;
pub use sparse_categorical_cross_entropy::SparseCategoricalCrossEntropyLoss;

use crate::common::Tensor;

/// A trait representing a loss function.
pub trait Loss: Debug {
    /// Calculates the loss between the output and the target tensors.
    ///
    /// # Arguments
    ///
    /// * `output` - The output tensor from the model.
    /// * `target` - The target tensor.
    ///
    /// # Returns
    ///
    /// The calculated loss as a `f32` value.
    fn calculate_loss(&self, output: &Tensor, target: &Tensor) -> f32;

    /// Calculates the gradient of the loss with respect to the output tensor.
    ///
    /// # Arguments
    ///
    /// * `output` - The output tensor from the model.
    /// * `target` - The target tensor.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the gradient of the loss with respect to the output tensor.
    fn calculate_loss_grad(&self, output: &Tensor, target: &Tensor) -> Tensor;
}
