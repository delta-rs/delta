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

use std::fmt;
use crate::neuralnet::layers::error::LayerError;

/// An enumeration of possible errors that can occur in a model.
#[derive(Debug)]
pub enum ModelError {
    /// Error indicating that the optimizer is missing.
    MissingOptimizer,
    /// Error indicating that the loss function is missing.
    MissingLossFunction,
    /// Error related to the dataset, with a message describing the issue.
    DatasetError(String),
    /// Error that occurs during training, with a message describing the issue.
    TrainingError(String),
    /// Error related to a specific layer in the model.
    LayerError(LayerError),
}

impl fmt::Display for ModelError {
    /// Formats the `ModelError` for display purposes.
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating the success or failure of the formatting operation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::MissingOptimizer => write!(f, "Optimizer must be set before training"),
            ModelError::MissingLossFunction => {
                write!(f, "Loss function must be set before training")
            }
            ModelError::DatasetError(msg) => write!(f, "Dataset error: {}", msg),
            ModelError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            ModelError::LayerError(err) => write!(f, "Layer error: {}", err),
        }
    }
}

impl std::error::Error for ModelError {}
