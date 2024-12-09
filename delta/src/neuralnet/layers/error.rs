//! BSD 3-Clause License
//!
//! Copyright (c) 2024, The Delta Project
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
use crate::optimizers::error::OptimizerError;

/// Errors that can occur in the Dense layer.
#[derive(Debug)]
pub enum LayerError {
    /// Error when weights are not initialized.
    UninitializedWeights,
    /// Error when bias is not initialized.
    UninitializedBias,
    /// Error when input is not set for backward pass.
    UninitializedInput,
    /// Error related to invalid shape.
    InvalidShape,
    /// Error when input is not set.
    MissingInput,

    /// Error when an optimizer error occurs.
    OptimizerError(OptimizerError),
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerError::UninitializedWeights => write!(f, "Weights must be initialized"),
            LayerError::UninitializedBias => write!(f, "Bias must be initialized"),
            LayerError::UninitializedInput => write!(f, "Input must be initialized"),
            LayerError::InvalidShape => write!(f, "Invalid shape"),
            LayerError::MissingInput => write!(f, "Input must be set"),
            LayerError::OptimizerError(err) => write!(f, "Optimizer error: {}", err),
        }
    }
}

impl std::error::Error for LayerError {}
