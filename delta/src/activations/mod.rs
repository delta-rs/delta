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

pub mod softmax;
pub mod relu;

use std::fmt::Debug;
pub use softmax::SoftmaxActivation;
pub use relu::ReluActivation;
use crate::common::Tensor;

/// A trait representing an activation function.
pub trait Activation: Debug {
    /// Applies the activation function to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the activation function.
    fn activate(&self, input: &Tensor) -> Tensor;

    /// Computes the derivative of the activation function.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The derivative tensor of the activation function.
    fn derivative(&self, input: &Tensor) -> Tensor;

    /// Returns the name of the activation function.
    ///
    /// # Returns
    ///
    /// A string slice containing the name of the activation function.
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }
}
