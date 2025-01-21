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

pub mod ada_delta;
pub mod ada_grad;
pub mod adam;
pub mod error;
pub mod gradient_descent;
pub mod mini_batch_gd;
pub mod rms_prop;
pub mod sgd;
pub mod sgd_momentum;

pub use ada_delta::AdaDelta;
pub use ada_grad::AdaGrad;
pub use adam::Adam;
pub use gradient_descent::GradientDescent;
pub use mini_batch_gd::MiniBatchGD;
use ndarray::ArrayD;
pub use rms_prop::RMSProp;
pub use sgd::SGD;
pub use sgd_momentum::SGDWithMomentum;
use std::fmt::Debug;

use crate::common::Tensor;
use crate::devices::Device;
use crate::optimizers::error::OptimizerError;

/// A trait representing an optimizer for training neural networks.
pub trait Optimizer: Debug {
    /// Performs an optimization step using the given weights and gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError>;

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device);
}

/// A struct representing the configuration for an optimizer.
#[derive(Debug)]
pub struct OptimizerConfig {
    /// The learning rate for the optimizer.
    pub learning_rate: f32,
}

/// Asserts that the elements of the actual array are almost equal to the expected values within a given tolerance.
///
/// # Arguments
///
/// * `actual` - A reference to the actual `ArrayD<f32>` array.
/// * `expected` - A slice of expected `f32` values.
/// * `tolerance` - The tolerance within which the values are considered almost equal.
///
/// # Panics
///
/// Panics if the conversion of `actual` to a slice fails or if any element in `actual` differs from the corresponding element in `expected` by more than `tolerance`.
#[allow(dead_code)]
fn assert_almost_equal(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
    let actual_slice = actual.as_slice().expect("Failed to convert ArrayD to slice");
    for (a, e) in actual_slice.iter().zip(expected.iter()) {
        assert!((a - e).abs() < tolerance, "Expected: {:?}, Actual: {:?}", e, a);
    }
}
