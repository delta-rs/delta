// BSD 3-Clause License
//
// Copyright (c) 2025, BlackPortal ○
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

pub mod activations;
pub mod algorithms;
pub mod losses;
pub mod optimizers;

use losses::Loss;
use num_traits::Float;
use optimizers::Optimizer;

use ndarray::{Array1, Array2};

/// Defines a common interface for classical machine learning models.
///
/// This trait outlines the basic methods that all classical ML models should implement,
/// providing a uniform way to instantiate, train, and use models for predictions.
pub trait Algorithm<T, L, O>
where
    T: Float,
    L: Loss<T>,
    O: Optimizer<T>,
{
    /// Creates and returns a new instance of the model.
    ///
    /// This method should initialize the model with default parameters or learnable parameters
    /// set to initial values. The `Sized` constraint ensures that `Self` has a known size at compile time.
    fn new(loss_function: L, optimizer: O) -> Self
    where
        Self: Sized;

    /// Trains the model using the provided data.
    ///
    /// This method adjusts the model's parameters based on the input data to minimize the error
    /// between predictions and actual values. The training process involves multiple iterations
    /// or epochs, with each iteration potentially updating the model's parameters.
    ///
    /// # Arguments
    ///
    /// * `x` - An `Array2<f64>` representing the input features where each row is a sample
    ///   and each column is a feature.
    /// * `y` - An `Array1<f64>` representing the target values or labels for each sample.
    /// * `learning_rate` - A `f64` specifying how much to adjust the model's parameters with
    ///   each iteration.
    /// * `epochs` - A `usize` indicating the number of training iterations.
    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize);

    /// Makes predictions using the trained model.
    ///
    /// This method applies the model's learned parameters to new or test data to generate predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - An `Array2<f64>` representing the input features for which predictions are to be made.
    ///
    /// # Returns
    ///
    /// Returns an `Array1<f64>` where each element is the model's prediction for the corresponding
    /// input sample.
    fn predict(&self, x: &Array2<T>) -> Array1<T>;
}
