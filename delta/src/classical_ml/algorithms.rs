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

use ndarray::{Array1, Array2};

use super::{Algorithm, losses::Loss, optimizers::Optimizer};

pub struct LinearRegression<T, L, O>
where
    L: Loss<T>,
    O: Optimizer<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
    optimizer: O,
}

pub struct LogisticRegression<T, L, O>
where
    L: Loss<T>,
    O: Optimizer<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
    optimizer: O,
}

impl<T, L, O> LinearRegression<T, L, O>
where
    T: num_traits::Float + ndarray::ScalarOperand,
    L: Loss<T>,
    O: Optimizer<T>,
{
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }
}

impl<T, L, O> Algorithm<T, L, O> for LinearRegression<T, L, O>
where
    T: num_traits::Float + ndarray::ScalarOperand,
    L: Loss<T>,
    O: Optimizer<T>,
{
    fn new(loss_function: L, optimizer: O) -> Self {
        LinearRegression { weights: Array1::zeros(1), bias: T::zero(), loss_function, optimizer }
    }

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let _loss = self.calculate_loss(&predictions, y);
            // println!("Epoch {}: Loss = {:?}", epoch, loss);

            // Compute gradients
            let errors = predictions - y;
            let gradients = x.t().dot(&errors) / T::from(y.len()).unwrap();
            let bias_gradient = errors.sum() / T::from(y.len()).unwrap();

            // Update weights and bias using the optimizer
            self.optimizer.update(
                &mut self.weights,
                &mut self.bias,
                &gradients,
                bias_gradient,
                learning_rate,
            );
        }
    }

    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }
}

impl<T, L, O> LogisticRegression<T, L, O>
where
    T: num_traits::Float + ndarray::ScalarOperand,
    L: Loss<T>,
    O: Optimizer<T>,
{
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }

    // TODO: we should create generics for Activation
    fn sigmoid(&self, linear_output: Array1<T>) -> Array1<T> {
        linear_output.mapv(|x| T::one() / (T::one() + (-x).exp()))
    }

    fn predict_linear(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }
}

impl<T, L, O> Algorithm<T, L, O> for LogisticRegression<T, L, O>
where
    T: num_traits::Float + ndarray::ScalarOperand,
    L: Loss<T>,
    O: Optimizer<T>,
{
    fn new(loss_function: L, optimizer: O) -> Self {
        LogisticRegression { weights: Array1::zeros(1), bias: T::zero(), loss_function, optimizer }
    }

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        for _ in 0..epochs {
            // Compute predictions using sigmoid
            let linear_output = self.predict_linear(x);
            let predictions = self.sigmoid(linear_output);

            // Calculate loss
            let _loss = self.calculate_loss(&predictions, y);

            // Compute gradients
            let errors = &predictions - y;
            let gradients = x.t().dot(&errors) / T::from(y.len()).unwrap();
            let bias_gradient = errors.sum() / T::from(y.len()).unwrap();

            // Update weights and bias using the optimizer
            self.optimizer.update(
                &mut self.weights,
                &mut self.bias,
                &gradients,
                bias_gradient,
                learning_rate,
            );
        }
    }

    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        let linear_output = self.predict_linear(x);
        self.sigmoid(linear_output)
    }
}
