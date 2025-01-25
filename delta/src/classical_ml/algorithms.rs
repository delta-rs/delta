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

use super::{Classical, losses::Loss, optimizers::Optimizer};

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

// pub struct LogisticRegression {
//     weights: Array1<f64>,
//     bias: f64,
// }

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

impl<T, L, O> Classical<T, L, O> for LinearRegression<T, L, O>
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
            let loss = self.calculate_loss(&predictions, y);
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

// impl Classical for LogisticRegression {
//     fn new() -> Self {
//         LogisticRegression { weights: Array1::zeros(1), bias: 0.0 }
//     }

//     fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize) {
//         for _ in 0..epochs {
//             let predictions = self.predict(x);
//             let loss = calculate_log_loss(&predictions, y);
//             let gradients = logistic_gradient_descent(x, y, &self.weights, self.bias);

//             self.weights = &self.weights - &(gradients.0 * learning_rate);
//             self.bias -= gradients.1 * learning_rate;

//             println!("Loss: {}", loss);
//         }
//     }

//     fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
//         let linear_output = x.dot(&self.weights) + self.bias;
//         linear_output.mapv(|x| 1.0 / (1.0 + (-x).exp()))
//     }
// }
