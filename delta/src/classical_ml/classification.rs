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

use ndarray::{Array1, Array2};

use crate::classical_ml::{calculate_log_loss, logistic_gradient_descent};

use super::Classical;

pub struct LogisticRegression {
    weights: Array1<f64>,
    bias: f64,
}

impl Classical for LogisticRegression {
    fn new() -> Self {
        LogisticRegression { weights: Array1::zeros(1), bias: 0.0 }
    }

    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let loss = calculate_log_loss(&predictions, y);
            let gradients = logistic_gradient_descent(x, y, &self.weights, self.bias);

            self.weights = &self.weights - &(gradients.0 * learning_rate);
            self.bias -= gradients.1 * learning_rate;

            println!("Loss: {}", loss);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let linear_output = x.dot(&self.weights) + self.bias;
        linear_output.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}
