//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin, Chase Willden
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

use delta_common::{tensor_ops::Tensor, Loss};

#[derive(Debug)]
pub struct MeanSquaredLoss;

impl MeanSquaredLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MeanSquaredLoss {
    fn calculate_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Step 1: Ensure the shapes of y_true and y_pred match
        if y_true.shape != y_pred.shape {
            panic!("Shape mismatch: y_true and y_pred must have the same shape.");
        }

        // Step 2: Compute the squared differences
        let squared_diff: Vec<f32> = y_true
            .data
            .iter()
            .zip(y_pred.data.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
            .collect();

        // Step 3: Calculate the mean of the squared differences
        let mean_squared_error = squared_diff.iter().sum::<f32>() / squared_diff.len() as f32;

        mean_squared_error
    }
}
