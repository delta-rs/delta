//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin
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

use delta_common::{Dataset, Layer, Optimizer};
use delta_common::data::DatasetOps;

#[derive(Debug)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new(), optimizer: None }
    }

    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn compile<O: Optimizer + 'static>(&mut self, optimizer: O) {
        self.optimizer = Some(Box::new(optimizer));
    }

    pub fn train(&self, train_data: &Dataset, batch_size: usize) {
        // Implement training logic here
    }

    pub fn fit<D: DatasetOps>(&self, train_data: &D, epochs: i32, batch_size: i32) {
        // Implement training logic here
    }

    pub fn validate(&self, test_data: &Dataset) -> f32 {
        // Implement validation logic here
        0.0 // Placeholder
    }

    pub fn evaluate<D: DatasetOps>(&self, test_data: &D) -> f32 {
        // Implement evaluation logic here
        0.0 // Placeholder
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        // Implement model saving logic here
        Ok(())
    }

    /*pub fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(input.clone(), |acc, layer| layer.forward(&acc))
    }*/

    pub fn summary(&self) -> String {
        let mut summary = String::new();
        for (i, layer) in self.layers.iter().enumerate() {
            summary.push_str(&format!("Layer {}: {:?}\n", i + 1, layer));
        }
        summary
    }
}