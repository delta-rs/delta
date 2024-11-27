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

use delta_common::data::DatasetOps;
use delta_common::{Dataset, Layer, Loss, Optimizer};

#[derive(Debug)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    loss: Option<Box<dyn Loss>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            loss: None,
        }
    }

    /// Add a layer to the model
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer to add
    ///
    /// # Returns
    ///
    /// A reference to the model
    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Compile the model
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to use
    ///
    /// # Returns
    ///
    /// A reference to the model
    pub fn compile<O: Optimizer + 'static, L: Loss + 'static>(&mut self, optimizer: O, loss: L) {
        self.optimizer = Some(Box::new(optimizer));
        self.loss = Some(Box::new(loss));
    }

    pub fn train(&self, train_data: &Dataset, batch_size: usize) {
        let _ = batch_size;
        let _ = train_data;
        // Implement training logic here
    }

    /// Train the model
    ///
    /// # Arguments
    ///
    /// * `train_data` - The training data
    /// * `epochs` - The number of epochs to train for
    /// * `batch_size` - The batch size to use
    ///
    /// # Returns
    ///
    /// A reference to the model
    pub fn fit<D: DatasetOps>(&mut self, train_data: &D, epochs: i32, batch_size: usize) {
        // Ensure optimizer is set
        if self.optimizer.is_none() {
            panic!("Optimizer must be set before training");
        }

        let _optimizer = self.optimizer.as_mut().unwrap();

        for epoch in 0..epochs {
            println!("Epoch {}/{}", epoch + 1, epochs);

            // Shuffle dataset if necessary
            let dataset = train_data;
            // TODO: Probably should implement a shuffle capability
            // dataset.shuffle();

            let num_batches = dataset.len() / batch_size;
            let mut epoch_loss = 0.0;

            for batch_idx in 0..num_batches {
                // Fetch batch
                let (inputs, targets) = dataset.get_batch(batch_idx, batch_size);

                // Forward pass
                let mut outputs = inputs.clone();
                for layer in &self.layers {
                    outputs = layer.forward(&outputs);
                }

                // Compute loss
                if self.loss.is_some() {
                    let loss = self.loss.as_ref().unwrap();
                    epoch_loss += loss.calculate_loss(&outputs, &targets);
                } else {
                    epoch_loss += 0.0;
                }

                // Need to figure out how to handle this properly
                // Backward pass
                // let mut grad = train_data.loss_grad(&outputs, &targets);
                // for layer in self.layers.iter().rev() {
                //     grad = layer.backward(&grad);
                // }

                // Update weights
                // for layer in &mut self.layers {
                //     layer.update(optimizer);
                // }
            }

            println!(
                "Epoch {} completed. Average Loss: {:.4}",
                epoch + 1,
                epoch_loss / num_batches as f32
            );
        }
    }

    pub fn validate(&self, test_data: &Dataset) -> f32 {
        let _ = test_data;
        // Implement validation logic here
        0.0 // Placeholder
    }

    pub fn evaluate<D: DatasetOps>(&self, test_data: &D) -> f32 {
        let _ = test_data;
        // Implement evaluation logic here
        0.0 // Placeholder
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let _ = path;
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
