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

use serde_json;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::common::layer::Layer;
use crate::common::loss::Loss;
use crate::common::optimizer::Optimizer;
use crate::common::{Dataset, DatasetOps, Tensor};

/// A sequential model that contains a list of layers, an optimizer, and a loss function.
#[derive(Debug)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
    pub optimizer: Option<Box<dyn Optimizer>>,
    pub loss: Option<Box<dyn Loss>>,

    layer_names: Vec<String>,
}

impl Sequential {
    /// Creates a new sequential model.
    ///
    /// # Returns
    ///
    /// A new instance of the sequential model.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            layer_names: Vec::new(),
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
    pub fn add<L: Layer + 'static>(mut self, mut layer: L) -> Self {
        let layer_type = std::any::type_name::<L>().split("::").last().unwrap();
        let layer_name = match layer_type {
            "Dense" => format!("{}_{}_{}", layer_type, layer.units(), self.layers.len()),
            _ => format!("{}_{}", layer_type, self.layers.len()),
        };

        // Call the build method to initialize weights and biases
        if !self.layers.is_empty() {
            let input_shape = self.layers.last().unwrap().output_shape();
            layer.build(input_shape);
        }

        self.layers.push(Box::new(layer));
        self.layer_names.push(layer_name);
        self
    }

    /// Compiles the model with the given optimizer and loss function.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to use.
    /// * `loss` - The loss function to use.
    pub fn compile<O: Optimizer + 'static, L: Loss + 'static>(&mut self, optimizer: O, loss: L) {
        self.optimizer = Some(Box::new(optimizer));
        self.loss = Some(Box::new(loss));
    }

    /// Trains the model with the given training data, number of epochs, and batch size.
    ///
    /// # Arguments
    ///
    /// * `train_data` - The training data.
    /// * `epochs` - The number of epochs to train.
    /// * `batch_size` - The batch size to use.
    ///
    /// # Returns
    ///
    /// None
    ///
    /// # Example
    ///
    /// ```rust
    /// use deltaml::activations::{ReluActivation, SoftmaxActivation};
    /// use deltaml::common::DatasetOps;
    /// use deltaml::data::MnistDataset;
    /// use deltaml::losses::CrossEntropyLoss;
    /// use deltaml::neuralnet::{Dense, Sequential};
    /// use deltaml::optimizers::Adam;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut model = Sequential::new()
    ///       .add(Dense::new(128, Some(ReluActivation::new()), true))
    ///       .add(Dense::new(10, None::<SoftmaxActivation>, false));
    ///
    ///     let optimizer = Adam::new(0.001);
    ///     let loss = CrossEntropyLoss::new();
    ///
    ///     model.compile(optimizer, loss);
    ///
    ///     let mut train_data = MnistDataset::load_train().await;
    ///
    ///     model.fit(&mut train_data, 10, 32);
    /// }
    /// ```
    pub fn fit<D: DatasetOps>(&mut self, train_data: &mut D, epochs: i32, batch_size: usize) {
        self.ensure_optimizer_and_loss();

        let mut optimizer = self.optimizer.take().unwrap();

        for epoch in 0..epochs {
            println!("\nEpoch {}/{}", epoch + 1, epochs);
            self.train_one_epoch(train_data, batch_size, &mut optimizer);
        }
    }

    /// Ensures that the optimizer and loss function are set before training.
    fn ensure_optimizer_and_loss(&mut self) {
        if self.optimizer.is_none() {
            panic!("Optimizer must be set before training");
        }
        if self.loss.is_none() {
            panic!("Loss function must be set before training");
        }
    }

    /// Trains the model for one epoch using the given training data and batch size.
    ///
    /// # Arguments
    ///
    /// * `train_data` - The training data.
    /// * `batch_size` - The batch size to use.
    /// * `optimizer` - The optimizer to use.
    ///
    /// # Returns
    ///
    /// The average loss for the epoch.
    fn train_one_epoch<D: DatasetOps>(
        &mut self,
        train_data: &mut D,
        batch_size: usize,
        optimizer: &mut Box<dyn Optimizer>,
    ) -> f32 {
        let num_batches = train_data.len() / batch_size;
        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_samples = 0;

        let start_time = Instant::now();

        for batch_idx in 0..num_batches {
            let (inputs, targets) = train_data.get_batch(batch_idx, batch_size);
            let batch_loss = self.train_one_batch(&inputs, &targets, optimizer);
            epoch_loss += batch_loss;

            // Calculate accuracy
            let outputs = self.forward(&inputs);
            let predictions = outputs.argmax(1);
            let actuals = targets.argmax(1);
            correct_predictions += predictions
                .data
                .iter()
                .zip(actuals.data.iter())
                .filter(|(pred, actual)| pred == actual)
                .count();
            total_samples += targets.shape()[0];

            let accuracy = correct_predictions as f32 / total_samples as f32;
            self.display_progress(batch_idx, num_batches, epoch_loss, accuracy, start_time);
        }

        epoch_loss / num_batches as f32
    }

    /// Trains the model for one batch using the given inputs and targets.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The inputs for the batch.
    /// * `targets` - The targets for the batch.
    /// * `optimizer` - The optimizer to use.
    ///
    /// # Returns
    ///
    /// The loss for the batch.
    fn train_one_batch(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        optimizer: &mut Box<dyn Optimizer>,
    ) -> f32 {
        // Forward pass
        let mut outputs = inputs.clone();
        for layer in &mut self.layers {
            outputs = layer.forward(&outputs);
        }

        // Compute loss
        let loss_fn = self.loss.as_ref().unwrap();
        let batch_loss = loss_fn.calculate_loss(&outputs, targets);

        // Backward pass and update
        let mut grad = loss_fn.calculate_loss_grad(&outputs, targets);
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
            layer.update_weights(optimizer);
        }

        batch_loss
    }

    /// Displays the training progress bar.
    ///
    /// # Arguments
    ///
    /// * `batch_idx` - The index of the current batch.
    /// * `num_batches` - The total number of batches.
    /// * `epoch_loss` - The current epoch loss.
    /// * `accuracy` - The current accuracy.
    /// * `start_time` - The start time of the training process.
    fn display_progress(
        &mut self,
        batch_idx: usize,
        num_batches: usize,
        epoch_loss: f32,
        accuracy: f32,
        start_time: Instant,
    ) {
        let progress = (batch_idx + 1) as f32 / num_batches as f32;
        let current_avg_loss = epoch_loss / (batch_idx + 1) as f32;
        let bar_width = 30;
        let filled = (progress * bar_width as f32) as usize;
        let arrow = if filled < bar_width { ">" } else { "=" };
        let bar: String = std::iter::repeat('=')
            .take(filled)
            .chain(std::iter::once(arrow.chars().next().unwrap()))
            .chain(
                std::iter::repeat(' ')
                    .take((bar_width as isize - filled as isize - 1).max(0) as usize),
            )
            .collect();

        let elapsed = start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f32();
        let estimated_total = elapsed_secs / progress;
        let remaining_secs = (estimated_total - elapsed_secs).max(0.0);

        print!(
            "\rProgress: [{}] - ETA: {:.2}s - loss: {:.6} - accuracy: {:.4}",
            bar, remaining_secs, current_avg_loss, accuracy
        );
        std::io::stdout().flush().unwrap();
    }

    /// Validates the model with the given test data.
    ///
    /// # Arguments
    ///
    /// * `test_data` - The test data.
    ///
    /// # Returns
    ///
    /// The validation loss.
    pub fn validate(&self, test_data: &Dataset) -> f32 {
        let _ = test_data;
        // Implement validation logic here
        0.0 // Placeholder
    }

    /// Evaluates the model with the given test data.
    ///
    /// # Arguments
    ///
    /// * `test_data` - The test data.
    /// * `batch_size` - The batch size to use.
    ///
    /// # Returns
    ///
    /// The evaluation metric.
    pub fn evaluate<D: DatasetOps>(&mut self, test_data: &D, batch_size: usize) -> f32 {
        let mut correct_predictions = 0;
        let mut total_samples = 0;

        let num_batches = (test_data.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let (inputs, targets) = test_data.get_batch(batch_idx, batch_size);

            // Forward pass to get predictions
            let mut outputs = inputs.clone();
            for layer in &mut self.layers {
                outputs = layer.forward(&outputs);
            }

            // Determine the predicted class (argmax for classification)
            let predictions = outputs.argmax(1); // Assumes outputs support argmax
            let actuals = targets.argmax(1); // Assumes targets are one-hot encoded

            // Count correct predictions
            correct_predictions += predictions
                .data
                .iter()
                .zip(actuals.data.iter())
                .filter(|(pred, actual)| pred == actual)
                .count();

            total_samples += targets.shape()[0];
        }

        // Calculate accuracy as a percentage
        if total_samples == 0 {
            panic!("Test data contains no samples");
        }

        let accuracy = correct_predictions as f32 / total_samples as f32;
        accuracy
    }

    /// Saves the model to the specified path.
    ///
    /// # Arguments
    ///
    /// * `path_str` - The path to save the model to.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
    pub fn save(&self, path_str: &str) -> Result<(), std::io::Error> {
        // Create the model state as a JSON object
        let model_state = serde_json::json!({
            "layer_names": self.layer_names,
            "layers": self.layers.iter().map(|layer| {
                serde_json::json!({
                    "type": layer.type_name(),
                    "weights": layer.get_weights(),
                    "config": layer.get_config()
                })
            }).collect::<Vec<_>>()
        });

        // Create or open the file for writing
        let path = Path::new(path_str);
        let path = path.parent().unwrap();
        std::fs::create_dir_all(path)?;

        let mut file = File::create(path.join("model.json"))?;

        // Write the JSON data to the file
        file.write_all(serde_json::to_string_pretty(&model_state)?.as_bytes())?;

        Ok(())
    }

    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after passing through all layers.
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.layers
            .iter_mut()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    /// Prints a summary of the model.
    pub fn summary(&self) {
        println!("Model Summary:");
        println!(
            "{:<30} {:<25} {:<10}",
            "Layer (type)", "Output Shape", "Param #"
        );
        println!("{:-<65}", "");

        let mut total_params = 0;
        let mut trainable_params = 0;
        let mut non_trainable_params = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = &self.layer_names[i];
            let layer_type_only = layer_type.split('_').next().unwrap();
            let display_name = format!("{} ({})", layer_type, layer_type_only);
            let output_shape = format!("{:?}", layer.output_shape());
            let (trainable, non_trainable) = layer.param_count();
            total_params += trainable + non_trainable;
            trainable_params += trainable;
            non_trainable_params += non_trainable;
            println!(
                "{:<30} {:<25} {:<10}",
                display_name,
                output_shape,
                trainable + non_trainable
            );
        }

        println!("{:-<65}", "");
        println!("Total params: {}", total_params);
        println!("Trainable params: {}", trainable_params);
        println!("Non-trainable params: {}", non_trainable_params);
        println!("{:-<65}", "");
    }
}
