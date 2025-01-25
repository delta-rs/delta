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

use std::future::{self, Future};
use std::pin::Pin;

use ndarray::{IxDyn, Shape, s};

use crate::deep_learning::dataset::{Dataset, DatasetOps};
use crate::deep_learning::tensor_ops::Tensor;

/// A struct representing a test dataset.
pub struct TestDataset {
    train: Option<Dataset>,
    test: Option<Dataset>,
    val: Option<Dataset>,
}

impl Default for TestDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl TestDataset {
    /// Creates a new `TestDataset` instance.
    #[inline]
    pub fn new() -> Self {
        TestDataset { train: None, test: None, val: None }
    }

    /// Generates a dummy dataset with the given size and number of features.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of samples in the dataset.
    /// * `features` - The number of features per sample.
    ///
    /// # Returns
    ///
    /// A `Dataset` instance with dummy dataset.
    #[inline]
    fn generate_dummy_dataset(size: usize, features: usize) -> Dataset {
        let inputs = Tensor::new(
            (0..size * features).map(|x| x as f32).collect(),
            Shape::from(IxDyn(&[size, features])),
        );
        let labels =
            Tensor::new((0..size).map(|x| (x % 2) as f32).collect(), Shape::from(IxDyn(&[size])));
        Dataset { inputs, labels }
    }

    /// Splits the training data into training and validation datasets.
    ///
    /// # Arguments
    ///
    /// * `validation_split` - The fraction of the training data to use for validation.
    ///
    /// # Returns
    ///
    /// A tuple containing the training and validation datasets.
    fn split_train_validation(&mut self, validation_split: f32) {
        if let Some(train_data) = &self.train {
            let total_samples = train_data.inputs.shape().raw_dim()[0];
            let validation_size = (total_samples as f32 * validation_split).round() as usize;
            let train_size = total_samples - validation_size;

            let (train_inputs, val_inputs) = train_data.inputs.split_at(train_size);
            let (train_labels, val_labels) = train_data.labels.split_at(train_size);

            self.train = Some(Dataset::new(train_inputs, train_labels));
            self.val = Some(Dataset::new(val_inputs, val_labels));
        } else {
            panic!("Training dataset not loaded!");
        }
    }
}

impl DatasetOps for TestDataset {
    type LoadFuture = Pin<Box<dyn Future<Output = Self> + Send>>;

    /// Loads the training dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to a `TestDataset` instance with the training dataset loaded.
    fn load_train() -> Self::LoadFuture {
        Box::pin(future::ready(Self {
            train: Some(Self::generate_dummy_dataset(100, 10)),
            test: None,
            val: None,
        }))
    }

    /// Loads the test dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to a `TestDataset` instance with the test dataset loaded.
    fn load_test() -> Self::LoadFuture {
        Box::pin(future::ready(Self {
            train: None,
            test: Some(Self::generate_dummy_dataset(50, 10)),
            val: None,
        }))
    }

    /// Loads the validation dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to a `TestDataset` instance with the test dataset loaded.
    fn load_val() -> Self::LoadFuture {
        Box::pin(async {
            let mut dataset = TestDataset::new();
            dataset.train = Some(Self::generate_dummy_dataset(100, 10));
            dataset.split_train_validation(0.2); // Use 20% of the training data for validation
            dataset
        })
    }

    /// Normalizes the dataset to the given range.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the normalized range.
    /// * `max` - The maximum value of the normalized range.
    fn normalize(&mut self, min: f32, max: f32) {
        if let Some(dataset) = &mut self.train {
            dataset.inputs.normalize(min, max);
            dataset.labels.normalize(min, max);
        }
        if let Some(dataset) = &mut self.test {
            dataset.inputs.normalize(min, max);
            dataset.labels.normalize(min, max);
        }
    }

    /// Adds noise to the dataset.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - The level of noise to add.
    fn add_noise(&mut self, noise_level: f32) {
        if let Some(dataset) = &mut self.train {
            dataset.inputs.add_noise(noise_level);
        }
        if let Some(dataset) = &mut self.test {
            dataset.inputs.add_noise(noise_level);
        }
    }

    /// Returns the length of the dataset.
    ///
    /// # Returns
    ///
    /// The number of samples in the dataset.
    #[inline]
    fn len(&self) -> usize {
        self.train
            .as_ref()
            .map(|d| d.inputs.data.len())
            .unwrap_or_else(|| self.test.as_ref().map(|d| d.inputs.data.len()).unwrap_or(0))
    }

    /// Retrieves a batch of dataset from the dataset.
    ///
    /// # Arguments
    ///
    /// * `batch_idx` - The index of the batch to retrieve.
    /// * `batch_size` - The size of the batch to retrieve.
    ///
    /// # Returns
    ///
    /// A tuple containing the inputs and labels for the batch.
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor) {
        if let Some(dataset) = &self.train {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(dataset.inputs.data.len());

            let inputs = dataset.inputs.data.slice(s![start..end, ..]).to_owned();
            let labels = dataset.labels.data.slice(s![start..end]).to_owned();

            return (
                Tensor::new(
                    inputs.iter().cloned().collect(),
                    Shape::from(IxDyn(&[end - start, dataset.inputs.shape().raw_dim()[1]])),
                ),
                Tensor::new(labels.iter().cloned().collect(), Shape::from(IxDyn(&[end - start]))),
            );
        }

        (Tensor::default(), Tensor::default())
    }

    /// Computes the loss between the outputs and targets.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The predicted outputs.
    /// * `targets` - The true targets.
    ///
    /// # Returns
    ///
    /// The computed loss value.
    fn loss(&self, outputs: &Tensor, targets: &Tensor) -> f32 {
        outputs.data.iter().zip(&targets.data).map(|(o, t)| (o - t).powi(2)).sum()
    }

    /// Computes the gradient of the loss with respect to the outputs.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The predicted outputs.
    /// * `targets` - The true targets.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the computed gradients.
    fn loss_grad(&self, outputs: &Tensor, targets: &Tensor) -> Tensor {
        let grad = outputs.data.iter().zip(&targets.data).map(|(o, t)| 2.0 * (o - t)).collect();
        Tensor::new(grad, outputs.shape().clone())
    }

    /// Shuffles the dataset.
    ///
    /// This method shuffles the training dataset by randomly permuting the indices of the samples.
    fn shuffle(&mut self) {
        todo!();
    }

    /// Clones the dataset.
    ///
    /// # Returns
    ///
    /// A new `TestDataset` instance that is a clone of the current instance.
    #[inline]
    fn clone(&self) -> Self {
        Self { train: self.train.clone(), test: self.test.clone(), val: self.val.clone() }
    }

    /// Transfers the dataset to the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to transfer the dataset to.
    ///
    /// # Returns
    ///
    /// A `Result` containing the dataset on the specified device.
    fn to_device(&mut self, device: crate::devices::Device) -> Result<(), String> {
        if let Some(train) = self.train.as_mut() {
            train.to_device(&device);
        }
        if let Some(test) = self.test.as_mut() {
            test.to_device(&device);
        }
        if let Some(val) = self.val.as_mut() {
            val.to_device(&device);
        }
        Ok(())
    }
}
