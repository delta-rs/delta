// File: dataset/image/custom.rs

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use crate::common::Tensor;
use crate::dataset::{Dataset, ImageDatasetOps};
use std::future::Future;
use std::pin::Pin;

/// CustomImageDataset is a dataset that loads images and labels from a CSV file.
///
/// # Fields
///
/// * `data` - An optional Dataset containing the loaded inputs and labels.
pub struct CustomImageDataset {
    pub data: Option<Dataset>,
}

impl CustomImageDataset {
    /// Creates a new CustomImageDataset from a CSV file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the CSV file.
    ///
    /// # Returns
    ///
    /// A Result containing the CustomImageDataset or an error message.
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);

        let mut inputs = Vec::new();
        let mut labels = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let mut values = line.split(',').map(|s| s.parse::<f32>().unwrap());
            let label = values.next().unwrap();
            labels.push(label);
            inputs.extend(values);
        }

        let features = inputs.len() / labels.len();
        let inputs_tensor = Tensor::new(inputs, vec![labels.len(), features]);
        let labels_tensor = Tensor::new(labels, vec![labels.len()]);

        Ok(Self {
            data: Some(Dataset::new(inputs_tensor, labels_tensor)),
        })
    }
}

impl ImageDatasetOps for CustomImageDataset {
    type LoadFuture = Pin<Box<dyn Future<Output = Self> + Send>>;

    /// Loads the training dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the loaded training dataset.
    fn load_train() -> Self::LoadFuture {
        unimplemented!();
    }

    /// Loads the test dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the loaded test dataset.
    fn load_test() -> Self::LoadFuture {
        unimplemented!();
    }

    /// Normalizes the dataset inputs and labels to a specified range.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the normalization range.
    /// * `max` - The maximum value of the normalization range.
    fn normalize(&mut self, min: f32, max: f32) {
        if let Some(dataset) = &mut self.data {
            dataset.inputs.normalize(min, max);
            dataset.labels.normalize(min, max);
        }
    }

    /// Adds noise to the dataset inputs.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - The level of noise to add.
    fn add_noise(&mut self, noise_level: f32) {
        if let Some(dataset) = &mut self.data {
            dataset.inputs.add_noise(noise_level);
        }
    }

    /// Returns the number of samples in the dataset.
    ///
    /// # Returns
    ///
    /// The number of samples in the dataset.
    fn len(&self) -> usize {
        self.data.as_ref().map_or(0, |dataset| dataset.inputs.data.len())
    }

    /// Retrieves a batch of inputs and labels from the dataset.
    ///
    /// # Arguments
    ///
    /// * `batch_idx` - The index of the batch to retrieve.
    /// * `batch_size` - The size of the batch to retrieve.
    ///
    /// # Returns
    ///
    /// A tuple containing the batch of inputs and labels.
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor) {
        let dataset = self.data.as_ref().expect("Dataset not loaded");
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(dataset.inputs.data.len());

        let inputs = dataset.inputs.data.slice(s![start..end, ..]).to_owned();
        let labels = dataset.labels.data.slice(s![start..end]).to_owned();

        (
            Tensor::new(inputs.iter().cloned().collect(), vec![end - start, dataset.inputs.shape()[1]]),
            Tensor::new(labels.iter().cloned().collect(), vec![end - start]),
        )
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
    /// The loss between the outputs and targets.
    fn loss(&self, outputs: &Tensor, targets: &Tensor) -> f32 {
        self.data.as_ref().expect("Dataset not loaded").loss(outputs, targets)
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
    /// A Tensor containing the computed gradients.
    fn loss_grad(&self, outputs: &Tensor, targets: &Tensor) -> Tensor {
        self.data.as_ref().expect("Dataset not loaded").loss_grad(outputs, targets)
    }

    /// Shuffles the dataset.
    fn shuffle(&mut self) {
        if let Some(dataset) = &mut self.data {
            dataset.shuffle();
        }
    }

    /// Clones the dataset.
    ///
    /// # Returns
    ///
    /// A clone of the dataset.
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}