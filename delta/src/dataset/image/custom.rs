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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use crate::common::Tensor;
use crate::dataset::{Dataset, ImageDatasetOps};
use std::future::Future;
use std::pin::Pin;
use image::{DynamicImage, ImageReader};
use ndarray::s;

/// CustomImageDataset is a dataset that loads images and labels from a CSV file.
///
/// # Fields
///
/// * `images` - A vector of DynamicImages containing the loaded images.
/// * `data` - An optional Dataset containing the loaded inputs and labels.
pub struct CustomImageDataset {
    images: Vec<DynamicImage>,
    labels: Vec<f32>,
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

        let mut images = Vec::new();
        let mut labels = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let parts: Vec<&str> = line.split(',').collect();
            let image_path = parts[0];
            let label: f32 = parts[1].parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;

            let image = ImageReader::open(image_path)
                .map_err(|e| e.to_string())?
                .decode()
                .map_err(|e| e.to_string())?;

            images.push(image);
            labels.push(label);
        }

        Ok(Self { images, labels })
    }

    pub fn load_train_from_csv<P: AsRef<Path> + Send + 'static>(path: P) -> Pin<Box<dyn Future<Output = Self> + Send>> {
        Box::pin(async move {
            let file = File::open(path).expect("Failed to open train CSV file");
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            let mut images = Vec::new();
            let mut labels = Vec::new();

            while let Some(result) = lines.next() {
                let line = result.expect("Failed to read line");
                let parts: Vec<&str> = line.split(',').collect();
                let image_path = parts[0];
                let label: f32 = parts[1].parse().expect("Failed to parse label");

                let image = ImageReader::open(image_path)
                    .expect("Failed to open image file")
                    .decode()
                    .expect("Failed to decode image");

                images.push(image);
                labels.push(label);
            }

            CustomImageDataset { images, labels }
        })
    }

    pub fn load_test_from_csv<P: AsRef<Path> + Send + 'static>(path: P) -> Pin<Box<dyn Future<Output = Self> + Send>> {
        Box::pin(async move {
            let file = File::open(path).expect("Failed to open test CSV file");
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            let mut images = Vec::new();
            let mut labels = Vec::new();

            while let Some(line) = lines.next() {
                let line = line.expect("Failed to read line");
                let parts: Vec<&str> = line.split(',').collect();
                let image_path = parts[0];
                let label: f32 = parts[1].parse().expect("Failed to parse label");

                let image = ImageReader::open(image_path)
                    .expect("Failed to open image file")
                    .decode()
                    .expect("Failed to decode image");

                images.push(image);
                labels.push(label);
            }

            CustomImageDataset { images, labels }
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
        unimplemented!();
    }

    /// Adds noise to the dataset inputs.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - The level of noise to add.
    fn add_noise(&mut self, noise_level: f32) {
        unimplemented!();
    }

    /// Returns the number of samples in the dataset.
    ///
    /// # Returns
    ///
    /// The number of samples in the dataset.
    fn len(&self) -> usize {
        unimplemented!();
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
        unimplemented!();
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
    fn loss(&self, _outputs: &Tensor, _targets: &Tensor) -> f32 {
        unimplemented!();
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
    fn loss_grad(&self, _outputs: &Tensor, _targets: &Tensor) -> Tensor {
        unimplemented!();
    }

    /// Shuffles the dataset.
    fn shuffle(&mut self) {
        unimplemented!();
    }

    /// Clones the dataset.
    ///
    /// # Returns
    ///
    /// A clone of the dataset.
    fn clone(&self) -> Self {
        unimplemented!();
    }
}