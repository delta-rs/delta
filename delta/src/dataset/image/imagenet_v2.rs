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

use crate::common::{Tensor};
use crate::encoders::one_hot_encode;
use flate2::read::GzDecoder;
use log::debug;
use ndarray::{s, ArrayBase, Axis, Dim, OwnedRepr};
use rand::seq::SliceRandom;
use rand::thread_rng;
use reqwest;
use std::fs::File;
use std::future::Future;
use std::{
    io::{self},
    path::Path,
    pin::Pin,
    process::Command,
};
use tokio::fs as async_fs;
use walkdir::WalkDir;
use crate::dataset::base::{Dataset, ImageDatasetOps};

/// A struct representing the ImageNetV2 dataset.
pub struct ImageNetV2Dataset {
    data: Option<Dataset>,
}

impl ImageNetV2Dataset {
    const IMAGENETV2_URLS: &'static [&'static str] = &[
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
    ];

    /// Load the ImageNetV2 dataset.
    ///
    /// # Arguments
    /// * `variant_index` - The index of the dataset variant to load.
    ///
    /// # Returns
    /// A future that resolves to the `ImageNetV2Dataset` with the ImageNetV2 dataset loaded.
    pub async fn load(variant_index: usize) -> Result<Dataset, String> {
        if variant_index >= Self::IMAGENETV2_URLS.len() {
            return Err("Invalid variant index".to_string());
        }

        let url = Self::IMAGENETV2_URLS[variant_index];
        let dataset_path = format!("{}/.cache/dataset/imagenetv2/variant_{}", env!("CARGO_MANIFEST_DIR"), variant_index);
        let archive_path = format!("{}.tar.gz", dataset_path);
        debug!("Downloading ImageNetV2 dataset from {}", &url);

        // If archive_path doesn't exist, download it
        if !Path::new(&archive_path).exists() {
            let response = reqwest::get(url).await.map_err(|e| e.to_string())?;
            let data = response.bytes().await.map_err(|e| e.to_string())?;
            async_fs::create_dir_all(&dataset_path)
                .await
                .map_err(|e| e.to_string())?;
            async_fs::write(&archive_path, &data)
                .await
                .map_err(|e| e.to_string())?;
            debug!("File successfully written to {}", &archive_path);
        }

        Self::decompress_and_untar(&archive_path, &dataset_path).map_err(|e| e.to_string())?;

        Self::parse_images_and_labels(&dataset_path).await
    }

    /// Decompress a gzip file
    fn decompress_and_untar(file_path: &str, output_path: &str) -> io::Result<()> {
        // First try using the Rust implementation
        let tar_gz = File::open(file_path)?;
        let decompressed = GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(decompressed);

        match archive.unpack(output_path) {
            Ok(_) => {
                debug!("Successfully decompressed to: {}", output_path);
                Ok(())
            }
            Err(e) => {
                debug!(
                    "Rust implementation failed: {}. Trying system tar command...",
                    e
                );
                // Fallback to system tar command
                // For some reason, the rust implementation fails with invalid gzip header dataset
                let status = Command::new("tar")
                    .arg("-xzf")
                    .arg(file_path)
                    .arg("-C")
                    .arg(output_path)
                    .status()?;

                if status.success() {
                    debug!(
                        "Successfully decompressed using system tar to: {}",
                        output_path
                    );
                    Ok(())
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Both Rust and system tar implementations failed",
                    ))
                }
            }
        }
    }

    /// Parses images and labels into a Dataset.
    ///
    /// # Arguments
    /// * `dataset_path` - The path to the directory containing the images and labels.
    ///
    /// # Returns
    /// A future that resolves to the parsed Dataset.
    async fn parse_images_and_labels(dataset_path: &str) -> Result<Dataset, String> {
        let mut images: Vec<Tensor> = vec![];
        let mut labels: Vec<usize> = vec![];
        let mut label_map: Vec<usize> = vec![];

        for entry in WalkDir::new(dataset_path) {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.file_type().is_file() {
                let path = entry.path();

                if let Some(parent) = path.parent() {
                    if let Some(label) = parent.file_name().and_then(|os_str| os_str.to_str()) {
                        if let Ok(label_idx) = label.parse::<usize>() {
                            if !label_map.contains(&label_idx) {
                                label_map.push(label_idx);
                            }
                            let img_data = async_fs::read(path).await.map_err(|e| {
                                format!("Failed to read image {}: {}", path.display(), e)
                            })?;

                            let img_tensor = Tensor::from_image_bytes(img_data)?;
                            images.push(img_tensor);
                            labels.push(label_idx);
                        }
                    }
                }
            }
        }

        let label_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
            one_hot_encode(&labels, label_map.len());
        let label_data_dyn = label_data.into_dyn();

        let image_tensor = Tensor::stack(&images)?;

        Ok(Dataset::new(
            image_tensor,
            Tensor {
                data: label_data_dyn,
            },
        ))
    }
}

impl ImageDatasetOps for ImageNetV2Dataset {
    type LoadFuture = Pin<Box<dyn Future<Output = ImageNetV2Dataset> + Send>>;

    /// Loads the ImageNetV2 dataset.
    ///
    /// # Returns
    /// A future that resolves to the `ImageNetV2Dataset` with the ImageNetV2 dataset loaded.
    fn load_train() -> Self::LoadFuture {
        Box::pin(async {
            match ImageNetV2Dataset::load(0).await {
                Ok(data) => ImageNetV2Dataset { data: Some(data) },
                Err(err) => panic!("Failed to load dataset: {}", err),
            }
        })
    }

    /// Loads the ImageNetV2 dataset.
    ///
    /// # Returns
    /// A future that resolves to the `ImageNetV2Dataset` with the ImageNetV2 dataset loaded.
    fn load_test() -> Self::LoadFuture {
        Box::pin(async {
            match ImageNetV2Dataset::load(0).await {
                Ok(data) => ImageNetV2Dataset { data: Some(data) },
                Err(err) => panic!("Failed to load dataset: {}", err),
            }
        })
    }

    /// Normalizes the dataset to the given range.
    ///
    /// # Arguments
    /// * `min` - The minimum value of the normalized range.
    /// * `max` - The maximum value of the normalized range.
    fn normalize(&mut self, min: f32, max: f32) {
        // Ensure the dataset is loaded
        if let Some(dataset) = &mut self.data {
            let data_min = dataset
                .inputs
                .data
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let data_max = dataset
                .inputs
                .data
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            // Scale each value to the new range
            dataset
                .inputs
                .data
                .mapv_inplace(|x| (x - data_min) / (data_max - data_min) * (max - min) + min);
        } else {
            panic!("Dataset not loaded. Call `load_train` or `load_test` first.");
        }
    }

    fn add_noise(&mut self, noise_level: f32) {
        let _ = noise_level;
        todo!();
    }

    fn len(&self) -> usize {
        self.data
            .as_ref()
            .map(|ds| ds.inputs.shape()[0])
            .unwrap_or(0)
    }

    /// Gets a batch of dataset from the dataset.
    ///
    /// # Arguments
    /// * `batch_idx` - The index of the batch to get.
    /// * `batch_size` - The size of the batch to get.
    ///
    /// # Returns
    /// A tuple containing the input and label tensors for the batch.
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor) {
        // Ensure the dataset is loaded
        let dataset = self
            .data
            .as_ref()
            .expect("Dataset not loaded. Call `load_train` or `load_test` first.");

        let num_samples = dataset.inputs.shape()[0];
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(num_samples);

        // Ensure indices are valid
        if start_idx >= num_samples {
            panic!(
                "Invalid batch index: {}, batch_size: {}. Dataset only has {} samples.",
                batch_idx, batch_size, num_samples
            );
        }

        // Extract the batch from inputs and labels using slicing
        let batch_inputs = dataset.inputs.slice(vec![start_idx..end_idx]).to_owned();
        let batch_labels = dataset
            .labels
            .data
            .slice(s![start_idx..end_idx, ..])
            .to_owned();

        (
            Tensor {
                data: batch_inputs.data.into_dyn(), // Convert to dynamic dimensionality
            },
            Tensor {
                data: batch_labels.into_dyn(),
            },
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
    /// The loss between the outputs and targets.
    fn loss(&self, outputs: &Tensor, targets: &Tensor) -> f32 {
        // Apply softmax to outputs to get probabilities
        let softmax = outputs.data.mapv(|z| z.exp())
            / outputs
                .data
                .mapv(|z| z.exp())
                .sum_axis(Axis(1))
                .insert_axis(Axis(1));

        // Compute the Cross-Entropy Loss
        let log_probs = softmax.mapv(f32::ln); // Logarithm of softmax probabilities
        let loss_matrix = &targets.data * log_probs; // Element-wise multiplication
        let total_loss: f32 = -loss_matrix.sum(); // Sum all elements and negate

        // Return the mean loss
        total_loss / targets.data.shape()[0] as f32
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
        // Apply softmax to outputs to get probabilities
        let softmax = outputs.data.mapv(|z| z.exp())
            / outputs
                .data
                .mapv(|z| z.exp())
                .sum_axis(Axis(1))
                .insert_axis(Axis(1));

        // Calculate gradient: softmax(outputs) - targets
        let grad_data = softmax - &targets.data;

        Tensor { data: grad_data }
    }

    /// Shuffles the dataset.
    fn shuffle(&mut self) {
        if let Some(dataset) = &mut self.data {
            // Get the number of samples
            let num_samples = dataset.inputs.shape()[0];

            // Create an index array and shuffle it
            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut thread_rng());

            // Apply the shuffled indices to the inputs and labels
            let shuffled_inputs = dataset.inputs.data.select(Axis(0), &indices);
            let shuffled_labels = dataset.labels.data.select(Axis(0), &indices);

            // Update the dataset with shuffled dataset
            dataset.inputs.data = shuffled_inputs;
            dataset.labels.data = shuffled_labels;
        }
    }

    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}
