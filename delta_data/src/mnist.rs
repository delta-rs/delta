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

use delta_common::data::DatasetOps;
use delta_common::tensor_ops::Tensor;
use delta_common::{Dataset, Shape};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{self, Read};

pub struct MnistDataset {
    train: Option<Dataset>,
    test: Option<Dataset>,
}

impl MnistDataset {
    const MNIST_URL: &'static str = "https://storage.googleapis.com/cvdf-datasets/mnist";
    const MNIST_TRAIN_DATA_FILENAME: &'static str = "train-images-idx3-ubyte.gz";
    const MNIST_TRAIN_LABELS_FILENAME: &'static str = "train-labels-idx1-ubyte.gz";
    const MNIST_TEST_DATA_FILENAME: &'static str = "t10k-images-idx3-ubyte.gz";
    const MNIST_TEST_LABELS_FILENAME: &'static str = "t10k-labels-idx1-ubyte.gz";
    const MNIST_IMAGE_SIZE: usize = 28;
    const MNIST_NUM_CLASSES: usize = 10;
    const TRAIN_EXAMPLES: usize = 60000;
    const TEST_EXAMPLES: usize = 10000;

    /// Load the MNIST dataset
    ///
    /// # Arguments
    ///
    /// * `is_train` - Whether to load the training or test dataset
    ///
    /// # Returns
    ///
    /// A dataset containing the MNIST data
    async fn load_data(is_train: bool) -> Dataset {
        let (data_filename, labels_filename, num_examples) = if is_train {
            (
                Self::MNIST_TRAIN_DATA_FILENAME,
                Self::MNIST_TRAIN_LABELS_FILENAME,
                Self::TRAIN_EXAMPLES,
            )
        } else {
            (
                Self::MNIST_TEST_DATA_FILENAME,
                Self::MNIST_TEST_LABELS_FILENAME,
                Self::TEST_EXAMPLES,
            )
        };

        let data_bytes = Self::get_bytes_data(data_filename).await;
        let labels_bytes = Self::get_bytes_data(labels_filename).await;

        let data = Self::parse_images(&data_bytes, num_examples);
        let labels = Self::parse_labels(&labels_bytes, num_examples);

        Dataset::new(data, labels)
    }

    /// Parse the images from the MNIST dataset
    ///
    /// # Arguments
    ///
    /// * `data` - The data to parse
    /// * `num_images` - The number of images to parse
    ///
    /// # Returns
    ///
    /// A tensor containing the parsed images
    fn parse_images(data: &[u8], num_images: usize) -> Tensor {
        let image_data = &data[16..]; // Skip the 16-byte header
        let num_pixels = Self::MNIST_IMAGE_SIZE * Self::MNIST_IMAGE_SIZE;
        let mut tensor_data = vec![0.0; num_images * num_pixels];

        for i in 0..num_images {
            let start = i * num_pixels;
            let end = start + num_pixels;
            for (j, &pixel) in image_data[start..end].iter().enumerate() {
                tensor_data[i * num_pixels + j] = pixel as f32 / 255.0; // Normalize to [0, 1]
            }
        }

        Tensor::new(
            tensor_data,
            Shape::new(vec![
                num_images,
                Self::MNIST_IMAGE_SIZE,
                Self::MNIST_IMAGE_SIZE,
                1,
            ]),
        )
    }

    /// Parse the labels from the MNIST dataset
    ///
    /// # Arguments
    ///
    /// * `data` - The data to parse
    /// * `num_labels` - The number of labels to parse
    ///
    /// # Returns
    ///
    /// A tensor containing the parsed labels
    fn parse_labels(data: &[u8], num_labels: usize) -> Tensor {
        let label_data = &data[8..]; // Skip the 8-byte header
        let mut tensor_data = vec![0.0; num_labels * Self::MNIST_NUM_CLASSES];

        for (i, &label) in label_data.iter().enumerate() {
            tensor_data[i * Self::MNIST_NUM_CLASSES + label as usize] = 1.0; // One-hot encoding
        }

        Tensor::new(
            tensor_data,
            Shape::new(vec![num_labels, Self::MNIST_NUM_CLASSES]),
        )
    }

    /// Download a file from the MNIST dataset
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to download
    ///
    /// # Returns
    ///
    /// A vector of bytes containing the downloaded data
    async fn get_bytes_data(name: &str) -> Vec<u8> {
        let file_path = format!(".cache/data/mnist/{}", name);
        if std::path::Path::new(&file_path).exists() {
            return Self::decompress_gz(&file_path).unwrap();
        }

        let url = format!("{}/{}", Self::MNIST_URL, name);
        println!("Downloading {}", url);

        let compressed_data = reqwest::get(url)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
            .to_vec();

        std::fs::create_dir_all(".cache/data/mnist").unwrap();
        std::fs::write(&file_path, &compressed_data).unwrap();

        Self::decompress_gz(&file_path).unwrap()
    }

    /// Decompress a gzip file
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the gzip file
    ///
    /// # Returns
    ///
    /// A vector of bytes containing the decompressed data
    fn decompress_gz(file_path: &str) -> io::Result<Vec<u8>> {
        let file = File::open(file_path)?;
        let mut decoder = GzDecoder::new(file);
        let mut decompressed_data = Vec::new();
        decoder.read_to_end(&mut decompressed_data)?;
        Ok(decompressed_data)
    }
}

impl DatasetOps for MnistDataset {
    /// Load the training dataset
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delta_data::mnist::MnistDataset;
    ///
    /// let dataset = MnistDataset::load_train().await;
    /// ```
    async fn load_train() -> Self {
        let train = Self::load_data(true).await;
        MnistDataset {
            train: Some(train),
            test: None,
        }
    }

    /// Load the test dataset
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delta_data::mnist::MnistDataset;
    ///
    /// let dataset = MnistDataset::load_test().await;
    /// ```
    async fn load_test() -> Self {
        let test = Self::load_data(false).await;
        MnistDataset {
            train: None,
            test: Some(test),
        }
    }

    fn normalize(&mut self, min: f32, max: f32) {
        let _ = max;
        let _ = min;
        todo!()
    }

    fn add_noise(&mut self, noise_level: f32) {
        let _ = noise_level;
        todo!()
    }

    fn len(&self) -> usize {
        if let Some(ref train) = self.train {
            train.inputs.shape().0[0]
        } else if let Some(ref test) = self.test {
            test.inputs.shape().0[0]
        } else {
            0
        }
    }
}
