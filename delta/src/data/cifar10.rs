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

use std::io;
use crate::common::{Dataset, DatasetOps, Tensor};
use crate::data::MnistDataset;

/// A struct representing the CIFAR10 dataset.
pub struct Cifar10Dataset {
    train: Option<Dataset>,
    test: Option<Dataset>,
}

impl Cifar10Dataset {
    const CIFAR10_URL: &'static str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    const CIFAR10_TRAIN_DATA_FILENAME: &'static str = "train-images-idx3-ubyte.gz";
    const CIFAR10_TRAIN_LABELS_FILENAME: &'static str = "train-labels-idx1-ubyte.gz";
    const CIFAR10_TEST_DATA_FILENAME: &'static str = "t10k-images-idx3-ubyte.gz";
    const CIFAR10_TEST_LABELS_FILENAME: &'static str = "t10k-labels-idx1-ubyte.gz";
    const CIFAR10_IMAGE_SIZE: usize = 28;
    const CIFAR10_NUM_CLASSES: usize = 10;
    const TRAIN_EXAMPLES: usize = 60_000;
    const TEST_EXAMPLES: usize = 10_000;

    async fn load_data(is_train: bool) -> Dataset {}

    fn parse_images(data: &[u8], num_images: usize) -> Tensor {}

    fn parse_labels(data: &[u8], num_labels: usize) -> Tensor {}

    async fn get_bytes_data(filename: &str) -> Vec<u8> {}

    fn decompress_gz(file_path: &str) -> io::Result<Vec<u8>> {}
}

impl DatasetOps for Cifar10Dataset {
    type LoadFuture = ();

    fn load_train() -> Self::LoadFuture {}

    fn load_test() -> Self::LoadFuture {}

    fn normalize(&mut self, min: f32, max: f32) {}

    fn add_noise(&mut self, noise_level: f32) {}

    fn len(&self) -> usize {}

    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor) {}

    fn loss(&self, outputs: &Tensor, targets: &Tensor) -> f32 {}

    fn loss_grad(&self, outputs: &Tensor, targets: &Tensor) -> Tensor {}

    fn shuffle(&mut self) {}

    fn clone(&self) -> Self {}
}