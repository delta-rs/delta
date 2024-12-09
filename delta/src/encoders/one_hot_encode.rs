//! BSD 3-Clause License
//!
//! Copyright (c) 2024, The Delta Project Δ
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

use ndarray::Array2;

/// Converts a vector of class indices into a one-hot encoded 2D array.
///
/// # Arguments
///
/// * `class_indices` - A vector of class indices to encode.
/// * `num_classes` - The number of unique classes.
///
/// # Returns
///
/// A 2D array where each row represents the one-hot encoded vector for the corresponding class index.
///
/// # Panics
///
/// Panics if any class index is out of the range `[0, num_classes - 1]`.
///
/// # Example
///
/// ```
/// use deltaml::encoders::one_hot_encode;
///
/// let class_indices = vec![0, 1, 2];
/// let num_classes = 3;
/// let one_hot = one_hot_encode(&class_indices, num_classes);
///
/// assert_eq!(one_hot.shape(), &[3, 3]);
/// ```
pub fn one_hot_encode(class_indices: &[usize], num_classes: usize) -> Array2<f32> {
    if class_indices.iter().any(|&index| index >= num_classes) {
        panic!(
            "Class indices must be within the range [0, num_classes - 1]. Found out-of-range index."
        );
    }

    let num_samples = class_indices.len();
    let mut one_hot = Array2::<f32>::zeros((num_samples, num_classes));

    for (i, &class_index) in class_indices.iter().enumerate() {
        one_hot[(i, class_index)] = 1.0;
    }

    one_hot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encode() {
        let class_indices = vec![0, 1, 2];
        let num_classes = 3;
        let one_hot = one_hot_encode(&class_indices, num_classes);

        assert_eq!(one_hot.shape(), &[3, 3]);
        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[0, 1]], 0.0);
        assert_eq!(one_hot[[0, 2]], 0.0);
        assert_eq!(one_hot[[1, 0]], 0.0);
        assert_eq!(one_hot[[1, 1]], 1.0);
        assert_eq!(one_hot[[1, 2]], 0.0);
        assert_eq!(one_hot[[2, 0]], 0.0);
        assert_eq!(one_hot[[2, 1]], 0.0);
        assert_eq!(one_hot[[2, 2]], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_one_hot_encode_invalid_index() {
        let class_indices = vec![0, 1, 4];
        let num_classes = 3;
        let results = one_hot_encode(&class_indices, num_classes);

        assert_eq!(results.shape(), &[3, 3]);
    }
}
