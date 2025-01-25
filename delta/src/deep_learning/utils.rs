// BSD 3-Clause License
//
// Copyright (c) 2025, BlackPortal ○
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use ndarray::ArrayD;

use super::tensor_ops::Tensor;

/// Formats a given number with commas as a thousand separators.
///
/// # Arguments
///
/// * `num` - The number to be formatted.
///
/// # Returns
///
/// A `String` representing the formatted number with commas.
pub fn format_with_commas(num: usize) -> String {
    let num_str = num.to_string();
    let mut formatted = String::new();

    for (count, c) in num_str.chars().rev().enumerate() {
        if count > 0 && count % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(c);
    }

    formatted.chars().rev().collect()
}

/// Checks for NaN values in the given tensors and panics if any are found.
///
/// # Arguments
///
/// * `y_true` - The true values tensor.
/// * `y_pred` - The predicted values tensor.
///
/// # Panics
///
/// Panics if any NaN values are found in either `y_true` or `y_pred`.
pub fn check_for_nan(y_true: &Tensor, y_pred: &Tensor) {
    if y_true.data.iter().any(|&x| x.is_nan()) || y_pred.data.iter().any(|&x| x.is_nan()) {
        panic!("NaN value found in inputs");
    }
}

/// Asserts that the elements of the actual array are almost equal to the expected values within a given tolerance.
///
/// # Arguments
///
/// * `actual` - A reference to the actual `ArrayD<f32>` array.
/// * `expected` - A slice of expected `f32` values.
/// * `tolerance` - The tolerance within which the values are considered almost equal.
///
/// # Panics
///
/// Panics if the conversion of `actual` to a slice fails or if any element in `actual` differs from the corresponding element in `expected` by more than `tolerance`.
#[allow(dead_code)]
pub fn assert_almost_equal(actual: &ArrayD<f32>, expected: &[f32], tolerance: f32) {
    let actual_slice = actual.as_slice().expect("Failed to convert ArrayD to slice");
    for (a, e) in actual_slice.iter().zip(expected.iter()) {
        assert!((a - e).abs() < tolerance, "Expected: {:?}, Actual: {:?}", e, a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_with_commas() {
        assert_eq!(format_with_commas(0), "0");
        assert_eq!(format_with_commas(1), "1");
        assert_eq!(format_with_commas(10), "10");
        assert_eq!(format_with_commas(100), "100");
        assert_eq!(format_with_commas(1000), "1,000");
        assert_eq!(format_with_commas(10000), "10,000");
        assert_eq!(format_with_commas(100000), "100,000");
        assert_eq!(format_with_commas(1000000), "1,000,000");
        assert_eq!(format_with_commas(1000000000), "1,000,000,000");
    }
}
