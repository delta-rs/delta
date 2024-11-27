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

use rand::Rng;

use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        Self { data, shape }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        // Implement element-wise addition logic here
        // This is a placeholder implementation
        Tensor::new(vec![], self.shape.clone())
    }

    /// Create a tensor filled with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor filled with zeros
    pub fn zeros(shape: &Shape) -> Self {
        let size = shape.len();
        let data = vec![0.0; size];

        Self {
            data,
            shape: shape.clone(),
        }
    }

    /// Create a tensor filled with random values
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    pub fn random(shape: &Shape) -> Self {
        let size = shape.len();
        let data = generate_random_data(size);

        Self {
            data,
            shape: shape.clone(),
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Implement matrix multiplication logic here
        // This is a placeholder implementation
        Tensor::new(vec![], self.shape.clone())
    }

    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f64) -> f64,
    {
        // Implement map logic here
        // This is a placeholder implementation
        Tensor::new(vec![], self.shape.clone())
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Slice the tensor along the specified indices
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of tuples, each containing the start and end indices to slice along
    ///
    /// # Returns
    ///
    /// A new tensor containing the sliced data
    pub fn slice(&self, indices: Vec<(usize, usize)>) -> Option<Tensor> {
        // Ensure the number of indices matches the number of dimensions
        if indices.len() > self.shape.0.len() {
            return None; // Invalid slicing request
        }

        // Compute the offset and the new shape
        let mut offset = 0;
        let mut new_dimensions = Vec::new();
        let mut stride = 1;

        // Calculate strides and validate indices
        for (i, &(start, end)) in indices.iter().enumerate().rev() {
            if i < indices.len() {
                // If an index is provided, check validity
                let dim_size = self.shape.0[i];
                if start >= dim_size || end > dim_size {
                    return None; // Index out of bounds
                }
                offset += start * stride;
                new_dimensions.push(end - start);
            } else {
                // If no index is provided, keep this dimension
                new_dimensions.push(self.shape.0[i]);
            }
            stride *= self.shape.0[i];
        }
        new_dimensions.reverse();

        // Extract data for the sliced tensor
        let new_size = new_dimensions.iter().product();
        let mut new_data = Vec::with_capacity(new_size);
        let mut current_offset = offset;
        let current_stride = stride / self.shape.0[0];

        // Collect the data for the new tensor
        for _ in 0..new_size {
            new_data.push(self.data[current_offset]);
            current_offset += current_stride;
        }

        Some(Tensor {
            data: new_data,
            shape: Shape(new_dimensions),
        })
    }
}

/// Generate a vector of random numbers
///
/// # Arguments
///
/// * `length` - The length of the vector
///
/// # Returns
///
/// A vector of random numbers
fn generate_random_data(length: usize) -> Vec<f32> {
    let mut random_number_generator = rand::thread_rng();
    let mut data = Vec::with_capacity(length);
    for _ in 0..length {
        data.push(random_number_generator.gen::<f32>());
    }
    data
}
