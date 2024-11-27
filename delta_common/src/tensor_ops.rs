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

    /// Calculate the mean of the tensor
    ///
    /// # Returns
    ///
    /// The mean of the tensor
    pub fn reduce_mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0; // Handle empty tensor case
        }
        let sum: f32 = self.data.iter().sum();
        sum / self.data.len() as f32
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

    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f64) -> f64,
    {
        // Implement map logic here
        // This is a placeholder implementation
        Tensor::new(vec![], self.shape.clone())
    }

    /// Get the shape of the tensor
    ///
    /// # Returns
    ///
    /// The shape of the tensor
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

    /// Perform matrix multiplication between two tensors
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Ensure tensors have at least 2 dimensions
        if self.shape.0.len() < 2 || other.shape.0.len() < 2 {
            println!("{:#?}", self.shape.0);
            println!("{:#?}", other.shape.0);
            panic!("Both tensors must have at least 2 dimensions for matmul");
        }

        let self_inner = self.shape.0[self.shape.0.len() - 1];
        let other_inner = other.shape.0[other.shape.0.len() - 2];

        // Ensure the inner dimensions match for matrix multiplication
        if self_inner != other_inner {
            panic!("Inner dimensions do not match for matrix multiplication");
        }

        let mut result_shape = Vec::new();

        // Determine the broadcasted shape for all dimensions except the last two
        let max_dims = std::cmp::max(self.shape.0.len(), other.shape.0.len()) - 2;
        for i in 0..max_dims {
            let dim_self = self.shape.0.get(i).copied().unwrap_or(1); // Treat missing dimensions as 1
            let dim_other = other.shape.0.get(i).copied().unwrap_or(1); // Treat missing dimensions as 1

            if dim_self != dim_other && dim_self != 1 && dim_other != 1 {
                panic!("Incompatible broadcast dimensions");
            }
            result_shape.push(std::cmp::max(dim_self, dim_other));
        }

        // Add the output matrix dimensions
        let self_outer = self.shape.0[self.shape.0.len() - 2];
        let other_outer = other.shape.0[other.shape.0.len() - 1];
        result_shape.push(self_outer);
        result_shape.push(other_outer);

        // Flatten input tensors for easier iteration
        let self_data = self.broadcast_and_flatten(&result_shape);
        let other_data = other.broadcast_and_flatten(&result_shape);

        // Compute the result data
        let batch_size = result_shape[..result_shape.len() - 2]
            .iter()
            .product::<usize>();
        let self_inner_size = self_inner;
        let other_inner_size = other_outer;
        let result_inner_size = self_outer * other_outer;

        let mut result_data = vec![0.0; batch_size * result_inner_size];

        for batch_idx in 0..batch_size {
            for i in 0..self_outer {
                for j in 0..other_outer {
                    let mut sum = 0.0;
                    for k in 0..self_inner_size {
                        let self_idx =
                            batch_idx * self_outer * self_inner_size + i * self_inner_size + k;
                        let other_idx =
                            batch_idx * other_inner_size * other_outer + k * other_outer + j;
                        sum += self_data[self_idx] * other_data[other_idx];
                    }
                    let result_idx = batch_idx * result_inner_size + i * other_outer + j;
                    result_data[result_idx] = sum;
                }
            }
        }

        Tensor {
            data: result_data,
            shape: Shape(result_shape),
        }
    }

    /// Broadcast and flatten the tensor to the target shape
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The target shape for broadcasting and flattening
    ///
    /// # Returns
    ///
    /// A vector containing the flattened and broadcasted data
    fn broadcast_and_flatten(&self, target_shape: &[usize]) -> Vec<f32> {
        let mut expanded_shape = vec![1; target_shape.len()];
        let offset = target_shape.len() - self.shape.0.len();
        expanded_shape[offset..].copy_from_slice(&self.shape.0);

        // Check for broadcast compatibility
        for (dim, target_dim) in expanded_shape.iter().zip(target_shape.iter()) {
            if *dim != *target_dim && *dim != 1 {
                panic!("Tensors are not broadcastable to the target shape");
            }
        }

        // Generate the flattened and broadcasted data
        let mut result = Vec::new();
        let strides: Vec<usize> = expanded_shape
            .iter()
            .scan(1, |stride, &dim| {
                let current = *stride;
                *stride *= dim;
                Some(current)
            })
            .collect();

        for idx in 0..target_shape.iter().product::<usize>() {
            let mut src_idx = 0;
            let mut remaining = idx;
            for (dim, stride) in expanded_shape.iter().zip(&strides) {
                let pos = remaining / stride;
                remaining %= stride;
                src_idx *= dim;
                src_idx += pos.min(*dim - 1); // Handle broadcasting
            }
            result.push(self.data[src_idx]);
        }

        result
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
