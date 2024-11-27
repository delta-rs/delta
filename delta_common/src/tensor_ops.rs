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
        // Get the target shape that both tensors will be broadcast to
        let target_shape = self.get_broadcast_shape(other);

        // Broadcast both tensors to the target shape
        let self_data = self.broadcast_and_flatten(&target_shape);
        let other_data = other.broadcast_and_flatten(&target_shape);

        // Perform element-wise addition
        let result_data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor {
            data: result_data,
            shape: Shape(target_shape),
        }
    }

    fn get_broadcast_shape(&self, other: &Tensor) -> Vec<usize> {
        let max_dims = std::cmp::max(self.shape.0.len(), other.shape.0.len());
        let mut result = Vec::with_capacity(max_dims);

        // Pad shorter shape with leading 1s
        let self_padded = {
            let pad_len = max_dims - self.shape.0.len();
            let mut padded = vec![1; pad_len];
            padded.extend(&self.shape.0);
            padded
        };

        let other_padded = {
            let pad_len = max_dims - other.shape.0.len();
            let mut padded = vec![1; pad_len];
            padded.extend(&other.shape.0);
            padded
        };

        // For each dimension, take the maximum of the two shapes
        for i in 0..max_dims {
            let dim1 = self_padded[i];
            let dim2 = other_padded[i];

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                panic!("Tensors are not broadcastable");
            }

            result.push(std::cmp::max(dim1, dim2));
        }

        result
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
        F: Fn(f32) -> f32,
    {
        let new_data = self.data.iter().map(|&x| f(x)).collect();
        Tensor {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    /// Get the shape of the tensor
    ///
    /// # Returns
    ///
    /// The shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Calculate the sum of the tensor
    ///
    /// # Returns
    ///
    /// The sum of the tensor
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
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
        if indices.len() != self.shape.0.len() {
            println!(
                "slice indices: {:?}, self.shape.0.len(): {}",
                indices,
                self.shape.0.len()
            );
            return None; // Ensure indices match dimensions
        }

        let mut new_dimensions = Vec::new();
        let mut start_offsets = Vec::new();
        for (dim_size, &(start, end)) in self.shape.0.iter().zip(&indices) {
            if start >= *dim_size || end > *dim_size || start >= end {
                println!("start: {}, end: {}, dim_size: {}", start, end, dim_size);
                return None; // Out-of-bounds or invalid range
            }
            new_dimensions.push(end - start);
            start_offsets.push(start);
        }

        // Correct stride calculation
        let strides = {
            let mut strides = vec![1; self.shape.0.len()];
            for i in (0..self.shape.0.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.shape.0[i + 1];
            }
            strides
        };

        fn collect_data(
            data: &[f32],
            dims: &[usize],
            strides: &[usize],
            offsets: &[usize],
            current_offset: usize,
            depth: usize,
            output: &mut Vec<f32>,
        ) {
            if depth == dims.len() {
                // Add the current element to the output
                output.push(data[current_offset]);
            } else {
                // Iterate over the specified slice range for the current dimension
                for i in 0..dims[depth] {
                    let next_offset = current_offset + (offsets[depth] + i) * strides[depth];
                    if next_offset >= data.len() {
                        panic!(
                            "Offset out of bounds: next_offset={}, data_len={}, depth={}, i={}, offsets={:?}",
                            next_offset, data.len(), depth, i, offsets
                        );
                    }
                    collect_data(data, dims, strides, offsets, next_offset, depth + 1, output);
                }
            }
        }

        let mut new_data = Vec::with_capacity(new_dimensions.iter().product());
        collect_data(
            &self.data,
            &new_dimensions,
            &strides,
            &start_offsets,
            0,
            0,
            &mut new_data,
        );

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
            println!("self_inner: {}, other_inner: {}", self_inner, other_inner);
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

    /// Perform element-wise operation between two tensors using a custom function
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor
    /// * `f` - The function to apply to corresponding elements of the tensors
    ///
    /// # Returns
    ///
    /// A new tensor with the result of applying the function
    ///
    /// # Panics
    ///
    /// Panics if the tensors are not broadcastable.
    pub fn zip_map<F>(&self, other: &Tensor, f: F) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        // Get the target shape that both tensors will be broadcast to
        let target_shape = self.get_broadcast_shape(other);

        // Broadcast both tensors to the target shape
        let self_data = self.broadcast_and_flatten(&target_shape);
        let other_data = other.broadcast_and_flatten(&target_shape);

        // Perform the operation
        let result_data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| f(*a, *b))
            .collect();

        Tensor {
            data: result_data,
            shape: Shape(target_shape),
        }
    }

    /// Reshape the tensor to a new shape
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The new shape for the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the same data but different shape
    pub fn reshape(&self, new_shape: Shape) -> Tensor {
        let new_size: usize = new_shape.0.iter().product();
        if new_size != self.data.len() {
            panic!("New shape must have the same number of elements as the original tensor");
        }

        Tensor {
            data: self.data.clone(),
            shape: new_shape,
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

// Create unit test for slice

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let sliced = tensor.slice(vec![(0, 1), (1, 3)]).unwrap();
        assert_eq!(sliced.data, vec![2.0, 3.0]);
        assert_eq!(sliced.shape.0, vec![1, 2]);
    }

    #[test]
    fn test_multi_dimensional_slice() {
        let tensor = Tensor::new(
            vec![1.0; 60000 * 28 * 28 * 1],
            Shape::new(vec![60000, 28, 28, 1]),
        );

        // Slice a smaller range for debugging
        let sliced = tensor
            .slice(vec![(0, 10), (0, 28), (0, 28), (0, 1)])
            .expect("Failed to slice tensor");

        assert_eq!(sliced.shape.0, vec![10, 28, 28, 1]);
        assert_eq!(sliced.data.len(), 10 * 28 * 28 * 1);
    }
}
