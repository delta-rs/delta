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

/// A struct representing a tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The data of the tensor.
    pub data: Vec<f32>,
    /// The shape of the tensor.
    pub shape: Shape,
}

impl Tensor {
    /// Creates a new tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of data.
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        Self { data, shape }
    }

    /// Adds two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to add.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let target_shape = self.get_broadcast_shape(other);

        let self_data = self.broadcast_and_flatten(&target_shape);
        let other_data = other.broadcast_and_flatten(&target_shape);

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

    /// Gets the maximum value in the tensor.
    ///
    /// # Returns
    ///
    /// The maximum value in the tensor.
    pub fn max(&self) -> f32 {
        self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone()
    }

    /// Gets the broadcast shape for two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor.
    ///
    /// # Returns
    ///
    /// A vector representing the broadcast shape.
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

    /// Applies a function to each element of the tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply.
    ///
    /// # Returns
    ///
    /// A new tensor with the result of applying the function.
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
            panic!("Both tensors must have at least 2 dimensions for matmul");
        }

        // Get the batch shapes and matrix dimensions
        let self_batch_shape = &self.shape.0[..self.shape.0.len() - 2];
        let other_batch_shape = &other.shape.0[..other.shape.0.len() - 2];

        // Compute the broadcasted batch shape
        let batch_shape = Tensor::get_broadcast_shape_tensors(self_batch_shape, other_batch_shape);

        // Get matrix dimensions
        let m = self.shape.0[self.shape.0.len() - 2];
        let k_self = self.shape.0[self.shape.0.len() - 1];

        let k_other = other.shape.0[other.shape.0.len() - 2];
        let n = other.shape.0[other.shape.0.len() - 1];

        // Ensure inner dimensions match
        if k_self != k_other {
            panic!("Inner dimensions do not match for matrix multiplication");
        }

        // Compute the result shape
        let mut result_shape = batch_shape.clone();
        result_shape.push(m);
        result_shape.push(n);

        // Broadcast self and other to the broadcasted batch shape
        let self_broadcasted = self.broadcast_to(&[batch_shape.clone(), vec![m, k_self]].concat());
        let other_broadcasted =
            other.broadcast_to(&[batch_shape.clone(), vec![k_other, n]].concat());

        // Now, perform matmul over the batch dimensions
        let batch_size = batch_shape.iter().product::<usize>();

        let mut result_data = vec![0.0; result_shape.iter().product()];

        let matrix_size = m * n;
        let self_matrix_size = m * k_self;
        let other_matrix_size = k_other * n;

        for batch_idx in 0..batch_size {
            // Compute the starting indices for self, other, and result
            let self_offset = batch_idx * self_matrix_size;
            let other_offset = batch_idx * other_matrix_size;
            let result_offset = batch_idx * matrix_size;

            // Perform matrix multiplication for this batch
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k_self {
                        let self_idx = self_offset + i * k_self + k;
                        let other_idx = other_offset + k * n + j;
                        sum += self_broadcasted.data[self_idx] * other_broadcasted.data[other_idx];
                    }
                    let result_idx = result_offset + i * n + j;
                    result_data[result_idx] = sum;
                }
            }
        }

        Tensor {
            data: result_data,
            shape: Shape(result_shape),
        }
    }

    /// Compute the broadcasted shape for two tensors
    ///
    /// # Arguments
    ///
    /// * `shape1` - The first shape
    /// * `shape2` - The second shape
    ///
    /// # Returns
    ///
    /// A vector representing the broadcasted shape
    fn get_broadcast_shape_tensors(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
        let mut result = Vec::new();
        let max_dims = std::cmp::max(shape1.len(), shape2.len());
        for i in 0..max_dims {
            let dim1 = shape1
                .get(shape1.len().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);
            let dim2 = shape2
                .get(shape2.len().saturating_sub(i + 1))
                .copied()
                .unwrap_or(1);

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                panic!("Shapes are not broadcastable for matmul");
            }

            result.push(std::cmp::max(dim1, dim2));
        }
        result.reverse();
        result
    }

    /// Broadcast the tensor to a new shape
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The target shape for broadcasting
    ///
    /// # Returns
    ///
    /// A new tensor with the same data but different shape
    fn broadcast_to(&self, target_shape: &[usize]) -> Tensor {
        let data = self.broadcast_and_flatten(target_shape);
        Tensor {
            data,
            shape: Shape(target_shape.to_vec()),
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

    /// Transpose the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the same data but different shape
    pub fn transpose(&self) -> Tensor {
        let mut new_shape = self.shape.0.clone();
        new_shape.reverse();
        Tensor {
            data: self.data.clone(),
            shape: Shape(new_shape),
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
        let expanded_shape = {
            let mut expanded = vec![1; target_shape.len()];
            let offset = target_shape.len() - self.shape.0.len();
            expanded[offset..].copy_from_slice(&self.shape.0);
            expanded
        };

        // Check compatibility
        for (dim, target_dim) in expanded_shape.iter().zip(target_shape.iter()) {
            if *dim != *target_dim && *dim != 1 {
                panic!("Tensors are not broadcastable to the target shape");
            }
        }

        // Generate broadcasted data
        let mut result = Vec::new();
        let strides: Vec<usize> = expanded_shape
            .iter()
            .rev()
            .scan(1, |stride, &dim| {
                let current = *stride;
                *stride *= dim;
                Some(current)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        for idx in 0..target_shape.iter().product::<usize>() {
            let mut src_idx = 0;
            let remaining = idx;
            for (&dim, &stride) in expanded_shape.iter().zip(&strides) {
                let pos = (remaining / stride) % dim;
                src_idx *= dim;
                src_idx += pos.min(dim - 1);
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

    #[test]
    fn test_matmul_with_broadcasting() {
        // Tensor A with shape [2, 3, 4]
        let a = Tensor::new(
            (0..24).map(|x| x as f32).collect(),
            Shape::new(vec![2, 3, 4]),
        );

        // Tensor B with shape [4, 5]
        let b = Tensor::new((0..20).map(|x| x as f32).collect(), Shape::new(vec![4, 5]));

        // Perform matmul
        let result = a.matmul(&b);

        // Expected shape is [2, 3, 5]
        assert_eq!(result.shape.0, vec![2, 3, 5]);
    }

    #[test]
    fn test_max() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let max = tensor.max();
        assert_eq!(max, 6.0);
    }
}
