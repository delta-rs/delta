use std::ops::Range;

use ndarray::{Array, ArrayD, Axis, IxDyn};
use ndarray::{Dimension, Ix2};
use rand::Rng;

/// A struct representing a tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The data of the tensor stored as an n-dimensional array.
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// Creates a new tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of data.
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let shape = IxDyn(&shape);
        Self {
            data: Array::from_shape_vec(shape, data).expect("Invalid shape for data"),
        }
    }

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let shape = IxDyn(&shape);
        Self {
            data: Array::zeros(shape),
        }
    }

    /// Creates a tensor filled with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor filled with random values.
    pub fn random(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let shape = IxDyn(&shape); // Convert shape to dynamic dimension
        let data: Vec<f32> = (0..shape.size()).map(|_| rng.gen::<f32>()).collect(); // Use size() method
        Self {
            data: Array::from_shape_vec(shape, data).expect("Invalid shape for random data"),
        }
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
        let result = &self.data + &other.data;
        Tensor { data: result }
    }

    /// Gets the maximum value in the tensor.
    ///
    /// # Returns
    ///
    /// The maximum value in the tensor.
    pub fn max(&self) -> f32 {
        *self
            .data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Calculates the mean of the tensor.
    ///
    /// # Returns
    ///
    /// The mean of the tensor.
    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap_or(0.0)
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// A new tensor with the reshaped data.
    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        let shape = IxDyn(&shape);
        Tensor {
            data: self
                .data
                .clone()
                .into_shape_with_order(shape)
                .expect("Invalid shape"),
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
        // Create a new array by applying the function `f` to each element of `self.data`
        let new_data = self.data.mapv(|x| f(x));

        Tensor { data: new_data }
    }

    /// Slices the tensor along the specified indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of ranges for slicing along each axis.
    ///
    /// # Returns
    ///
    /// A new tensor containing the sliced data.
    pub fn slice(&self, indices: Vec<Range<usize>>) -> Tensor {
        let slices: Vec<_> = indices.iter().map(|r| r.clone().into()).collect();
        let view = self.data.slice(slices.as_slice());
        Tensor {
            data: view.to_owned(),
        }
    }

    /// Performs matrix multiplication between two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the matrix multiplication.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Ensure both tensors have at least 2 dimensions for matrix multiplication
        if self.data.ndim() < 2 || other.data.ndim() < 2 {
            panic!("Both tensors must have at least 2 dimensions for matmul");
        }

        // Extract the last two dimensions for matrix multiplication
        let self_2d = self
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Self tensor must be 2D for matmul");
        let other_2d = other
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Other tensor must be 2D for matmul");

        // Perform the matrix multiplication
        let result = self_2d.dot(&other_2d);

        // Wrap the result back into a Tensor with dynamic dimensions
        Tensor {
            data: result.into_dyn(),
        }
    }

    /// Transposes the tensor by swapping axes.
    ///
    /// # Returns
    ///
    /// A new tensor containing the transposed data.
    ///
    /// # Panics
    ///
    /// This method assumes the tensor is at least 2D.
    pub fn transpose(&self) -> Tensor {
        let ndim = self.data.ndim();
        if ndim < 2 {
            panic!("Cannot transpose a tensor with less than 2 dimensions");
        }

        // Create a transposed array by reversing the axes
        let axes: Vec<usize> = (0..ndim).rev().collect();
        Tensor {
            data: self.data.clone().permuted_axes(axes),
        }
    }

    /// Gets the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A vector representing the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Permutes the axes of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - A vector representing the new order of axes.
    ///
    /// # Returns
    ///
    /// A new tensor with the permuted axes.
    pub fn permute(&self, axes: Vec<usize>) -> Tensor {
        Tensor {
            data: self.data.clone().permuted_axes(axes),
        }
    }

    pub fn sum_along_axis(&self, axis: usize) -> Tensor {
        let sum = self.data.sum_axis(Axis(axis));
        Tensor { data: sum }
    }
}
