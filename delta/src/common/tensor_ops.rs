use std::io::Cursor;
use std::ops::{Add, Index, IndexMut, Mul, Range, SubAssign};

use image::{GenericImageView, ImageReader};
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let shape = vec![2, 2];
    ///
    /// let tensor = Tensor::new(data, shape);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let shape = vec![2, 3];
    /// let tensor = Tensor::zeros(shape);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let shape = vec![2, 3];
    /// let tensor = Tensor::random(shape);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor1 = Tensor::new(data1, vec![2, 2]);
    /// let data2 = vec![5.0, 6.0, 7.0, 8.0];
    /// let tensor2 = Tensor::new(data2, vec![2, 2]);
    /// let result = tensor1.add(&tensor2);
    /// ```
    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: &self.data + &other.data,
        }
    }

    /// Gets the maximum value in the tensor.
    ///
    /// # Returns
    ///
    /// The maximum value in the tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let max_value = tensor.max();
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let mean = tensor.mean();
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let new_shape = vec![1, 4];
    /// let reshaped_tensor = tensor.reshape(new_shape);
    /// ```
    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        let shape = IxDyn(&shape);
        Tensor {
            data: self
                .data
                .clone()
                .into_shape_with_order(shape)
                .expect("Invalid shape for reshape"),
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let result = tensor.map(|x| x * 2.0);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let indices = vec![0..2, 0..2];
    /// let sliced_tensor = tensor.slice(indices);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor1 = Tensor::new(data1, vec![2, 2]);
    /// let data2 = vec![5.0, 6.0, 7.0, 8.0];
    /// let tensor2 = Tensor::new(data2, vec![2, 2]);
    /// let result = tensor1.matmul(&tensor2);
    /// ```
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

    /// Multiplies two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to multiply.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the element-wise multiplication.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor1 = Tensor::new(data1, vec![2, 2]);
    /// let data2 = vec![5.0, 6.0, 7.0, 8.0];
    /// let tensor2 = Tensor::new(data2, vec![2, 2]);
    /// let result = tensor1.mul(&tensor2);
    /// ```
    pub fn mul(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: &self.data * &other.data,
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let transposed_tensor = tensor.transpose();
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let shape = tensor.shape();
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let permuted_tensor = tensor.permute(vec![1, 0]);
    /// ```
    pub fn permute(&self, axes: Vec<usize>) -> Tensor {
        Tensor {
            data: self.data.clone().permuted_axes(axes),
        }
    }

    /// Sums the tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to sum along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the summed data.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let summed_tensor = tensor.sum_along_axis(1);
    /// ```
    pub fn sum_along_axis(&self, axis: usize) -> Tensor {
        let sum = self.data.sum_axis(Axis(axis));
        Tensor { data: sum }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().copied().sum()
    }

    /// Multiplies the tensor by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to multiply the tensor by.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let multiplied_tensor = tensor.mul_scalar(2.0);
    /// ```
    pub fn mul_scalar(&self, amount: f32) -> Tensor {
        self.map(|x| x * amount)
    }

    /// Raises the tensor to a power.
    ///
    /// # Arguments
    ///
    /// * `amount` - The power to raise the tensor to.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let powered_tensor = tensor.pow(2.0);
    /// ```
    pub fn pow(&self, amount: f32) -> Tensor {
        self.map(|x| x.powf(amount))
    }

    /// Divides the tensor by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to divide the tensor by.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let divided_tensor = tensor.div_scalar(2.0);
    /// ```
    pub fn div_scalar(&self, amount: f32) -> Tensor {
        self.map(|x| x / amount)
    }

    /// Computes the square root of each element in the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the square roots of the elements.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 4.0, 9.0, 16.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let sqrt_tensor = tensor.sqrt();
    /// ```
    pub fn sqrt(&self) -> Tensor {
        self.map(|x| x.sqrt())
    }

    /// Adds a scalar value to each element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to add to each element.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the addition.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let added_tensor = tensor.add_scalar(2.0);
    /// ```
    pub fn add_scalar(&self, amount: f32) -> Tensor {
        self.map(|x| x + amount)
    }

    /// Divides each element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to divide each element by.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the division.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let divided_tensor = tensor.div_scalar(2.0);
    /// ```
    pub fn div(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: &self.data / &other.data,
        }
    }

    /// Flattens the tensor into a 1D array.
    ///
    /// # Returns
    ///
    /// A new tensor containing the flattened data.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let flattened_tensor = tensor.flatten();
    /// ```
    pub fn flatten(&self) -> Tensor {
        let shape = IxDyn(&[self.data.len()]);
        Tensor {
            data: self.data.clone().into_shape_with_order(shape).unwrap(),
        }
    }

    /// Computes the mean along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to compute the mean along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the mean data.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let mean_tensor = tensor.mean_axis(1);
    /// ```
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let mean = self
            .data
            .mean_axis(Axis(axis))
            .expect("Failed to calculate mean");
        Tensor { data: mean }
    }

    /// Broadcasts the tensor to a target shape.
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The target shape to broadcast to.
    ///
    /// # Returns
    ///
    /// A new tensor with the broadcasted shape.
    ///
    /// # Panics
    ///
    /// Panics if the current shape cannot be broadcasted to the target shape.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let broadcasted_tensor = tensor.broadcast(vec![2, 2]);
    /// ```
    pub fn broadcast(&self, target_shape: Vec<usize>) -> Tensor {
        let self_shape = self.shape();
        let ndim_self = self_shape.len();
        let ndim_target = target_shape.len();

        // Pad the current shape with leading 1s to match the target dimensions
        let mut padded_shape = vec![1; ndim_target - ndim_self];
        padded_shape.extend(&self_shape);

        // Validate compatibility for broadcasting
        for (self_dim, target_dim) in padded_shape.iter().zip(&target_shape) {
            if *self_dim != *target_dim && *self_dim != 1 {
                panic!(
                    "Cannot broadcast shape {:?} to {:?}",
                    self.shape(),
                    target_shape
                );
            }
        }

        // Perform the broadcasting
        let broadcasted_data = self
            .data
            .clone()
            .broadcast(IxDyn(&target_shape))
            .expect("Broadcast failed")
            .to_owned();

        Tensor {
            data: broadcasted_data,
        }
    }

    /// Normalizes the tensor to a specified range.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the range.
    /// * `max` - The maximum value of the range.
    ///
    /// # Returns
    ///
    /// A new tensor containing the normalized data.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let normalized_tensor = tensor.normalize(0.0, 1.0);
    pub fn normalize(&self, min: f32, max: f32) -> Tensor {
        let normalized_data = self.data.mapv(|x| (x - min) / (max - min));
        Tensor {
            data: normalized_data,
        }
    }

    /// Adds noise to the tensor.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - The level of noise to add.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut tensor = Tensor::new(data, vec![2, 2]);
    /// tensor.add_noise(0.1);
    /// ```
    pub fn add_noise(&mut self, noise_level: f32) {
        let mut rng = rand::thread_rng();
        for value in self.data.iter_mut() {
            let noise: f32 = rng.gen_range(-noise_level..noise_level);
            *value += noise;
        }
    }

    /// Reduces the tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to reduce along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the reduced data.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let reduced_tensor = tensor.reduce_sum(1);
    /// ```
    pub fn reduce_sum(&self, axis: usize) -> Tensor {
        let sum = self.data.sum_axis(Axis(axis));
        Tensor { data: sum }
    }

    /// Gets the index of the maximum value along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to find the maximum along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the indices of the maximum values.
    ///
    /// # Panics
    ///
    /// Panics if the axis is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![2, 2]);
    /// let max_indices = tensor.argmax(1);
    /// ```
    pub fn argmax(&self, axis: usize) -> Tensor {
        // Ensure the axis is valid
        if axis >= self.data.ndim() {
            panic!(
                "Axis {} is out of bounds for tensor with shape {:?}",
                axis,
                self.shape()
            );
        }

        // Compute the indices of the maximum values along the specified axis
        let max_indices = self
            .data
            .map_axis(Axis(axis), |subview| {
                subview
                    .indexed_iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap() as f32 // Store indices as f32
            })
            .into_dyn(); // Convert to dynamic dimensionality

        Tensor { data: max_indices }
    }

    /// Takes elements from the tensor according to the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of indices to take.
    ///
    /// # Returns
    ///
    /// A new tensor containing the selected elements.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::new(data, vec![4]);
    /// let indices = vec![2, 0, 3, 1];
    /// let selected = tensor.take(&indices);
    /// ```
    pub fn take(&self, indices: &[usize]) -> Tensor {
        let mut data = Vec::with_capacity(indices.len());
        let mut shape = self.shape();
        shape[0] = indices.len();

        // Flatten the tensor to make indexing easier
        let flat_data: Vec<f32> = self.data.iter().cloned().collect();
        let stride = flat_data.len() / self.shape()[0];

        // Take elements according to indices
        for &idx in indices {
            let start = idx * stride;
            let end = start + stride;
            data.extend_from_slice(&flat_data[start..end]);
        }

        Tensor::new(data, shape)
    }

    /// Converts the tensor data to a vector.
    ///
    /// # Returns
    ///
    /// A vector containing the tensor data in row-major order.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.as_slice().unwrap_or(&[]).to_vec()
    }

    /// Creates a tensor from image bytes.
    ///
    /// # Arguments
    ///
    /// * `image_bytes` - The bytes of the image.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the image pixel data in the shape `(height, width, channels)`.
    pub fn from_image_bytes(image_bytes: Vec<u8>) -> Result<Self, String> {
        // Decode the image from bytes
        let image = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| format!("Failed to read image: {}", e))?
            .decode()
            .map_err(|e| format!("Failed to decode image: {}", e))?;

        // Get image dimensions and pixel data
        let (width, height) = image.dimensions();
        let pixel_data = image.to_rgba8().into_raw(); // Convert to RGBA and flatten the pixel data

        // Construct the Tensor with shape (height, width, 4)
        Ok(Tensor::new(
            pixel_data.iter().map(|&x| x as f32 / 255.0).collect(), // Normalize pixel values
            vec![height as usize, width as usize, 4],               // Shape (H, W, C)
        ))
    }

    /// Stacks multiple tensors along a new axis.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A slice of tensors to stack.
    ///
    /// # Returns
    ///
    /// A new tensor containing the stacked tensors.
    ///
    /// # Panics
    ///
    /// Panics if the tensors do not have the same shape.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    /// let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    /// let stacked = Tensor::stack(&[tensor1, tensor2]).unwrap();
    /// assert_eq!(stacked.shape(), vec![2, 3]);
    /// ```
    pub fn stack(tensors: &[Tensor]) -> Result<Tensor, String> {
        if tensors.is_empty() {
            return Err("Cannot stack an empty list of tensors.".to_string());
        }

        // Ensure all tensors have the same shape
        let first_shape = tensors[0].shape();
        for tensor in tensors {
            if tensor.shape() != first_shape {
                return Err(format!(
                    "All tensors must have the same shape. Expected {:?}, got {:?}",
                    first_shape,
                    tensor.shape()
                ));
            }
        }

        // Stack tensors along a new axis
        let stacked_data = ndarray::stack(
            Axis(0),
            &tensors.iter().map(|t| t.data.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;

        Ok(Tensor {
            data: stacked_data.into_dyn(),
        })
    }
}

impl SubAssign for Tensor {
    /// Subtracts another tensor from the current tensor in-place.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The tensor to subtract from the current tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let data2 = vec![4.0, 3.0, 2.0, 1.0];
    /// let mut tensor1 = Tensor::new(data1, vec![2, 2]);
    /// let tensor2 = Tensor::new(data2, vec![2, 2]);
    /// tensor1 -= tensor2;
    /// ```
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= &rhs.data;
    }
}

impl Default for Tensor {
    /// Creates a new tensor with default values.
    ///
    /// # Returns
    ///
    /// A new tensor with default values.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let tensor = Tensor::default();
    /// ```
    fn default() -> Self {
        Self::zeros(vec![1, 1])
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    /// Multiplies two tensors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The tensor to multiply with the current tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the multiplication.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let data2 = vec![2.0, 3.0, 4.0, 5.0];
    /// let tensor1 = Tensor::new(data1, vec![2, 2]);
    /// let tensor2 = Tensor::new(data2, vec![2, 2]);
    /// let multiplied_tensor = tensor1 * tensor2;
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        self.matmul(&rhs)
    }
}

impl PartialEq for Tensor {
    /// Checks if two tensors are equal.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to compare with.
    ///
    /// # Returns
    ///
    /// `true` if the tensors are equal, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use deltaml::common::Tensor;
    ///
    /// let data1 = vec![1.0, 2.0, 3.0, 4.0];
    /// let data2 = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor1 = Tensor::new(data1, vec![2, 2]);
    /// let tensor2 = Tensor::new(data2, vec![2, 2]);
    /// let is_equal = tensor1 == tensor2;
    /// ```
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Index<(usize, usize, usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let (b, i, j, f) = index;
        &self.data[b * self.shape()[1] * self.shape()[2] * self.shape()[3] + i * self.shape()[2] * self.shape()[3] + j * self.shape()[3] + f]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize, usize)) -> &mut Self::Output {
        let (b, i, j, f) = index;
        let shape = self.shape(); // Store the shape in a separate variable
        &mut self.data[b * shape[1] * shape[2] * shape[3] + i * shape[2] * shape[3] + j * shape[3] + f]
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Shapes must be equal for addition");
        let data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Tensor {
            data: ArrayD::from_shape_vec(self.shape().clone(), data).expect("Invalid shape for addition"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 1];
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.data.shape(), &[3, 1]);
    }

    #[test]
    fn test_zeros() {
        let shape = vec![2, 3];
        let tensor = Tensor::zeros(shape);
        assert_eq!(tensor.data.shape(), &[2, 3]);
    }

    #[test]
    fn test_random() {
        let shape = vec![2, 3];
        let tensor = Tensor::random(shape);
        assert_eq!(tensor.data.shape(), &[2, 3]);
    }

    #[test]
    fn test_add() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]);
        let result = tensor1.add(tensor2);
        assert_eq!(result.data.shape(), &[3, 1]);
    }

    #[test]
    fn test_max() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, vec![3, 1]);
        assert_eq!(tensor.max(), 3.0);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, vec![3, 1]);
        assert_eq!(tensor.mean(), 2.0);
    }

    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, vec![3, 1]);
        let reshaped = tensor.reshape(vec![1, 3]);
        assert_eq!(reshaped.data.shape(), &[1, 3]);
    }

    #[test]
    fn test_map() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, vec![3, 1]);
        let mapped = tensor.map(|x| x * 2.0);
        assert_eq!(mapped.data.shape(), &[3, 1]);
    }

    #[test]
    fn test_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, vec![3, 1]);
        let sliced = tensor.slice(vec![0..2, 0..1]);
        assert_eq!(sliced.data.shape(), &[2, 1]);
    }

    #[test]
    fn test_matmul() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(data1, vec![2, 2]);
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let tensor2 = Tensor::new(data2, vec![2, 2]);
        let result = tensor1.matmul(&tensor2);
        assert_eq!(result.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let transposed = tensor.transpose();
        assert_eq!(transposed.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_permute() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let permuted = tensor.permute(vec![1, 0]);
        assert_eq!(permuted.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_sum_along_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let summed = tensor.sum_along_axis(1);
        assert_eq!(summed.data.shape(), &[2]);
    }

    #[test]
    fn test_mul_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let multiplied = tensor.mul_scalar(2.0);
        assert_eq!(multiplied.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_pow() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let powered = tensor.pow(2.0);
        assert_eq!(powered.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_div_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let divided = tensor.div_scalar(2.0);
        assert_eq!(divided.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_sqrt() {
        let data = vec![1.0, 4.0, 9.0, 16.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let sqrted = tensor.sqrt();
        assert_eq!(sqrted.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_add_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let added = tensor.add_scalar(2.0);
        assert_eq!(added.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_div() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(data1, vec![2, 2]);
        let data2 = vec![2.0, 4.0, 6.0, 8.0];
        let tensor2 = Tensor::new(data2, vec![2, 2]);
        let divided = tensor1.div(&tensor2);
        assert_eq!(divided.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_flatten() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let flattened = tensor.flatten();
        assert_eq!(flattened.data.shape(), &[4]);
    }

    #[test]
    fn test_mean_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let meaned = tensor.mean_axis(1);
        assert_eq!(meaned.data.shape(), &[2]);
    }

    #[test]
    fn test_broadcast() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let broadcasted = tensor.broadcast(vec![2, 2]);
        assert_eq!(broadcasted.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]);
        let normalized = tensor.normalize(0.0, 1.0);
        assert_eq!(normalized.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_default() {
        let tensor = Tensor::default();
        assert_eq!(tensor.data.shape(), &[1, 1]);
    }

    #[test]
    fn test_add_noise() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(data, vec![2, 2]);
        tensor.add_noise(0.1);
        assert_eq!(tensor.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 5.0, 0.0];
        let tensor = Tensor::new(data, vec![2, 3]);

        let argmax = tensor.argmax(1);

        assert_eq!(argmax.data.shape(), &[2]);
        assert_eq!(
            argmax.data.iter().cloned().collect::<Vec<f32>>(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn test_mul_operator() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(data1, vec![2, 2]);
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let tensor2 = Tensor::new(data2, vec![2, 2]);
        let result = tensor1 * tensor2;
        assert_eq!(result.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_stack() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let stacked = Tensor::stack(&[tensor1, tensor2]).unwrap();
        assert_eq!(stacked.shape(), vec![2, 3]);
    }
}
