//! BSD 3-Clause License
//!
//! Copyright (c) 2025, BlackPortal ○
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

pub use metal;
use metal::MTLResourceOptions;
use ndarray::{Array, IxDyn, Shape};

use crate::common::Tensor;

/// Transfers the tensor to a Metal device.
///
/// # Arguments
///
/// * `device` - The Metal device to transfer the tensor to.
///
/// # Returns
///
/// A new tensor with data stored on the Metal device.
pub fn to_device_metal(
    tensor: &Tensor,
    device: &metal::Device,
    _queue: &metal::CommandQueue,
) -> metal::Buffer {
    let size = tensor.data.len() * size_of::<f32>();
    let view = tensor.data.view();
    let flat_data = view.to_shape(tensor.data.len()).unwrap();

    let buffer = device.new_buffer_with_data(
        flat_data.as_slice().unwrap().as_ptr() as *const _,
        size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    buffer
}

/// Transfers the tensor's data back from a Metal buffer to the CPU.
///
/// # Arguments
///
/// * `buffer` - The Metal buffer containing the tensor's data.
///
/// # Returns
///
/// A new `Tensor` instance with the data transferred from the Metal device.
pub fn from_device_metal(buffer: &metal::Buffer, shape: Shape<IxDyn>) -> Tensor {
    // Create a vector to hold the data
    let tensor_size = buffer.length() as usize / size_of::<f32>();
    let mut data = vec![0.0; tensor_size];

    // Copy the data from the Metal buffer
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents() as *const f32,
            data.as_mut_ptr(),
            tensor_size,
        );
    }

    // Create a new tensor from the data
    Tensor::new(data, shape)
}

/// Creates a Metal device and command queue.
///
/// # Returns
///
/// A tuple containing the Metal device and command queue.
pub fn get_device_and_queue_metal() -> (metal::Device, metal::CommandQueue) {
    let device = metal::Device::system_default().expect("no device found");
    let queue = device.new_command_queue();
    (device, queue)
}

/// Creates a Metal compute pipeline state.
///
/// # Arguments
///
/// * `device` - The Metal device to create the pipeline state on.
/// * `shader_name` - The name of the Metal shader to use.
///
/// # Returns
///
/// A new `metal::ComputePipelineState` instance.
pub fn create_compute_pipeline(
    device: &metal::Device,
    shader_name: &str,
    function_name: &str,
) -> Result<metal::ComputePipelineState, String> {
    let library = device
        .new_library_with_source(shader_name, &metal::CompileOptions::new())
        .map_err(|e| format!("Failed to compile Metal shader: {:?}", e))?;
    let kernel = library
        .get_function(function_name, None)
        .map_err(|e| format!("Failed to get kernel function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Failed to create compute pipeline: {:?}", e))
}

/// Executes a tensor operation on a Metal device.
///
/// # Arguments
///
/// * `tensor1` - An optional reference to the first tensor.
/// * `tensor2` - An optional reference to the second tensor.
/// * `operation` - The name of the operation to perform.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
/// * `extra_data` - Optional extra data for the operation.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
fn execute_tensor_operation(
    tensor1: Option<&Tensor>,
    tensor2: Option<&Tensor>,
    operation: &str,
    device: &metal::Device,
    queue: &metal::CommandQueue,
    extra_data: Option<&[f32]>,
) -> Result<Tensor, String> {
    let shader_source = include_str!("metal_shaders/tensor_ops.metal");
    let pipeline_state = create_compute_pipeline(device, shader_source, operation)?;

    // Set the Tensor1 data into Metal buffers
    let input1_buffer = tensor1.map(|t| to_device_metal(t, device, queue));

    // Set the Tensor2 data into Metal buffers
    let input2_buffer = tensor2.map(|t| to_device_metal(t, device, queue));

    // Set the tensor_size, which is tensor1.len() or tensor2.len()
    let tensor_size = tensor1.map(|t| t.data.len()).unwrap_or_else(|| tensor2.unwrap().data.len());
    let output_buffer = device
        .new_buffer((tensor_size * size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);

    // Set the tensor_length into Metal buffers
    let tensor_length_buffer = device.new_buffer_with_data(
        &(tensor_size as u32) as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Used for power and other operations
    let extra_buffer = extra_data.map(|data| {
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);

    // Set the data to the buffers
    if let Some(buffer) = &input1_buffer {
        encoder.set_buffer(0, Some(buffer), 0);
    }
    if let Some(buffer) = &input2_buffer {
        encoder.set_buffer(1, Some(buffer), 0);
    }
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.set_buffer(3, Some(&tensor_length_buffer), 0);

    // Optional set the extra data buffer
    if let Some(buffer) = &extra_buffer {
        encoder.set_buffer(4, Some(buffer), 0);
    }

    // Calculating the thread group size and count
    let thread_group_size = metal::MTLSize { width: 256, height: 1, depth: 1 };
    let thread_group_count = metal::MTLSize {
        width: (tensor_size as u64).div_ceil(thread_group_size.width),
        height: 1,
        depth: 1,
    };

    // Run the kernel
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    // Commit the command buffer and wait for completion
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Extract the data from the kernel back to the CPU
    let output_data =
        unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const f32, tensor_size) }
            .to_vec();

    Ok(Tensor {
        data: Array::from_shape_vec(tensor1.unwrap().shape(), output_data).unwrap(),
        device: tensor1.unwrap().device.clone(),
    })
}

/// Adds two tensors using a Metal device.
///
/// # Arguments
///
/// * `tensor1` - A reference to the first tensor.
/// * `tensor2` - A reference to the second tensor.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_add_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    execute_tensor_operation(Some(tensor1), Some(tensor2), "tensor_add", device, queue, None)
}

/// Subtracts the second tensor from the first using a Metal device.
///
/// # Arguments
///
/// * `tensor1` - A reference to the first tensor.
/// * `tensor2` - A reference to the second tensor.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_subtract_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    execute_tensor_operation(Some(tensor1), Some(tensor2), "tensor_subtract", device, queue, None)
}

/// Multiplies two tensors using a Metal device.
///
/// # Arguments
///
/// * `tensor1` - A reference to the first tensor.
/// * `tensor2` - A reference to the second tensor.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_multiply_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    execute_tensor_operation(Some(tensor1), Some(tensor2), "tensor_multiply", device, queue, None)
}

/// Divides the first tensor by the second using a Metal device.
///
/// # Arguments
///
/// * `tensor1` - A reference to the first tensor.
/// * `tensor2` - A reference to the second tensor.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_divide_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    execute_tensor_operation(Some(tensor1), Some(tensor2), "tensor_divide", device, queue, None)
}

/// Raises the elements of a tensor to a power using a Metal device.
///
/// # Arguments
///
/// * `tensor` - A reference to the tensor.
/// * `amount` - The power to raise the tensor elements to.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_power_metal(
    tensor: &Tensor,
    amount: f32,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    execute_tensor_operation(Some(tensor), None, "tensor_power", device, queue, Some(&[amount]))
}

/// Multiplies two matrices using a Metal device.
///
/// # Arguments
///
/// * `tensor1` - A reference to the first tensor.
/// * `tensor2` - A reference to the second tensor.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_matmul_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    let tensor1_shape = tensor1.shape();
    let tensor2_shape = tensor2.shape();
    let tensor1_dim = tensor1_shape.raw_dim();
    let tensor2_dim = tensor2_shape.raw_dim();

    let rows_a = tensor1_dim[0];
    let cols_a = tensor1_dim[1];
    let cols_b = tensor2_dim[1];

    let shader_source = include_str!("metal_shaders/tensor_ops.metal");
    let pipeline_state = create_compute_pipeline(device, shader_source, "tensor_matmul")?;

    let input1_buffer = to_device_metal(tensor1, device, queue);
    let input2_buffer = to_device_metal(tensor2, device, queue);

    let output_size = rows_a * cols_b;
    let output_buffer = device
        .new_buffer((output_size * size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);

    let rows_a_buffer = device.new_buffer_with_data(
        &(rows_a as u32) as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cols_a_buffer = device.new_buffer_with_data(
        &(cols_a as u32) as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cols_b_buffer = device.new_buffer_with_data(
        &(cols_b as u32) as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);

    encoder.set_buffer(0, Some(&input1_buffer), 0);
    encoder.set_buffer(1, Some(&input2_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.set_buffer(3, Some(&rows_a_buffer), 0);
    encoder.set_buffer(4, Some(&cols_a_buffer), 0);
    encoder.set_buffer(5, Some(&cols_b_buffer), 0);

    let thread_group_size = metal::MTLSize { width: 256, height: 1, depth: 1 };
    let thread_group_count = metal::MTLSize {
        width: (output_size as u64).div_ceil(thread_group_size.width),
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(from_device_metal(&output_buffer, Shape::from(IxDyn(&[rows_a, cols_b]))))
}

/// Maps the maximum value of a tensor using a Metal device.
///
/// # Arguments
///
/// * `tensor` - A reference to the tensor.
/// * `threshold` - The threshold value for the mapping.
/// * `device` - The Metal device to use.
/// * `queue` - The Metal command queue to use.
///
/// # Returns
///
/// A `Result` containing the resulting tensor or an error message.
pub fn tensor_map_max_metal(
    tensor: &Tensor,
    threshold: f32,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    execute_tensor_operation(
        Some(tensor),
        None, // No second tensor is needed
        "tensor_map_max",
        device,
        queue,
        Some(&[threshold]),
    )
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;
    use crate::common::Tensor;

    #[test]
    fn test_tensor_add_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_add_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![6.0, 8.0, 10.0, 12.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_subtract_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_subtract_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![-4.0, -4.0, -4.0, -4.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_multiply_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_multiply_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![5.0, 12.0, 21.0, 32.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_divide_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_divide_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![0.2, 0.33333334, 0.42857146, 0.5], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_power_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let amount = 2.0;
        let result = tensor_power_metal(&tensor1, amount, &device, &queue).unwrap();

        let expected = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], Shape::from(IxDyn(&[2, 2])));

        // Apparently GPU floating point power operations are not exact
        let tolerance = 1e-5;

        assert!(
            result.data.iter().zip(expected.data.iter()).all(|(a, b)| (a - b).abs() < tolerance)
        );
    }

    #[test]
    fn test_tensor_matmul_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_matmul_metal(&tensor1, &tensor2, &device, &queue).unwrap();

        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![19.0, 22.0, 43.0, 50.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_map_max_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_map_max_metal(&tensor1, 0.0, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
    }
}
