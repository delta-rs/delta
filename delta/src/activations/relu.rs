use crate::common::{activation::Activation, tensor_ops::Tensor};

/// A struct representing the Rectified Linear Unit (ReLU) activation function.
#[derive(Debug)]
pub struct ReluActivation;

impl ReluActivation {
    /// Creates a new instance of `ReluActivation`.
    pub fn new() -> Self {
        Self
    }
}

impl Activation for ReluActivation {
    /// Applies ReLU activation to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying ReLU activation.
    ///
    /// # Examples
    ///
    /// ```
    /// use deltaml::activations::relu::ReluActivation;
    /// use deltaml::common::Activation;
    /// use deltaml::common::tensor_ops::Tensor;
    ///
    /// let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
    /// let relu = ReluActivation::new();
    /// let output = relu.activate(&input);
    /// ```
    fn activate(&self, input: &Tensor) -> Tensor {
        input.map(|x| x.max(0.0))
    }

    /// Computes the derivative of ReLU activation for the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the derivative of ReLU activation.
    ///
    /// # Examples
    ///
    /// ```
    /// use deltaml::activations::relu::ReluActivation;
    /// use deltaml::common::Activation;
    /// use deltaml::common::tensor_ops::Tensor;
    ///
    /// let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
    /// let relu = ReluActivation::new();
    /// let derivative = relu.derivative(&input);
    /// ```
    fn derivative(&self, input: &Tensor) -> Tensor {
        input.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_activation() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        let relu = ReluActivation::new();
        let output = relu.activate(&input);

        assert_eq!(
            output.data.iter().cloned().collect::<Vec<f32>>(),
            vec![1.0, 0.0, 3.0, 0.0]
        );
        assert_eq!(output.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_relu_derivative() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        let relu = ReluActivation::new();
        let derivative = relu.derivative(&input);

        assert_eq!(
            derivative.data.iter().cloned().collect::<Vec<f32>>(),
            vec![1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(derivative.data.shape().to_vec(), vec![2, 2]);
    }
}
