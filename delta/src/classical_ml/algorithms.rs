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

use std::ops::SubAssign;

use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use super::{Algorithm, batch_gradient_descent, logistic_gradient_descent, losses::Loss};

/// A struct for performing linear regression.
///
/// # Generics
/// - `T`: The type of data, must implement `num_traits::Float` and `ndarray::ScalarOperand`.
/// - `L`: The type of the loss function, must implement `Loss<T>`.
pub struct LinearRegression<T, L>
where
    T: Float,
    L: Loss<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
}

impl<T, L> LinearRegression<T, L>
where
    T: Float + ScalarOperand,
    L: Loss<T>,
{
    /// Calculates the loss between predictions and actual values.
    ///
    /// # Arguments
    /// - `predictions`: Predicted values as a 1D array.
    /// - `actuals`: Actual values as a 1D array.
    ///
    /// # Returns
    /// The calculated loss as a value of type `T`.
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }
}

impl<T, L> Algorithm<T, L> for LinearRegression<T, L>
where
    T: Float + ScalarOperand + SubAssign,
    L: Loss<T>,
{
    /// Creates a new `LinearRegression` instance with the given loss function and optimizer.
    ///
    /// # Arguments
    /// - `loss_function`: The loss function to use.
    /// - `optimizer`: The optimizer to use.
    ///
    /// # Returns
    /// A new instance of `LinearRegression`.
    fn new(loss_function: L) -> Self {
        LinearRegression { weights: Array1::zeros(1), bias: T::zero(), loss_function }
    }

    /// Fits the model to the given data using batch gradient descent.
    ///
    /// # Arguments
    /// - `x`: The input features as a 2D array.
    /// - `y`: The target values as a 1D array.
    /// - `learning_rate`: The learning rate for gradient descent.
    /// - `epochs`: The number of iterations for gradient descent.
    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let _loss = self.calculate_loss(&predictions, y);

            let (grad_weights, grad_bias) = batch_gradient_descent(x, y, &self.weights, self.bias);

            self.weights -= &(grad_weights * learning_rate);
            self.bias -= grad_bias * learning_rate;
        }
    }

    /// Predicts target values for the given input features.
    ///
    /// # Arguments
    /// - `x`: The input features as a 2D array.
    ///
    /// # Returns
    /// Predicted values as a 1D array.
    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }
}

/// A struct for performing logistic regression.
///
/// # Generics
/// - `T`: The type of data, must implement `num_traits::Float` and `ndarray::ScalarOperand`.
/// - `L`: The type of the loss function, must implement `Loss<T>`.
pub struct LogisticRegression<T, L>
where
    T: Float,
    L: Loss<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
}

impl<T, L> LogisticRegression<T, L>
where
    T: Float + ScalarOperand,
    L: Loss<T>,
{
    /// Calculates the loss between predictions and actual values.
    ///
    /// # Arguments
    /// - `predictions`: Predicted probabilities as a 1D array.
    /// - `actuals`: Actual values as a 1D array.
    ///
    /// # Returns
    /// The calculated loss as a value of type `T`.
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }

    // TODO: we should create generics for Activation
    // /// Applies the sigmoid function to the given linear output.
    ///
    /// # Arguments
    /// - `linear_output`: A 1D array of linear outputs.
    ///
    /// # Returns
    /// A 1D array of sigmoid-transformed values.
    fn sigmoid(&self, linear_output: Array1<T>) -> Array1<T> {
        linear_output.mapv(|x| T::one() / (T::one() + (-x).exp()))
    }

    /// Predicts the output of a logistic regression model.
    ///
    /// This function performs the linear regression prediction by calculating the dot product
    /// of the input features `x` and the model weights, and then adding the bias term.
    ///
    /// # Parameters
    /// - `x`: A 2D array of input features (`Array2<T>`). Each row represents a feature vector for a single sample.
    ///
    /// # Returns
    /// - An `Array1<T>` containing the predicted values for each sample.
    fn predict_linear(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }

    /// Calculates the accuracy of a binary classification model.
    ///
    /// This function compares the predicted values with the actual values, considering a threshold of 0.5
    /// to determine binary classifications. It returns the proportion of correct predictions, where predictions
    /// that differ from the actual values by less than `T::epsilon()` are considered correct.
    ///
    /// # Parameters
    /// - `predictions`: A 1D array (`Array1<T>`) containing the model's predicted values.
    /// - `actuals`: A 1D array (`Array1<T>`) containing the actual ground truth values.
    ///
    /// # Returns
    /// - A `f64` representing the accuracy, calculated as the ratio of correct predictions to total samples.
    ///
    /// # Constraints
    /// - `T` must implement `num_traits::Float` so that numerical operations like comparisons and arithmetic can be performed.
    pub fn calculate_accuracy(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> f64
    where
        T: Float,
    {
        let binary_predictions =
            predictions.mapv(|x| if x >= T::from(0.5).unwrap() { T::one() } else { T::zero() });
        let matches = binary_predictions
            .iter()
            .zip(actuals.iter())
            .filter(|(pred, actual)| (**pred - **actual).abs() < T::epsilon())
            .count();
        matches as f64 / actuals.len() as f64
    }
}

impl<T, L> Algorithm<T, L> for LogisticRegression<T, L>
where
    T: Float + ScalarOperand + SubAssign,
    L: Loss<T>,
{
    /// Creates a new `LogisticRegression` model.
    ///
    /// This constructor initializes a logistic regression model with a given loss function.
    /// The weights are initialized to zeros, and the bias is set to `T::zero()`.
    ///
    /// # Parameters
    /// - `loss_function`: The loss function to use for model training.
    ///
    /// # Returns
    /// - A new `LogisticRegression` instance with initialized weights and bias.
    fn new(loss_function: L) -> Self {
        LogisticRegression { weights: Array1::zeros(1), bias: T::zero(), loss_function }
    }

    /// Fits the logistic regression model to the training data using gradient descent.
    ///
    /// This function trains the model by iteratively updating the weights and bias to minimize
    /// the loss function. It performs a specified number of epochs of gradient descent with
    /// a given learning rate.
    ///
    /// # Parameters
    /// - `x`: A 2D array of input features (`Array2<T>`), where each row represents a sample.
    /// - `y`: A 1D array of target labels (`Array1<T>`) corresponding to the input samples.
    /// - `learning_rate`: The learning rate used in gradient descent.
    /// - `epochs`: The number of iterations (epochs) to run the gradient descent.
    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        for _ in 0..epochs {
            let linear_output = self.predict_linear(x);
            let predictions = self.sigmoid(linear_output.clone());
            let _loss = self.calculate_loss(&predictions, y);

            let _errors = &predictions - y;

            let (grad_weights, grad_bias) =
                logistic_gradient_descent(x, y, &self.weights, self.bias);

            self.weights -= &(grad_weights * learning_rate);
            self.bias -= grad_bias * learning_rate;
        }
    }

    /// Makes predictions using the logistic regression model.
    ///
    /// This function first calculates the linear output by applying the learned weights and bias,
    /// and then applies the sigmoid function to obtain the predicted probabilities.
    ///
    /// # Parameters
    /// - `x`: A 2D array of input features (`Array2<T>`), where each row represents a sample.
    ///
    /// # Returns
    /// - A 1D array (`Array1<T>`) containing the predicted probabilities for each sample.
    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        let linear_output = self.predict_linear(x);
        self.sigmoid(linear_output)
    }
}

enum TreeNode<T> {
    Internal { feature: usize, threshold: T, left: Box<TreeNode<T>>, right: Box<TreeNode<T>> },
    Leaf { prediction: T },
}

/// A struct for performing Decision Tree classification.
///
/// # Generics
/// - `T`: The type of data, must implement `num_traits::Float` and `ndarray::ScalarOperand`.
/// - `L`: The type of the loss function, must implement `Loss<T>`.
pub struct DecisionTree<T, L>
where
    T: Float,
    L: Loss<T> + Clone,
{
    max_depth: usize,
    loss_function: L,
    tree: Option<TreeNode<T>>,
}

impl<T, L> DecisionTree<T, L>
where
    T: Float + FromPrimitive,
    L: Loss<T> + Clone,
{
    /// Creates a new `DecisionTree` instance.
    ///
    /// # Arguments
    /// - `max_depth`: The maximum depth of the tree.
    /// - `loss_function`: The loss function to use.
    ///
    /// # Returns
    /// A new instance of `DecisionTree`.
    pub fn new(max_depth: usize, loss_function: L) -> Self {
        DecisionTree { max_depth, loss_function, tree: None }
    }

    /// Recursively splits the data based on the best feature and threshold.
    fn build_tree(&mut self, x: &Array2<T>, y: &Array1<T>, depth: usize) {
        if depth >= self.max_depth || x.shape()[0] <= 1 {
            // Create a leaf node
            let prediction = y.mean().unwrap();
            self.tree = Some(TreeNode::Leaf { prediction });
            return;
        }

        // Find the best feature and threshold
        let (best_feature, best_threshold) = self.find_best_split(x, y);

        // Split the data
        let (left_x, left_y, right_x, right_y) =
            self.split_data(x, y, best_feature, best_threshold);

        // Recursively build the left and right subtrees
        let mut left_tree = Box::new(DecisionTree::new(self.max_depth, self.loss_function.clone()));
        left_tree.build_tree(&left_x, &left_y, depth + 1);

        let mut right_tree =
            Box::new(DecisionTree::new(self.max_depth, self.loss_function.clone()));
        right_tree.build_tree(&right_x, &right_y, depth + 1);

        self.tree = Some(TreeNode::Internal {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(left_tree.tree.take().unwrap()),
            right: Box::new(right_tree.tree.take().unwrap()),
        });
    }

    fn find_best_split(&self, x: &Array2<T>, y: &Array1<T>) -> (usize, T) {
        // Implement logic to find the best feature and threshold based on a splitting criterion
        // For simplicity, this is a placeholder.
        (0, T::zero())
    }

    fn split_data(
        &self,
        x: &Array2<T>,
        y: &Array1<T>,
        feature: usize,
        threshold: T,
    ) -> (Array2<T>, Array1<T>, Array2<T>, Array1<T>) {
        // Implement logic to split the data based on a feature and threshold
        // For simplicity, this is a placeholder.
        (x.clone(), y.clone(), x.clone(), y.clone())
    }
}

impl<T, L> Algorithm<T, L> for DecisionTree<T, L>
where
    T: Float + FromPrimitive,
    L: Loss<T> + Clone,
{
    fn new(loss_function: L) -> Self {
        DecisionTree { max_depth: 10, loss_function, tree: None }
    }

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, _learning_rate: T, _epochs: usize) {
        self.build_tree(x, y, 0);
    }

    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        // Implement the prediction logic by traversing the tree
        Array1::zeros(x.shape()[0]) // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_traits::Float;

    use crate::classical_ml::{
        Algorithm,
        algorithms::{LinearRegression, LogisticRegression},
        losses::{CrossEntropy, MSE},
    };

    #[test]
    fn test_linear_regression_fit_predict() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let mut model = LinearRegression::new(MSE);

        let learning_rate = 0.1;
        let epochs = 1000;
        model.fit(&x_data, &y_data, learning_rate, epochs);

        let new_data = Array2::from_shape_vec((2, 1), vec![5.0, 6.0]).unwrap();
        let predictions = model.predict(&new_data);

        assert!((predictions[0] - 10.0).abs() < 1e-2);
        assert!((predictions[1] - 12.0).abs() < 1e-2);
    }

    #[test]
    fn test_linear_regression_calculate_loss() {
        let predictions = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let actuals = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let model = LinearRegression::new(MSE);

        let loss = model.calculate_loss(&predictions, &actuals);
        assert!(loss.abs() < 1e-6, "Loss should be close to 0, got: {}", loss);
    }

    #[test]
    fn test_logistic_regression_fit_predict() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(CrossEntropy);

        let learning_rate = 0.1;
        let epochs = 1000;
        model.fit(&x_data, &y_data, learning_rate, epochs);

        let new_data = Array2::from_shape_vec((2, 1), vec![1.5, 3.5]).unwrap();
        let predictions = model.predict(&new_data);

        assert!(predictions[0] >= 0.0 && predictions[0] <= 1.0);
        assert!(predictions[1] >= 0.0 && predictions[1] <= 1.0);
    }

    #[test]
    fn test_logistic_regression_calculate_loss() {
        let predictions = Array1::from_vec(vec![0.1, 0.2, 0.7, 0.9]);
        let actuals = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let model = LogisticRegression::new(CrossEntropy);

        let loss = model.calculate_loss(&predictions, &actuals);
        assert!(loss > 0.0, "Loss should be positive, got: {}", loss);
    }

    #[test]
    fn test_logistic_regression_calculate_accuracy() {
        let predictions = Array1::from_vec(vec![0.1, 0.8, 0.3, 0.7]);
        let actuals = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let model = LogisticRegression::new(CrossEntropy);

        let accuracy = model.calculate_accuracy(&predictions, &actuals);
        assert!((accuracy - 0.5).abs() < 1e-6, "Accuracy should be 0.5, got: {}", accuracy);
    }
}
