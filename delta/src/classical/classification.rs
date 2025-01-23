use ndarray::{Array1, Array2};

use crate::classical::{calculate_loss, gradient_descent};

use super::Classical;

pub struct LinearRegression {
    weights: Array1<f64>,
    bias: f64,
}

impl Classical for LinearRegression {
    fn new() -> Self {
        LinearRegression { weights: Array1::zeros(1), bias: 0.0 }
    }

    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let loss = calculate_loss(&predictions, y);
            // Using Batch Gradient Descent here, we might want the user to have the option
            // to change optimizer such as SGD, Adam etc
            let gradients = gradient_descent(x, y, &self.weights, self.bias);

            self.weights = &self.weights - &(gradients.0 * learning_rate);
            self.bias -= gradients.1 * learning_rate;

            println!("Loss: {}", loss);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.weights) + self.bias
    }
}
