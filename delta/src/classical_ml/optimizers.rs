use std::ops::{Mul, SubAssign};

use ndarray::{Array1, ScalarOperand};
use num_traits::Float;

pub trait Optimizer<T> {
    fn update(
        &mut self,
        weights: &mut Array1<T>,
        bias: &mut T,
        gradients: &Array1<T>,
        bias_gradient: T,
        learning_rate: T,
    );
}

pub struct BatchGradientDescent;

impl<T> Optimizer<T> for BatchGradientDescent
where
    T: Float + SubAssign + Mul<Output = T> + ScalarOperand,
{
    fn update(
        &mut self,
        weights: &mut Array1<T>,
        bias: &mut T,
        gradients: &Array1<T>,
        bias_gradient: T,
        learning_rate: T,
    ) {
        *weights -= &(gradients * learning_rate);
        *bias -= bias_gradient * learning_rate;
    }
}
