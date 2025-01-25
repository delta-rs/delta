use std::iter::Sum;

use ndarray::Array1;
use num_traits::Float;

pub struct MSE;
pub struct CrossEntropy;

pub trait Loss<T> {
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T;
}

impl<T> Loss<T> for MSE
where
    T: Float,
{
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let diff = predictions - actuals;
        (diff.mapv(|x| x.powi(2)).sum()) / m
    }
}

impl<T> Loss<T> for CrossEntropy
where
    T: num_traits::Float + ndarray::ScalarOperand + Sum,
{
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let epsilon = T::from(1e-15).unwrap();

        predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, y)| {
                let p_clamped = p.max(epsilon).min(T::one() - epsilon);
                -(y.clone() * p_clamped.ln() + (T::one() - y.clone()) * (T::one() - p_clamped).ln())
            })
            .sum::<T>()
            / m
    }
}
