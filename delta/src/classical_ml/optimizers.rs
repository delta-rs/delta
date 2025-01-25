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
pub struct LogisticGradientDescent;

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

impl<T> Optimizer<T> for LogisticGradientDescent
where
    T: Float + SubAssign + Mul<Output = T> + ScalarOperand,
{
    fn update(
        &mut self,
        _weights: &mut Array1<T>,
        _bias: &mut T,
        _gradients: &Array1<T>,
        _bias_gradient: T,
        _learning_rate: T,
    ) {
    }
}
