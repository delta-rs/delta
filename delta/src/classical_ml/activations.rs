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

use ndarray::Array1;
use num_traits::Float;

pub trait Activation<T>
where
    T: Float,
{
    fn activate(&self, input: &Array1<T>) -> Array1<T>;

    fn derivative(&self, input: &Array1<T>) -> Array1<T>;
}

pub struct Sigmoid;
pub struct ReLU;
pub struct Tanh;

impl<T> Activation<T> for Sigmoid
where
    T: Float,
{
    fn activate(&self, input: &Array1<T>) -> Array1<T> {
        input.mapv(|x| T::one() / (T::one() + (-x).exp()))
    }

    fn derivative(&self, input: &Array1<T>) -> Array1<T> {
        let activated = self.activate(input);
        activated.mapv(|x| x * (T::one() - x))
    }
}

impl<T> Activation<T> for ReLU
where
    T: Float,
{
    fn activate(&self, input: &Array1<T>) -> Array1<T> {
        input.mapv(|x| if x > T::zero() { x } else { T::zero() })
    }

    fn derivative(&self, input: &Array1<T>) -> Array1<T> {
        input.mapv(|x| if x > T::zero() { T::one() } else { T::zero() })
    }
}

impl<T> Activation<T> for Tanh
where
    T: Float,
{
    fn activate(&self, input: &Array1<T>) -> Array1<T> {
        input.mapv(|x| x.tanh())
    }

    fn derivative(&self, input: &Array1<T>) -> Array1<T> {
        input.mapv(|x| T::one() - x.tanh().powi(2))
    }
}
