//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin
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

use delta_common::tensor_ops::Tensor;
use delta_common::{Layer, Shape};

#[derive(Debug)]
pub struct Dense {
    weights: Tensor,
    bias: Tensor,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Tensor::random(&Shape::from((input_size, output_size))),
            bias: Tensor::zeros(&Shape::new(vec![output_size])),
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights).add(&self.bias)
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        let _ = grad;
        todo!()
    }
}

#[derive(Debug)]
pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Relu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        let _ = grad;
        todo!()
    }
}
