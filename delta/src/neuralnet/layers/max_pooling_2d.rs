//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin, Chase Willden
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

use crate::common::shape::Shape;
use crate::common::tensor_ops::Tensor;
use crate::neuralnet::layers::error::LayerError;
use crate::neuralnet::layers::Layer;
use crate::optimizers::Optimizer;

#[derive(Debug)]
pub struct MaxPooling2D {
    #[allow(dead_code)]
    pool_size: usize,
    #[allow(dead_code)]
    stride: usize,
    input_shape: Option<Shape>,
}

impl MaxPooling2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self {
            pool_size,
            stride,
            input_shape: None,
        }
    }
}

impl Layer for MaxPooling2D {
    fn build(&mut self, input_shape: Shape) -> Result<(), LayerError> {
        self.input_shape = Some(input_shape);
        Ok(())
    }

    fn forward(&mut self, _input: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!()
    }

    fn backward(&mut self, _grad: &Tensor) -> Result<Tensor, LayerError> {
        unimplemented!()
    }

    fn output_shape(&self) -> Result<Shape, LayerError> {
        unimplemented!()
    }

    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        Ok((0, 0))
    }

    fn name(&self) -> &str {
        "MaxPooling2D"
    }

    fn update_weights(&mut self, _optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        unimplemented!()
    }
}
