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

pub mod layers;
pub mod models;

// used for operations on Tensors
pub mod functional; 

pub use layers::{Dense, Flatten};
pub use models::Sequential;

/// Putting tests here since it's using a collection of everything
#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use crate::activations::{ReluActivation, SoftmaxActivation};
    use crate::losses::MeanSquaredLoss;
    use crate::neuralnet::{Dense, Flatten, Sequential};
    use crate::optimizers::Adam;

    #[test]
    fn test_sequential_new() {
        let model = Sequential::new();
        assert!(model.layers.is_empty());
    }

    #[test]
    fn test_sequential_add() {
        let model = Sequential::new()
            .add(Flatten::new(Shape::from(IxDyn(&[28, 28]))))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));
        assert_eq!(model.layers.len(), 3);
    }

    #[test]
    fn test_sequential_compile() {
        let mut model = Sequential::new()
            .add(Flatten::new(Shape::from(IxDyn(&[28, 28]))))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));

        model.compile(Adam::new(0.001), MeanSquaredLoss::new());

        assert!(model.optimizer.is_some());
        assert!(model.loss.is_some());
    }
}
