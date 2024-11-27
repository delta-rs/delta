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
use delta_common::Optimizer;
use std::fmt;
use std::fmt::Debug;

/// A wrapper struct for a debuggable scheduler function.
#[allow(dead_code)]
struct DebuggableScheduler(Box<dyn Fn(usize) -> f32>);

impl Debug for DebuggableScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DebuggableScheduler")
    }
}

/// The Adam optimizer struct.
#[derive(Debug)]
pub struct Adam {
    #[allow(dead_code)]
    learning_rate: f32,
    scheduler: Option<DebuggableScheduler>,
}

impl Adam {
    /// Creates a new Adam optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the Adam optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            scheduler: None,
        }
    }

    /// Sets the scheduler function for the Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `scheduler` - A function that takes an epoch number and returns a learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use delta_optimizers::Adam;
    /// let mut optimizer = Adam::new(0.001);
    /// optimizer.set_scheduler(|epoch| 0.001 * (0.9f32.powi(epoch as i32)));
    /// ```
    pub fn set_scheduler<F>(&mut self, scheduler: F)
    where
        F: Fn(usize) -> f32 + 'static,
    {
        self.scheduler = Some(DebuggableScheduler(Box::new(scheduler)));
    }
}

impl Optimizer for Adam {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `gradients` - A mutable slice of tensors representing the gradients.
    fn step(&mut self, gradients: &mut [Tensor]) {
        let _ = gradients;
        todo!()
    }
}
