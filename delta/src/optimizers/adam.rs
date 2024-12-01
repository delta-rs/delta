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

use crate::common::optimizer::Optimizer;
use crate::common::tensor_ops::Tensor;
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
    m: Option<Tensor>,
    v: Option<Tensor>,
    timestep: usize,
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
            m: None,
            v: None,
            timestep: 0,
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
    /// use deltaml::optimizers::adam::Adam;
    ///
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
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, _weights: &mut Tensor, _gradients: &Tensor) {
        // self.timestep += 1;

        // // Ensure m and v are initialized with the correct shape
        // if self.m.is_none() || self.m.as_ref().unwrap().shape() != weights.shape() {
        //     self.m = Some(Tensor::zeros(weights.shape().clone()));
        // }
        // if self.v.is_none() || self.v.as_ref().unwrap().shape() != weights.shape() {
        //     self.v = Some(Tensor::zeros(weights.shape().clone()));
        // }

        // let m = self.m.as_mut().unwrap();
        // let v = self.v.as_mut().unwrap();

        // println!("Shape of weights: {:?}", weights.shape());
        // println!("Shape of gradients: {:?}", gradients.shape());

        // // Process gradients to match weights' shape
        // let processed_gradients = if gradients.shape() != weights.shape() {
        //     if gradients.data.len() == weights.data.len() {
        //         // Reshape gradients if the total number of elements matches
        //         gradients.reshape(weights.shape())
        //     } else {
        //         // Handle incompatible shapes, e.g., by reducing batch or spatial dimensions
        //         gradients.mean_axis(0)
        //     }
        // } else {
        //     gradients.clone()
        // };

        // println!(
        //     "Shape of processed gradients: {:?}",
        //     processed_gradients.shape()
        // );

        // // Update moving averages of gradients and squared gradients
        // let gradients = &processed_gradients.mul_scalar(0.1);
        // let m_new = m.mul_scalar(0.9).add(gradients);
        // let v_new = v
        //     .mul_scalar(0.999)
        //     .add(&gradients.pow(2.0).mul_scalar(0.001));

        // // Bias correction
        // let bias_correction_1 = 1.0 - 0.9f32.powi(self.timestep as i32);
        // let bias_correction_2 = 1.0 - 0.999f32.powi(self.timestep as i32);

        // let m_hat = m_new.div_scalar(bias_correction_1);
        // let v_hat = v_new.div_scalar(bias_correction_2);

        // // Update weights
        // let lr = self
        //     .scheduler
        //     .as_ref()
        //     .map(|scheduler| scheduler.0(self.timestep))
        //     .unwrap_or(self.learning_rate);

        // let update = m_hat.div(&v_hat.sqrt().add_scalar(1e-8));
        // *weights -= update.mul_scalar(lr);

        // // Save updated moments
        // *m = m_new;
        // *v = v_new;
    }
}
