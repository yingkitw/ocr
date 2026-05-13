//! Optimizers for neural network training

use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for optimizers
pub trait Optimizer {
    fn update(&mut self, parameters: &mut Array2<f32>, gradients: &Array2<f32>) -> Result<()>;
    fn name(&self) -> &'static str;
    fn get_state(&self) -> OptimizerState;
    fn set_state(&mut self, state: OptimizerState);
}

/// Optimizer state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub optimizer_type: String,
    pub parameters: HashMap<String, Vec<f32>>,
}

/// Stochastic Gradient Descent (SGD) optimizer
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: Option<Array2<f32>>,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: None,
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

impl Optimizer for SGD {
    fn update(&mut self, parameters: &mut Array2<f32>, gradients: &Array2<f32>) -> Result<()> {
        let mut gradients = gradients.clone();

        // Apply weight decay
        if self.weight_decay > 0.0 {
            gradients = &gradients + &(self.weight_decay * &*parameters);
        }

        // Apply momentum
        if self.momentum > 0.0 {
            if let Some(ref mut velocity) = self.velocity {
                *velocity = &(self.momentum * &*velocity) + &(self.learning_rate * &gradients);
                *parameters = &*parameters - &*velocity;
            } else {
                self.velocity = Some(self.learning_rate * &gradients);
                *parameters = &*parameters - &*self.velocity.as_ref().unwrap();
            }
        } else {
            *parameters = &*parameters - &(self.learning_rate * &gradients);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "SGD"
    }

    fn get_state(&self) -> OptimizerState {
        let mut parameters = HashMap::new();
        parameters.insert("learning_rate".to_string(), vec![self.learning_rate]);
        parameters.insert("momentum".to_string(), vec![self.momentum]);
        parameters.insert("weight_decay".to_string(), vec![self.weight_decay]);

        if let Some(ref velocity) = self.velocity {
            parameters.insert("velocity".to_string(), velocity.iter().cloned().collect());
        }

        OptimizerState {
            optimizer_type: "SGD".to_string(),
            parameters,
        }
    }

    fn set_state(&mut self, state: OptimizerState) {
        if let Some(lr) = state.parameters.get("learning_rate") {
            self.learning_rate = lr[0];
        }
        if let Some(momentum) = state.parameters.get("momentum") {
            self.momentum = momentum[0];
        }
        if let Some(weight_decay) = state.parameters.get("weight_decay") {
            self.weight_decay = weight_decay[0];
        }
        if let Some(velocity) = state.parameters.get("velocity") {
            if let Some(ref current_velocity) = self.velocity {
                let shape = current_velocity.shape();
                self.velocity =
                    Some(Array2::from_shape_vec((shape[0], shape[1]), velocity.clone()).unwrap());
            }
        }
    }
}

/// Adam optimizer
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: usize,
    m: Option<Array2<f32>>, // First moment estimate
    v: Option<Array2<f32>>, // Second moment estimate
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            step: 0,
            m: None,
            v: None,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

impl Optimizer for Adam {
    fn update(&mut self, parameters: &mut Array2<f32>, gradients: &Array2<f32>) -> Result<()> {
        let mut gradients = gradients.clone();

        // Apply weight decay
        if self.weight_decay > 0.0 {
            gradients = &gradients + &(self.weight_decay * &*parameters);
        }

        self.step += 1;

        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(Array2::zeros(gradients.raw_dim()));
            self.v = Some(Array2::zeros(gradients.raw_dim()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first moment estimate
        *m = &(self.beta1 * &*m) + &((1.0 - self.beta1) * &gradients);

        // Update biased second raw moment estimate
        *v = &(self.beta2 * &*v) + &((1.0 - self.beta2) * &gradients * &gradients);

        // Compute bias-corrected first moment estimate
        let m_hat = &*m / (1.0 - self.beta1.powi(self.step as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &*v / (1.0 - self.beta2.powi(self.step as i32));

        // Update parameters
        let v_hat_sqrt = v_hat.mapv(|x| x.sqrt());
        let denominator = &v_hat_sqrt + self.epsilon;
        let update = self.learning_rate * &m_hat / &denominator;
        *parameters = &*parameters - &update;

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Adam"
    }

    fn get_state(&self) -> OptimizerState {
        let mut parameters = HashMap::new();
        parameters.insert("learning_rate".to_string(), vec![self.learning_rate]);
        parameters.insert("beta1".to_string(), vec![self.beta1]);
        parameters.insert("beta2".to_string(), vec![self.beta2]);
        parameters.insert("epsilon".to_string(), vec![self.epsilon]);
        parameters.insert("weight_decay".to_string(), vec![self.weight_decay]);
        parameters.insert("step".to_string(), vec![self.step as f32]);

        if let Some(ref m) = self.m {
            parameters.insert("m".to_string(), m.iter().cloned().collect());
        }
        if let Some(ref v) = self.v {
            parameters.insert("v".to_string(), v.iter().cloned().collect());
        }

        OptimizerState {
            optimizer_type: "Adam".to_string(),
            parameters,
        }
    }

    fn set_state(&mut self, state: OptimizerState) {
        if let Some(lr) = state.parameters.get("learning_rate") {
            self.learning_rate = lr[0];
        }
        if let Some(beta1) = state.parameters.get("beta1") {
            self.beta1 = beta1[0];
        }
        if let Some(beta2) = state.parameters.get("beta2") {
            self.beta2 = beta2[0];
        }
        if let Some(epsilon) = state.parameters.get("epsilon") {
            self.epsilon = epsilon[0];
        }
        if let Some(weight_decay) = state.parameters.get("weight_decay") {
            self.weight_decay = weight_decay[0];
        }
        if let Some(step) = state.parameters.get("step") {
            self.step = step[0] as usize;
        }
        if let Some(m) = state.parameters.get("m") {
            if let Some(ref current_m) = self.m {
                let shape = current_m.shape();
                self.m = Some(Array2::from_shape_vec((shape[0], shape[1]), m.clone()).unwrap());
            }
        }
        if let Some(v) = state.parameters.get("v") {
            if let Some(ref current_v) = self.v {
                let shape = current_v.shape();
                self.v = Some(Array2::from_shape_vec((shape[0], shape[1]), v.clone()).unwrap());
            }
        }
    }
}

/// RMSprop optimizer
pub struct RMSprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    square_avg: Option<Array2<f32>>,
    momentum_buffer: Option<Array2<f32>>,
    grad_avg: Option<Array2<f32>>,
}

impl RMSprop {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            square_avg: None,
            momentum_buffer: None,
            grad_avg: None,
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn centered(mut self) -> Self {
        self.centered = true;
        self
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, parameters: &mut Array2<f32>, gradients: &Array2<f32>) -> Result<()> {
        let mut gradients = gradients.clone();

        // Apply weight decay
        if self.weight_decay > 0.0 {
            gradients = &gradients + &(self.weight_decay * &*parameters);
        }

        // Initialize buffers if needed
        if self.square_avg.is_none() {
            self.square_avg = Some(Array2::zeros(gradients.raw_dim()));
            if self.momentum > 0.0 {
                self.momentum_buffer = Some(Array2::zeros(gradients.raw_dim()));
            }
            if self.centered {
                self.grad_avg = Some(Array2::zeros(gradients.raw_dim()));
            }
        }

        let square_avg = self.square_avg.as_mut().unwrap();

        // Update square average
        *square_avg =
            &(self.alpha * &*square_avg) + &((1.0 - self.alpha) * &gradients * &gradients);

        let avg;
        if self.centered {
            let grad_avg = self.grad_avg.as_mut().unwrap();
            *grad_avg = &(self.alpha * &*grad_avg) + &((1.0 - self.alpha) * &gradients);
            avg = &*square_avg - &(grad_avg.clone() * &*grad_avg);
        } else {
            avg = square_avg.clone();
        }

        // Compute update
        let avg_sqrt = avg.mapv(|x| x.sqrt());
        let denominator = &avg_sqrt + self.epsilon;
        let update = self.learning_rate * &gradients / &denominator;

        // Apply momentum if enabled
        if self.momentum > 0.0 {
            let momentum_buffer = self.momentum_buffer.as_mut().unwrap();
            *momentum_buffer = &(self.momentum * &*momentum_buffer) + &update;
            *parameters = &*parameters - &*momentum_buffer;
        } else {
            *parameters = &*parameters - &update;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "RMSprop"
    }

    fn get_state(&self) -> OptimizerState {
        let mut parameters = HashMap::new();
        parameters.insert("learning_rate".to_string(), vec![self.learning_rate]);
        parameters.insert("alpha".to_string(), vec![self.alpha]);
        parameters.insert("epsilon".to_string(), vec![self.epsilon]);
        parameters.insert("weight_decay".to_string(), vec![self.weight_decay]);
        parameters.insert("momentum".to_string(), vec![self.momentum]);
        parameters.insert(
            "centered".to_string(),
            vec![if self.centered { 1.0 } else { 0.0 }],
        );

        if let Some(ref square_avg) = self.square_avg {
            parameters.insert(
                "square_avg".to_string(),
                square_avg.iter().cloned().collect(),
            );
        }
        if let Some(ref momentum_buffer) = self.momentum_buffer {
            parameters.insert(
                "momentum_buffer".to_string(),
                momentum_buffer.iter().cloned().collect(),
            );
        }
        if let Some(ref grad_avg) = self.grad_avg {
            parameters.insert("grad_avg".to_string(), grad_avg.iter().cloned().collect());
        }

        OptimizerState {
            optimizer_type: "RMSprop".to_string(),
            parameters,
        }
    }

    fn set_state(&mut self, state: OptimizerState) {
        if let Some(lr) = state.parameters.get("learning_rate") {
            self.learning_rate = lr[0];
        }
        if let Some(alpha) = state.parameters.get("alpha") {
            self.alpha = alpha[0];
        }
        if let Some(epsilon) = state.parameters.get("epsilon") {
            self.epsilon = epsilon[0];
        }
        if let Some(weight_decay) = state.parameters.get("weight_decay") {
            self.weight_decay = weight_decay[0];
        }
        if let Some(momentum) = state.parameters.get("momentum") {
            self.momentum = momentum[0];
        }
        if let Some(centered) = state.parameters.get("centered") {
            self.centered = centered[0] != 0.0;
        }
        if let Some(square_avg) = state.parameters.get("square_avg") {
            if let Some(ref current_square_avg) = self.square_avg {
                let shape = current_square_avg.shape();
                self.square_avg =
                    Some(Array2::from_shape_vec((shape[0], shape[1]), square_avg.clone()).unwrap());
            }
        }
        if let Some(momentum_buffer) = state.parameters.get("momentum_buffer") {
            if let Some(ref current_momentum_buffer) = self.momentum_buffer {
                let shape = current_momentum_buffer.shape();
                self.momentum_buffer = Some(
                    Array2::from_shape_vec((shape[0], shape[1]), momentum_buffer.clone()).unwrap(),
                );
            }
        }
        if let Some(grad_avg) = state.parameters.get("grad_avg") {
            if let Some(ref current_grad_avg) = self.grad_avg {
                let shape = current_grad_avg.shape();
                self.grad_avg =
                    Some(Array2::from_shape_vec((shape[0], shape[1]), grad_avg.clone()).unwrap());
            }
        }
    }
}

/// Learning rate scheduler
pub trait LearningRateScheduler {
    fn get_learning_rate(&self, epoch: usize, step: usize) -> f32;
    fn name(&self) -> &'static str;
}

/// Step learning rate scheduler
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }
}

impl LearningRateScheduler for StepLR {
    fn get_learning_rate(&self, _epoch: usize, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi((step / self.step_size) as i32)
    }

    fn name(&self) -> &'static str {
        "StepLR"
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn get_learning_rate(&self, _epoch: usize, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }

    fn name(&self) -> &'static str {
        "ExponentialLR"
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
    eta_min: f32,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        Self {
            initial_lr,
            t_max,
            eta_min,
        }
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn get_learning_rate(&self, _epoch: usize, step: usize) -> f32 {
        let step = step % self.t_max;
        self.eta_min
            + (self.initial_lr - self.eta_min)
                * (1.0 + (std::f32::consts::PI * step as f32 / self.t_max as f32).cos())
                / 2.0
    }

    fn name(&self) -> &'static str {
        "CosineAnnealingLR"
    }
}
