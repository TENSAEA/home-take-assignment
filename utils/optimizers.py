"""
Optimizers for Neural Network Training
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent with Momentum.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def step(self, layers):
        """
        Update parameters for all layers.
        
        layers: list of layer objects with get_params() and get_grads() methods
        """
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'get_params'):
                continue
            
            params = layer.get_params()
            grads = layer.get_grads()
            
            if i not in self.velocities:
                self.velocities[i] = {}
                for key in params:
                    self.velocities[i][key] = np.zeros_like(params[key])
            
            new_params = {}
            for key in params:
                # Momentum update
                self.velocities[i][key] = (self.momentum * self.velocities[i][key] - 
                                           self.lr * grads[key])
                new_params[key] = params[key] + self.velocities[i][key]
            
            layer.set_params(new_params)
    
    def zero_grad(self, layers):
        """Reset gradients to zero."""
        for layer in layers:
            if hasattr(layer, 'dW'):
                layer.dW = np.zeros_like(layer.dW)
            if hasattr(layer, 'db'):
                layer.db = np.zeros_like(layer.db)


class Adam:
    """
    Adam optimizer with adaptive learning rates.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def step(self, layers):
        """Update parameters using Adam."""
        self.t += 1
        
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'get_params'):
                continue
            
            params = layer.get_params()
            grads = layer.get_grads()
            
            if i not in self.m:
                self.m[i] = {}
                self.v[i] = {}
                for key in params:
                    self.m[i][key] = np.zeros_like(params[key])
                    self.v[i][key] = np.zeros_like(params[key])
            
            new_params = {}
            for key in params:
                # Update biased first moment estimate
                self.m[i][key] = self.beta1 * self.m[i][key] + (1 - self.beta1) * grads[key]
                # Update biased second moment estimate
                self.v[i][key] = self.beta2 * self.v[i][key] + (1 - self.beta2) * (grads[key] ** 2)
                
                # Bias-corrected estimates
                m_hat = self.m[i][key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][key] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                new_params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            layer.set_params(new_params)
    
    def zero_grad(self, layers):
        """Reset gradients to zero."""
        for layer in layers:
            if hasattr(layer, 'dW'):
                layer.dW = np.zeros_like(layer.dW)
            if hasattr(layer, 'db'):
                layer.db = np.zeros_like(layer.db)
