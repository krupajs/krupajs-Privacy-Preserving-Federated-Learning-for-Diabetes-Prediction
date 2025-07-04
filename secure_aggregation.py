import numpy as np
import torch
import syft as sy

def add_noise_to_model(parameters, noise_level=0.01):
    """Add small Gaussian noise to simulate differential privacy"""
    return [p + np.random.normal(0, noise_level, p.shape) for p in parameters]

def apply_differential_privacy(model_parameters):
    """Apply noise to model parameters for differential privacy"""
    privacy_model = add_noise_to_model(model_parameters, noise_level=0.01)
    return privacy_model
