"""
This module provides functionality for sampling from categorical distributions using torch.multinomial.
Unlike the original implementation that used Gumbel normalization (which created an implicit score boost),
this implementation uses direct categorical sampling which avoids this issue.
"""

import torch

def sample_categorical(categorical_probs, method="hard"):
    """
    Sample from a categorical distribution using the specified method.
    
    Args:
        categorical_probs (torch.Tensor): Probability distribution tensor
        method (str): Sampling method to use. Currently only "hard" sampling is supported.
    
    Returns:
        torch.Tensor: Sampled indices from the categorical distribution
        
    Raises:
        ValueError: If an invalid sampling method is specified
    """
    if method == "hard":
        # Ensure non-negative probabilities with a small epsilon
        categorical_probs = categorical_probs.clamp(min=1e-11)
        
        # Normalize probabilities to sum to 1 along the last dimension
        categorical_probs = categorical_probs / categorical_probs.sum(dim=-1, keepdim=True)
        
        # Handle multi-dimensional input by reshaping to 2D
        original_shape = categorical_probs.shape
        if categorical_probs.dim() > 2:
            categorical_probs = categorical_probs.view(-1, categorical_probs.size(-1))
        
        # Sample indices using torch.multinomial
        indices = torch.multinomial(categorical_probs, num_samples=1, replacement=True)
        
        # Restore original shape (excluding the sampled dimension)
        indices = indices.view(*original_shape[:-1])
        return indices
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    