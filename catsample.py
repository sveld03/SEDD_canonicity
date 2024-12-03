import torch

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        # Clamp to ensure non-negative values
        categorical_probs = categorical_probs.clamp(min=1e-11)
        
        # Normalize to sum to 1 along the last dimension
        categorical_probs = categorical_probs / categorical_probs.sum(dim=-1, keepdim=True)
        
        # Reshape to 2D if necessary
        original_shape = categorical_probs.shape
        if categorical_probs.dim() > 2:
            categorical_probs = categorical_probs.view(-1, categorical_probs.size(-1))
        
        # Use torch.multinomial to sample indices
        indices = torch.multinomial(categorical_probs, num_samples=1, replacement=True)
        
        # Reshape back to the original shape (excluding the sampled dimension)
        indices = indices.view(*original_shape[:-1])
        return indices
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    