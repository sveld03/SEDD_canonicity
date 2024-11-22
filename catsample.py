import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)

""" Change this to torch.multinomial, figure out how to implement"""
def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")

# def sample_categorical(categorical_probs, method="multinomial"):
#     # Get original shape and number of categories
#     original_shape = categorical_probs.shape
#     num_categories = original_shape[-1]
    
#     # Reshape to 2D: (batch_size, num_categories)
#     # -1 automatically computes the batch dimension
#     probs_2d = categorical_probs.view(-1, num_categories)
    
#     # Ensure probabilities sum to 1 along the last dimension
#     probs_normalized = torch.nn.functional.softmax(probs_2d, dim=-1)
    
#     # Sample one category per row
#     samples_2d = torch.multinomial(probs_normalized, num_samples=1).squeeze(-1)
    
#     # Restore original shape (excluding the last dimension since we sampled one category)
#     return samples_2d.view(original_shape[:-1])
    