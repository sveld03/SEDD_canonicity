# This script was originally used to directly generate samples with a command-line prompt.
# However, I modified it to create a sample_tokens function that can be used to generate samples.

"""
Script for generating token sequences using the Score Entropy Discrete Diffusion model.
Provides a function for sampling tokens with optional intermediate step tracking.
"""

import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Configure torch to display full tensor contents
torch.set_printoptions(threshold=10000)

def sample_tokens(batch_size, num_tokens, steps, intermediates=False):
    """
    Generate token sequences using the Score Entropy Discrete Diffusion model.
    
    Args:
        batch_size (int): Number of sequences to generate in parallel
        num_tokens (int): Length of each token sequence
        steps (int): Number of diffusion steps to perform
        intermediates (bool): Whether to return intermediate states during diffusion
        
    Returns:
        If intermediates=True:
            dict: Mapping of step numbers to intermediate token sequences
        Otherwise:
            torch.Tensor: Final generated token sequences
    """
    # Set up device and load model
    device = torch.device('cuda:2')
    model, graph, noise = load_model("louaaron/sedd-medium", device)

    # Configure sampling function
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (batch_size, num_tokens), 'analytic', steps, device=device, intermediates=intermediates
    )

    # Generate samples
    return sampling_fn(model)