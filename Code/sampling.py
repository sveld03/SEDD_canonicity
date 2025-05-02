# This script was left mostly unchanged, except for pc_campler.
# The pc_sampler function was modified to return all intermediate denoising steps.

"""
Module for implementing various sampling strategies for the Score Entropy Discrete Diffusion model.
Includes predictor classes and sampling functions for the diffusion process.
"""

import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical
from transformers import GPT2TokenizerFast

from model import utils as mutils

# Registry for predictor classes
_PREDICTORS = {}

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def register_predictor(cls=None, *, name=None):
    """
    Decorator for registering predictor classes.
    
    Args:
        cls: The predictor class to register
        name: Optional name for the predictor. If None, uses class name
        
    Returns:
        The registered class or a decorator function
        
    Raises:
        ValueError: If a predictor with the given name is already registered
    """
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def get_predictor(name):
    """Get a predictor class by name."""
    return _PREDICTORS[name]

class Predictor(abc.ABC):
    """Abstract base class for predictor algorithms."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """
        Perform one update step of the predictor.
        
        Args:
            score_fn: Function to compute scores
            x: Current state tensor
            t: Current time step tensor
            step_size: Size of the update step
            
        Returns:
            Updated state tensor
        """
        pass

@register_predictor(name="euler")
class EulerPredictor(Predictor):
    """Euler predictor implementation."""
    
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    """Predictor that performs no updates."""
    
    def update_fn(self, score_fn, x, t, step_size):
        return x

@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    """Analytic predictor implementation."""
    
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

class Denoiser:
    """Class for performing final denoising step."""
    
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        
        # Truncate probabilities if using absorbing state
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        return sample_categorical(probs)

def get_sampling_fn(config, graph, noise, batch_dims, eps, device, intermediates=False):
    """
    Get a sampling function configured with the given parameters.
    
    Args:
        config: Configuration object containing sampling parameters
        graph: Graph object for the diffusion process
        noise: Noise schedule
        batch_dims: Dimensions of the batch
        eps: Small constant for numerical stability
        device: Device to run sampling on
        intermediates: Whether to return intermediate states
        
    Returns:
        Configured sampling function
    """
    sampling_fn = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
        intermediates=intermediates
    )
    
    return sampling_fn

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, 
                  device=torch.device('cuda:2'), proj_fun=lambda x: x, intermediates=False):
    """
    Get a predictor-corrector sampler function.
    
    Args:
        graph: Graph object for the diffusion process
        noise: Noise schedule
        batch_dims: Dimensions of the batch
        predictor: Name of predictor to use
        steps: Number of sampling steps
        denoise: Whether to perform final denoising step
        eps: Small constant for numerical stability
        device: Device to run sampling on
        proj_fun: Projection function to apply at each step
        intermediates: Whether to return intermediate states
        
    Returns:
        Sampling function that takes a model and returns generated samples
    """
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        # Get score function from model
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        
        # Initialize with fully masked tokens
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        intermediate_outputs = {}  # Store intermediate states if requested

        # Perform sampling steps
        for i in range(steps):
            if intermediates:
                intermediate_outputs[i] = x.tolist()

            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)

        if intermediates:
            intermediate_outputs[steps] = x.tolist()
        
        # Perform final denoising step if requested
        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
            if intermediates:
                intermediate_outputs[steps+1] = x.tolist()

        if intermediates:
            return intermediate_outputs
        else:
            return x

    return pc_sampler

