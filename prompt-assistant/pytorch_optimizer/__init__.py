"""
Black Box Prompt Optimization Engine
PyTorch-based Latent Space Contrastive Learning
"""

from .latent_prompt_optimizer import (
    LatentPromptOptimizer,
    ContrastivePromptTrainer
)

__version__ = "1.0.0"
__all__ = [
    'LatentPromptOptimizer',
    'ContrastivePromptTrainer'
]




