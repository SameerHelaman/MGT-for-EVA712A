"""Original MGT MLP projection and Gaussian radial-basis expansion layers."""

import torch
import numpy as np
from typing import Optional
from torch import nn, Tensor


class MLPLayer(nn.Module):
    """Apply a linear projection followed by layer normalization and SiLU activation."""
    def __init__(self, in_features: int, out_features: int):
        """Initialize this object and its required state."""
        super(MLPLayer, self).__init__()

        # Learn an affine projection into the requested output feature width.
        self.linear = nn.Linear(in_features, out_features)
        # Normalize each projected feature vector before nonlinear activation.
        self.norm = nn.LayerNorm(out_features)
        self.SiLU = nn.SiLU(inplace=True)        

    def forward(self, x: Tensor):
        """Apply this module to its input tensors or graphs."""
        x = self.linear(x)
        x = self.norm(x)
        return self.SiLU(x)


class RBFExpansion(nn.Module):
    """Expand scalar geometric values into fixed Gaussian radial-basis vectors."""
    def __init__(self, vmin: float = 0, vmax: float = 8, bins: int = 40, lenghtscale: Optional[float] = None):
        """Initialize this object and its required state."""
        super(RBFExpansion, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        # Store evenly spaced non-trainable RBF centers with the model state.
        self.register_buffer('centers', torch.linspace(vmin, vmax, bins))

        if lenghtscale is None:
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lenghtscale
            self.gamma = 1 / (lenghtscale ** 2)

    def forward(self, x: Tensor):
        """Apply this module to its input tensors or graphs."""
        # Evaluate every scalar against every Gaussian center: input N becomes N×bins.
        return torch.exp(-self.gamma * (x.unsqueeze(1) - self.centers) ** 2)
