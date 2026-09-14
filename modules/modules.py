# =============================================================================
# MODULE: modules/modules.py
# PURPOSE: Provides dense MLP embedding layers and Gaussian radial-basis expansion.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: RBF_i(x) = exp(-gamma (x-centre_i)^2), mapping scalar distances or angle cosines to smooth basis vectors.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original MGT MLP projection and Gaussian radial-basis expansion layers."""

import torch  # Load a standard-library, scientific, or local project dependency.
import numpy as np  # Load a standard-library, scientific, or local project dependency.
from typing import Optional  # Import selected classes or functions from the named dependency.
from torch import nn, Tensor  # Import selected classes or functions from the named dependency.


# CLASS: MLPLayer — reusable model/data abstraction.
class MLPLayer(nn.Module):  # Define this reusable class and its inheritance contract.
    """Apply a linear projection followed by layer normalization and SiLU activation."""
    def __init__(self, in_features: int, out_features: int):  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        super(MLPLayer, self).__init__()  # Initialize or delegate to the parent class implementation.

        # Learn an affine projection into the requested output feature width.
        self.linear = nn.Linear(in_features, out_features)  # Store this configuration value or neural-network submodule on the instance.
        # Normalize each projected feature vector before nonlinear activation.
        self.norm = nn.LayerNorm(out_features)  # Store this configuration value or neural-network submodule on the instance.
        self.SiLU = nn.SiLU(inplace=True)          # Store this configuration value or neural-network submodule on the instance.

    def forward(self, x: Tensor):  # Define this callable; its indented block implements the documented operation.
        """Apply this module to its input tensors or graphs."""
        x = self.linear(x)  # Create or apply a trainable neural-network component.
        x = self.norm(x)  # Bind this name to an intermediate value, configuration setting, or result.
        return self.SiLU(x)  # Return this computed tensor, metric, object, or collection to the caller.


# CLASS: RBFExpansion — reusable model/data abstraction.
class RBFExpansion(nn.Module):  # Define this reusable class and its inheritance contract.
    """Expand scalar geometric values into fixed Gaussian radial-basis vectors."""
    def __init__(self, vmin: float = 0, vmax: float = 8, bins: int = 40, lenghtscale: Optional[float] = None):  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        super(RBFExpansion, self).__init__()  # Initialize or delegate to the parent class implementation.
        self.vmin = vmin  # Store this configuration value or neural-network submodule on the instance.
        self.vmax = vmax  # Store this configuration value or neural-network submodule on the instance.
        self.bins = bins  # Store this configuration value or neural-network submodule on the instance.
        # Store evenly spaced non-trainable RBF centers with the model state.
        self.register_buffer('centers', torch.linspace(vmin, vmax, bins))

        if lenghtscale is None:  # Evaluate this condition before executing the associated branch.
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()  # Store this configuration value or neural-network submodule on the instance.
            self.gamma = 1 / self.lengthscale  # Store this configuration value or neural-network submodule on the instance.
        else:  # Handle the remaining case not covered by earlier conditions.
            self.lengthscale = lenghtscale  # Store this configuration value or neural-network submodule on the instance.
            self.gamma = 1 / (lenghtscale ** 2)  # Store this configuration value or neural-network submodule on the instance.

    def forward(self, x: Tensor):  # Define this callable; its indented block implements the documented operation.
        """Apply this module to its input tensors or graphs."""
        # Evaluate every scalar against every Gaussian center: input N becomes N×bins.
        return torch.exp(-self.gamma * (x.unsqueeze(1) - self.centers) ** 2)  # Return this computed tensor, metric, object, or collection to the caller.
