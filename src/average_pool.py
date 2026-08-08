"""Global average pooling used by ResNet."""

import torch
from torch import Tensor, nn


class AveragePool(nn.Module):
    """Average each channel over all spatial dimensions."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected a 4D NCHW tensor, got shape {tuple(x.shape)}")
        return torch.mean(x, dim=(2, 3))
