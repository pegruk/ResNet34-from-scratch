"""An educational BatchNorm2d module with PyTorch-compatible state."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BatchNorm2d(nn.Module):
    """Normalize NCHW activations and maintain running channel statistics."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected a 4D NCHW tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(f"expected {self.num_features} channels, got {x.shape[1]}")

        if self.training:
            self.num_batches_tracked.add_(1)

        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"
