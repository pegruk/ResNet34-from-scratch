"""Groups of residual blocks in a ResNet."""

from torch import Tensor, nn

from .residual_block import ResidualBlock
from .sequential import Sequential


class BlockGroup(nn.Module):
    """A stage which optionally downsamples once, then preserves shape."""

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        first_stride: int,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.blocks = Sequential(
            ResidualBlock(in_channels, out_channels, first_stride),
            *(ResidualBlock(out_channels, out_channels) for _ in range(num_blocks - 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)
