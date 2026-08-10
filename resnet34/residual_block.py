"""The basic residual block used in ResNet-18 and ResNet-34."""

from torch import Tensor, nn

from .batch_norm2d import BatchNorm2d
from .sequential import Sequential


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an optional projected skip connection."""

    def __init__(
        self, in_channels: int, out_channels: int, first_stride: int = 1
    ) -> None:
        super().__init__()
        if first_stride not in (1, 2):
            raise ValueError("first_stride must be 1 or 2")

        self.left = Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=first_stride,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
        )
        self.right = (
            nn.Identity()
            if in_channels == out_channels and first_stride == 1
            else Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False,
                ),
                BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.left(x) + self.right(x))
