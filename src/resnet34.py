"""ResNet-34 assembled from the project's educational building blocks."""

from collections.abc import Mapping

from torch import Tensor, nn

from .average_pool import AveragePool
from .batch_norm2d import BatchNorm2d
from .block_group import BlockGroup
from .sequential import Sequential


class ResNet34(nn.Module):
    """The ResNet-34 image classifier described by He et al. (2015)."""

    block_counts = (3, 4, 6, 3)
    channels = (64, 128, 256, 512)
    strides = (1, 2, 2, 2)

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.in_layers = Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.residual_layers = Sequential(
            *(
                BlockGroup(num_blocks, in_channels, out_channels, first_stride)
                for num_blocks, in_channels, out_channels, first_stride in zip(
                    self.block_counts,
                    (64, *self.channels[:-1]),
                    self.channels,
                    self.strides,
                    strict=True,
                )
            )
        )
        self.out_layers = Sequential(
            AveragePool(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layers(x)
        x = self.residual_layers(x)
        return self.out_layers(x)

    def load_ordered_state_dict(self, source: Mapping[str, Tensor]) -> None:
        """Copy compatible state by order, allowing different educational names."""

        target = self.state_dict()
        if len(target) != len(source):
            raise ValueError(f"state entry count differs: expected {len(target)}, got {len(source)}")

        copied: dict[str, Tensor] = {}
        for (target_name, target_value), (source_name, source_value) in zip(
            target.items(), source.items(), strict=True
        ):
            if target_value.shape != source_value.shape:
                raise ValueError(
                    f"state shape mismatch: {target_name} {tuple(target_value.shape)} != "
                    f"{source_name} {tuple(source_value.shape)}"
                )
            copied[target_name] = source_value

        self.load_state_dict(copied, strict=True)

    @classmethod
    def from_pretrained(cls, *, progress: bool = True) -> "ResNet34":
        """Build a model with torchvision's ImageNet-1K V1 weights."""

        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as error:
            raise ImportError("pretrained weights require torchvision") from error

        reference = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=progress)
        model = cls(num_classes=1000)
        model.load_ordered_state_dict(reference.state_dict())
        model.eval()
        return model


def resnet34(*, pretrained: bool = False, progress: bool = True, num_classes: int = 1000) -> ResNet34:
    """Construct ResNet-34, optionally initialized with ImageNet weights."""

    if pretrained:
        if num_classes != 1000:
            raise ValueError("pretrained ImageNet weights require num_classes=1000")
        return ResNet34.from_pretrained(progress=progress)
    return ResNet34(num_classes=num_classes)
