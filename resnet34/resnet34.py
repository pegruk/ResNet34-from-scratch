"""ResNet-34 assembled from the project's educational building blocks."""

from collections.abc import Mapping
import re

from torch import Tensor, nn

from .average_pool import AveragePool
from .batch_norm2d import BatchNorm2d
from .block_group import BlockGroup
from .sequential import Sequential


def _translate_torchvision_key(source_name: str) -> str:
    """Translate a torchvision ResNet-34 state key to the local module tree."""

    top_level = {
        "conv1": "in_layers.0",
        "bn1": "in_layers.1",
        "fc": "out_layers.1",
    }
    for source_prefix, target_prefix in top_level.items():
        if source_name.startswith(f"{source_prefix}."):
            return f"{target_prefix}.{source_name.removeprefix(f'{source_prefix}.')}"

    match = re.fullmatch(r"layer([1-4])\.(\d+)\.(.+)", source_name)
    if match is None:
        raise ValueError(f"unsupported torchvision state key: {source_name}")

    stage_number, block_number, tail = match.groups()
    component, suffix = tail.split(".", maxsplit=1)
    component_paths = {
        "conv1": "left.0",
        "bn1": "left.1",
        "conv2": "left.3",
        "bn2": "left.4",
    }
    if component in component_paths:
        target_component = component_paths[component]
    elif component == "downsample":
        downsample_number, suffix = suffix.split(".", maxsplit=1)
        target_component = {"0": "right.0", "1": "right.1"}.get(downsample_number)
        if target_component is None:
            raise ValueError(f"unsupported torchvision state key: {source_name}")
    else:
        raise ValueError(f"unsupported torchvision state key: {source_name}")

    return f"residual_layers.{int(stage_number) - 1}.blocks.{block_number}.{target_component}.{suffix}"


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

    def load_torchvision_state_dict(self, source: Mapping[str, Tensor]) -> None:
        """Load a torchvision ResNet-34 state dict by explicit key translation."""

        target = self.state_dict()
        translated: dict[str, Tensor] = {}
        for source_name, source_value in source.items():
            target_name = _translate_torchvision_key(source_name)
            if target_name in translated:
                raise ValueError(f"multiple source keys map to {target_name}")
            if target_name not in target:
                raise ValueError(
                    f"target has no state entry for {source_name} -> {target_name}"
                )
            if target[target_name].shape != source_value.shape:
                raise ValueError(
                    f"state shape mismatch: {target_name} {tuple(target[target_name].shape)} != "
                    f"{source_name} {tuple(source_value.shape)}"
                )
            translated[target_name] = source_value

        missing = sorted(set(target) - set(translated))
        if missing:
            raise ValueError(f"source is missing translated state entries: {missing}")

        self.load_state_dict(translated, strict=True)

    @classmethod
    def from_pretrained(cls, *, progress: bool = True) -> "ResNet34":
        """Build a model with torchvision's ImageNet-1K V1 weights."""

        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as error:
            raise ImportError("pretrained weights require torchvision") from error

        reference = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=progress)
        model = cls(num_classes=1000)
        model.load_torchvision_state_dict(reference.state_dict())
        model.eval()
        return model


def resnet34(
    *, pretrained: bool = False, progress: bool = True, num_classes: int = 1000
) -> ResNet34:
    """Construct ResNet-34, optionally initialized with ImageNet weights."""

    if pretrained:
        if num_classes != 1000:
            raise ValueError("pretrained ImageNet weights require num_classes=1000")
        return ResNet34.from_pretrained(progress=progress)
    return ResNet34(num_classes=num_classes)
