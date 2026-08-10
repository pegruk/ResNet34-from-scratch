"""A small educational reimplementation of :class:`torch.nn.Sequential`."""

from collections import OrderedDict
from collections.abc import Iterator

from torch import Tensor, nn


class Sequential(nn.Module):
    """Apply a sequence of modules in the order they were supplied."""

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self._modules.update(
            OrderedDict((str(index), module) for index, module in enumerate(modules))
        )

    def forward(self, x: Tensor) -> Tensor:
        for module in self:
            x = module(x)
        return x

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> nn.Module:
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range")
        return list(self._modules.values())[index % len(self)]
