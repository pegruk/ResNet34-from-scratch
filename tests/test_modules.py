import pytest
import torch
from torch import nn

from src import AveragePool, BatchNorm2d, ResidualBlock, Sequential


def test_sequential_matches_pytorch() -> None:
    custom = Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    reference = nn.Sequential(*list(custom))
    x = torch.randn(5, 4)

    torch.testing.assert_close(custom(x), reference(x))
    assert len(custom) == 3
    assert custom[-1] is reference[-1]


def test_average_pool_reduces_spatial_dimensions() -> None:
    x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4)

    result = AveragePool()(x)

    torch.testing.assert_close(result, x.mean(dim=(2, 3)))
    assert result.shape == (2, 3)


def test_batch_norm_matches_pytorch_in_train_and_eval_modes() -> None:
    custom = BatchNorm2d(4)
    reference = nn.BatchNorm2d(4)
    x = torch.randn(8, 4, 5, 5)

    torch.testing.assert_close(custom(x), reference(x))
    torch.testing.assert_close(custom.running_mean, reference.running_mean)
    torch.testing.assert_close(custom.running_var, reference.running_var)

    custom.eval()
    reference.eval()
    torch.testing.assert_close(custom(x), reference(x))


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "stride", "expected_shape"),
    [
        (64, 64, 1, (2, 64, 16, 16)),
        (64, 128, 2, (2, 128, 8, 8)),
    ],
)
def test_residual_block_shapes(
    in_channels: int,
    out_channels: int,
    stride: int,
    expected_shape: tuple[int, ...],
) -> None:
    block = ResidualBlock(in_channels, out_channels, stride)
    x = torch.randn(2, in_channels, 16, 16)

    assert block(x).shape == expected_shape
