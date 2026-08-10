import re

import pytest
import torch

from resnet34 import ResNet34, resnet34


def test_resnet34_has_canonical_parameter_count() -> None:
    model = ResNet34()

    assert sum(parameter.numel() for parameter in model.parameters()) == 21_797_672


def test_resnet34_output_shape() -> None:
    model = ResNet34(num_classes=10).eval()

    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 10)


def test_custom_class_count_rejects_pretrained_weights() -> None:
    with pytest.raises(ValueError, match="num_classes=1000"):
        resnet34(pretrained=True, num_classes=10)


def test_torchvision_state_loading_rejects_incomplete_state() -> None:
    model = ResNet34()

    with pytest.raises(ValueError, match="missing translated state entries"):
        model.load_torchvision_state_dict({"conv1.weight": model.in_layers[0].weight})


def test_torchvision_state_loading_maps_every_state_entry() -> None:
    model = ResNet34()
    component_names = {
        ("left", "0"): "conv1",
        ("left", "1"): "bn1",
        ("left", "3"): "conv2",
        ("left", "4"): "bn2",
        ("right", "0"): "downsample.0",
        ("right", "1"): "downsample.1",
    }

    source = {}
    expected = {}
    for target_name, value in model.state_dict().items():
        if target_name.startswith("in_layers.0."):
            source_name = f"conv1.{target_name.removeprefix('in_layers.0.')}"
        elif target_name.startswith("in_layers.1."):
            source_name = f"bn1.{target_name.removeprefix('in_layers.1.')}"
        elif target_name.startswith("out_layers.1."):
            source_name = f"fc.{target_name.removeprefix('out_layers.1.')}"
        else:
            match = re.fullmatch(
                r"residual_layers\.(\d+)\.blocks\.(\d+)\.(left|right)\.(\d+)\.(.+)",
                target_name,
            )
            assert match is not None
            stage, block, side, index, suffix = match.groups()
            source_name = f"layer{int(stage) + 1}.{block}.{component_names[(side, index)]}.{suffix}"
        source[source_name] = value.clone()
        expected[target_name] = value.clone()

    model.load_torchvision_state_dict(source)

    for target_name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[target_name])


@pytest.mark.integration
def test_pretrained_model_matches_torchvision() -> None:
    torchvision = pytest.importorskip("torchvision")
    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    reference = torchvision.models.resnet34(weights=weights).eval()
    custom = ResNet34.from_pretrained().eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.inference_mode():
        expected = reference(x)
        actual = custom(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
