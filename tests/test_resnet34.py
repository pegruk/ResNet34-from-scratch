import pytest
import torch

from src import ResNet34, resnet34


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
