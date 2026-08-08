# ResNet-34 from scratch

An educational implementation of ResNet-34 using PyTorch building blocks. The
project follows the approach used in the
[ARENA CNNs & ResNets lesson](https://learn.arena.education/chapter0_fundamentals/02_cnns/4-resnets/):
assemble the architecture ourselves, then transfer the official torchvision
ImageNet-1K weights and verify that both models produce the same output.

The model code does not instantiate or wrap torchvision's ResNet. Torchvision is
used only for its pretrained checkpoint, preprocessing recipe, and ImageNet
class labels.

## Architecture

The implementation is split into small modules that mirror the ResNet paper:

- `BatchNorm2d`, `AveragePool`, and `Sequential` provide the core operations.
- `ResidualBlock` implements two 3x3 convolutions and a skip connection.
- `BlockGroup` stacks residual blocks into the `[3, 4, 6, 3]` stages.
- `ResNet34` combines the stem, four stages, global average pool, and classifier.

The resulting network has the canonical **21,797,672 parameters** and emits
1,000 ImageNet class logits when pretrained weights are used.

## Setup

Python 3.11 or newer is required. Create an isolated environment and install the
project with its test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Inference

Pass either a local image or an HTTP(S) URL. On the first run, torchvision
downloads the 83 MB checkpoint to PyTorch's standard cache directory.

```bash
python -m examples.run_inference \
  https://github.com/pytorch/hub/raw/master/images/dog.jpg
```

The model can also be used directly:

```python
import torch
from src import ResNet34

model = ResNet34.from_pretrained()
with torch.inference_mode():
    logits = model(torch.randn(1, 3, 224, 224))
```

Use `ResNet34(num_classes=...)` for random initialization and a custom output
size. Pretrained ImageNet weights require `num_classes=1000`.

## Tests

```bash
pytest -q
```

The suite checks individual module behavior, output shapes, the canonical
parameter count, and numerical parity with torchvision after transferring the
pretrained state.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ARENA: CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/4-resnets/)
- [torchvision ResNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
