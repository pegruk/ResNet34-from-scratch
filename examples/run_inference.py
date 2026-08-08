"""Classify an image with the custom pretrained ResNet-34."""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from PIL import Image
from torchvision.models import ResNet34_Weights

from src import ResNet34


def load_image(source: str) -> Image.Image:
    """Load an RGB image from a local path or an HTTP(S) URL."""

    if urlparse(source).scheme in {"http", "https"}:
        with urlopen(source, timeout=30) as response:  # noqa: S310 - explicit CLI input
            image = Image.open(BytesIO(response.read()))
    else:
        image = Image.open(Path(source).expanduser())
    return image.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="local image path or HTTP(S) URL")
    parser.add_argument("--top-k", type=int, default=5, help="number of predictions to display")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.top_k <= 1000:
        raise SystemExit("--top-k must be between 1 and 1000")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    weights = ResNet34_Weights.IMAGENET1K_V1
    image = load_image(args.image)
    batch = weights.transforms()(image).unsqueeze(0).to(device)
    model = ResNet34.from_pretrained().to(device)

    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[0]
    scores, class_ids = probabilities.topk(args.top_k)

    categories = weights.meta["categories"]
    for rank, (score, class_id) in enumerate(zip(scores, class_ids, strict=True), start=1):
        print(f"{rank:>2}. {categories[class_id]:<30} {score.item():.2%}")


if __name__ == "__main__":
    main()
