"""Telling players apart by their looks: a feature vector per player crop.

Jersey numbers are read for some players some of the time, and a track ends when the
tracker loses its player or the video cuts. To know that the player in one point is the
one from an earlier point, something else must tell: cleats, socks, a hat, sleeves,
skin, build. A small network turns the crop of a player into a vector such that two
crops of one player lie near each other and those of two players apart; tracks are then
matched by how near their vectors are.

A prototype: nothing in the app uses it yet. `scripts/build_reid_dataset.py` collects the
crops, `scripts/train_reid.py` trains and `scripts/benchmark_reid.py` measures it; how
well it does, and which of the networks tried does best, is in docs/MEASUREMENTS.md.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torchvision import models

    TORCH_AVAILABLE = True
except ImportError:  # The app runs without this prototype
    TORCH_AVAILABLE = False

# Crops are brought to this size (width, height): players are some 90 pixels tall, and
# what tells them apart is small, so they are enlarged rather than shrunk
CROP_SIZE = (64, 128)
VECTOR_LENGTH = 128
# What the networks' first layers were trained with (ImageNet), as RGB
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SPREAD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fit(
    crop: np.ndarray, size: Sequence[int] = CROP_SIZE, keep_proportions: bool = False
) -> np.ndarray:
    """A crop (BGR, any size) brought to the size the network takes.

    Stretched to it, or, keeping its proportions, scaled to fit and set in the middle
    of a grey picture: a player with arms out has a wide box, and stretching squeezes
    them thin.
    """
    width, height = size
    if not keep_proportions:
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    scale = min(width / crop.shape[1], height / crop.shape[0])
    new_width = max(1, min(width, round(crop.shape[1] * scale)))
    new_height = max(1, min(height, round(crop.shape[0] * scale)))
    picture = np.empty((height, width, 3), dtype=np.uint8)
    picture[:] = np.round(MEAN[::-1] * 255)  # Nothing, as the network sees it
    left, top = (width - new_width) // 2, (height - new_height) // 2
    picture[top : top + new_height, left : left + new_width] = cv2.resize(
        crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return picture


def sharpness(crop: np.ndarray) -> float:
    """How sharp a crop is: high for crisp edges, low for a player smeared by motion."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def prepare(
    crops: Sequence[np.ndarray],
    size: Sequence[int] = CROP_SIZE,
    keep_proportions: bool = False,
) -> "torch.Tensor":
    """Crops (BGR, any size) as one batch for the network, all at one size."""
    batch = np.empty((len(crops), 3, size[1], size[0]), dtype=np.float32)
    for index, crop in enumerate(crops):
        rgb = fit(crop, size, keep_proportions)[:, :, ::-1].astype(np.float32) / 255.0
        batch[index] = ((rgb - MEAN) / SPREAD).transpose(2, 0, 1)
    return torch.from_numpy(batch)


if TORCH_AVAILABLE:

    def _weights(pretrained: bool) -> Optional[str]:
        return "DEFAULT" if pretrained else None

    def _resnet(name: str, stages: int) -> Callable[[bool], "nn.Module"]:
        def build(pretrained: bool) -> nn.Module:
            net = getattr(models, name)(weights=_weights(pretrained))
            layers = [net.layer1, net.layer2, net.layer3, net.layer4][:stages]
            return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, *layers)

        return build

    def _features(name: str) -> Callable[[bool], "nn.Module"]:
        return lambda pretrained: getattr(models, name)(weights=_weights(pretrained)).features

    def _shufflenet(pretrained: bool) -> "nn.Module":
        net = models.shufflenet_v2_x1_0(weights=_weights(pretrained))
        return nn.Sequential(net.conv1, net.maxpool, net.stage2, net.stage3, net.stage4, net.conv5)

    def _regnet(pretrained: bool) -> "nn.Module":
        net = models.regnet_y_400mf(weights=_weights(pretrained))
        return nn.Sequential(net.stem, net.trunk_output)

    def _small_cnn(pretrained: bool) -> "nn.Module":
        """Four plain stages, trained from nothing: what the crops alone can teach."""

        def stage(inputs: int, outputs: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        return nn.Sequential(stage(3, 32), stage(32, 64), stage(64, 128), stage(128, 256))

    # Name -> what makes the part of a network that turns a picture into a map of
    # features. "_s3" and "_s2": a ResNet without its last one or two stages, which at
    # 128x64 would look at the player through too few cells for a pair of cleats.
    ARCHITECTURES: Dict[str, Callable[[bool], "nn.Module"]] = {
        "small_cnn": _small_cnn,
        "resnet18_s2": _resnet("resnet18", 2),
        "resnet18_s3": _resnet("resnet18", 3),
        "resnet18": _resnet("resnet18", 4),
        "resnet34_s3": _resnet("resnet34", 3),
        "resnet50_s3": _resnet("resnet50", 3),
        "mobilenet_v3_small": _features("mobilenet_v3_small"),
        "mobilenet_v3_large": _features("mobilenet_v3_large"),
        "efficientnet_b0": _features("efficientnet_b0"),
        "convnext_tiny": _features("convnext_tiny"),
        "densenet121": _features("densenet121"),
        "shufflenet_v2": _shufflenet,
        "regnet_y_400mf": _regnet,
    }
    DEFAULT_ARCHITECTURE = "resnet18_s3"

    class PlayerEmbedder(nn.Module):
        """A crop of a player -> a vector of length 1 that is near those of the same player."""

        def __init__(
            self,
            architecture: str = DEFAULT_ARCHITECTURE,
            pretrained: bool = True,
            identities: int = 0,
            parts: int = 2,
            size: Sequence[int] = CROP_SIZE,
            keep_proportions: bool = False,
        ):
            """
            Args:
                architecture: One of ARCHITECTURES
                pretrained: Start from the weights the network has for ImageNet
                identities: For training: how many players are to be told by name
                parts: The picture is pooled in this many bands from head to foot, each
                    kept apart: a hat is not a cleat
                size: (width, height) the crops are brought to
                keep_proportions: Whether they keep their proportions on the way (see fit)
            """
            super().__init__()
            self.architecture, self.parts = architecture, parts
            self.size, self.keep_proportions = tuple(size), keep_proportions
            self.trunk = ARCHITECTURES[architecture](pretrained)
            with torch.no_grad():
                was_training = self.trunk.training
                self.trunk.eval()
                channels = self.trunk(torch.zeros(1, 3, self.size[1], self.size[0])).shape[1]
                self.trunk.train(was_training)
            self.pool = nn.AdaptiveAvgPool2d((parts, 1))
            self.neck = nn.Sequential(
                nn.Linear(parts * channels, VECTOR_LENGTH), nn.BatchNorm1d(VECTOR_LENGTH)
            )
            self.who = nn.Linear(VECTOR_LENGTH, identities, bias=False) if identities else None

        def forward(self, batch: "torch.Tensor") -> "torch.Tensor":
            features = self.pool(self.trunk(batch)).flatten(1)
            return nn.functional.normalize(self.neck(features), dim=1)


def save_embedder(model: "PlayerEmbedder", path: Path) -> None:
    """Write a network with what is needed to build it again."""
    state = {key: value for key, value in model.state_dict().items() if not key.startswith("who.")}
    torch.save(
        {
            "architecture": model.architecture,
            "parts": model.parts,
            "size": list(model.size),
            "keep_proportions": model.keep_proportions,
            "state": state,
        },
        str(path),
    )


def load_embedder(weights: Path, device: Optional[str] = None) -> "PlayerEmbedder":
    """A trained embedder, ready to use."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(str(weights), map_location=device)
    model = PlayerEmbedder(
        saved["architecture"],
        pretrained=False,
        parts=saved["parts"],
        size=saved.get("size", CROP_SIZE),
        keep_proportions=saved.get("keep_proportions", False),
    )
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


def embed(
    model: "PlayerEmbedder", crops: Sequence[np.ndarray], batch_size: int = 256
) -> np.ndarray:
    """The vectors of some crops (BGR), one row each."""
    device = next(model.parameters()).device
    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = prepare(
                crops[start : start + batch_size], model.size, model.keep_proportions
            ).to(device)
            vectors.append(model(batch).cpu().numpy())
    return np.concatenate(vectors) if vectors else np.zeros((0, VECTOR_LENGTH), dtype=np.float32)


def track_vector(vectors: np.ndarray) -> np.ndarray:
    """One vector for a track from those of its crops: their mean, brought to length 1."""
    mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)
