"""Whether a player holds the disc, from a picture of the player (a prototype).

Who has the disc is worked out from where the disc is seen (possession.py), and the disc
is often hidden: by the thrower's body, by the mark. A player with the disc can be told
without it, by how they stand: on a pivot foot, the arm drawn back, still while the
others run. A small network is taught that from crops the possession tracking has marked
(scripts/build_holder_dataset.py, scripts/train_holder.py).

Not used by the app yet.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .reid import TORCH_AVAILABLE, prepare

if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from .reid import ARCHITECTURES

# Width and height the crops are brought to. Wider for its height than a player: the
# crops take in some of what is beside the box, where a held disc often is.
CROP_SIZE = (96, 128)
CROP_PAD = 0.25  # Share of the box added around it in a crop
DEFAULT_ARCHITECTURE = "resnet18_s3"
# A player's speed over the ground is taken over this long
SPEED_SECONDS = 0.5

Box = Tuple[int, int, int, int]


def crop_around(
    frame: np.ndarray, box: Sequence[float], pad: float = CROP_PAD
) -> Tuple[np.ndarray, Box]:
    """The crop of a player as the network takes it: the box widened by `pad` on every
    side, within the frame. Returns the crop and the widened box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = pad * (x2 - x1), pad * (y2 - y1)
    widened = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )
    return frame[widened[1] : widened[3], widened[0] : widened[2]], widened


def disc_on(box: Sequence[float], discs: Sequence[Sequence[float]]) -> bool:
    """Whether the centre of any of the disc boxes lies within a box."""
    return any(
        box[0] <= (d[0] + d[2]) / 2 <= box[2] and box[1] <= (d[1] + d[3]) / 2 <= box[3]
        for d in discs
    )


def place_on_field(
    image_to_field: Optional[np.ndarray], box: Sequence[float]
) -> Optional[Tuple[float, float]]:
    """Where a player stands on the field (their feet), or None if the field is not known."""
    if image_to_field is None:
        return None
    place = image_to_field @ [(box[0] + box[2]) / 2.0, box[3], 1.0]
    if abs(place[2]) < 1e-12:
        return None
    return float(place[0] / place[2]), float(place[1] / place[2])


def speed_of(
    places: Dict[int, Dict[Any, Tuple[float, float]]], player: Any, frame: int, rate: float
) -> Optional[float]:
    """How fast a player moves over the ground at a frame, in field units a second, or
    None if that cannot be said. The holder of a disc stands, or stops within a few
    steps: a player at a run is not one.

    Args:
        places: Frame -> {player: place on the field}
        player: The player
        frame: The frame
        rate: Frames per second
    """
    now = places.get(frame, {}).get(player)
    reach = SPEED_SECONDS * rate
    earlier = [f for f in places if 0.5 * reach <= frame - f <= 1.5 * reach and player in places[f]]
    if now is None or not earlier:
        return None
    before = min(earlier, key=lambda f: abs(frame - f - reach))
    then = places[before][player]
    return float(np.hypot(now[0] - then[0], now[1] - then[1]) * rate / (frame - before))


if TORCH_AVAILABLE:

    class HolderNet(nn.Module):
        """A crop of a player -> how sure it is that they hold the disc (before the sigmoid)."""

        def __init__(
            self,
            architecture: str = DEFAULT_ARCHITECTURE,
            pretrained: bool = True,
            size: Sequence[int] = CROP_SIZE,
        ):
            super().__init__()
            self.architecture, self.size = architecture, tuple(size)
            self.trunk = ARCHITECTURES[architecture](pretrained)
            with torch.no_grad():
                was_training = self.trunk.training
                self.trunk.eval()
                channels = self.trunk(torch.zeros(1, 3, self.size[1], self.size[0])).shape[1]
                self.trunk.train(was_training)
            # The mean says what the picture shows on the whole, the largest value
            # whether something small is anywhere in it: a disc is a few cells
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * channels, 1))

        def forward(self, batch: "torch.Tensor") -> "torch.Tensor":
            features = self.trunk(batch)
            pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
            return self.head(pooled).squeeze(1)


def save_holder_net(model: "HolderNet", path: Path) -> None:
    """Write a network with what is needed to build it again."""
    torch.save(
        {
            "architecture": model.architecture,
            "size": list(model.size),
            "state": model.state_dict(),
        },
        str(path),
    )


def load_holder_net(weights: Path, device: Optional[str] = None) -> "HolderNet":
    """A trained network, ready to use."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(str(weights), map_location=device)
    model = HolderNet(saved["architecture"], pretrained=False, size=saved["size"])
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


def holding(model: "HolderNet", crops: Sequence[np.ndarray], batch_size: int = 256) -> np.ndarray:
    """For each crop (BGR, as wide around the player as the training crops) how sure the
    network is that the player holds the disc, 0 to 1."""
    device = next(model.parameters()).device
    sure: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = prepare(crops[start : start + batch_size], model.size).to(device)
            sure.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(sure) if sure else np.zeros(0, dtype=np.float32)
