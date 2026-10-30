"""Where a disc is likely to be in the frame: a heat map.

A network looks at the frame and at the two frames before it, a thirtieth of a second
apart and laid onto it by how the camera moved, so that what stands still lies on
itself in all three and what moves does not. For every pixel it says how surely the
middle of a disc is there (after TrackNet, which finds a tennis ball this way).

It is shown, not used: the disc is still found by the disc detector. The network is
trained from the terminal and kept under data/models/disc_heatmap.
"""

from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from .model_lock import GPU_SETUP_LOCK
from .reid import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torchvision import models

logger = get_logger("DISC_HEATMAP")

# The frames a network sees are this far apart (seconds): the next frame of a video of
# 30 a second, every other one at 60
FRAME_GAP_SECONDS = 1.0 / 30.0
# The frame as a network sees it, unless its file says otherwise: width and height, and
# the height filled up to a multiple of 16
FRAME_SIZE = (960, 540)


if TORCH_AVAILABLE:

    def _block(inside: int, outside: int) -> "nn.Sequential":
        return nn.Sequential(
            nn.Conv2d(inside, outside, 3, padding=1, bias=False),
            nn.BatchNorm2d(outside),
            nn.ReLU(inplace=True),
            nn.Conv2d(outside, outside, 3, padding=1, bias=False),
            nn.BatchNorm2d(outside),
            nn.ReLU(inplace=True),
        )

    class HeatmapNet(nn.Module):
        """Three frames (9 channels) -> for every pixel, how surely a disc's middle is
        there (before the sigmoid). Learned from nothing."""

        def __init__(self, width: int = 32):
            super().__init__()
            self.down = nn.ModuleList(
                [
                    _block(9, width),
                    _block(width, 2 * width),
                    _block(2 * width, 4 * width),
                    _block(4 * width, 8 * width),
                ]
            )
            self.up = nn.ModuleList(
                [
                    _block(12 * width, 4 * width),
                    _block(6 * width, 2 * width),
                    _block(3 * width, width),
                ]
            )
            self.head = nn.Conv2d(width, 1, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            kept = []
            for number, layer in enumerate(self.down):
                x = layer(x if number == 0 else nn.functional.max_pool2d(x, 2))
                kept.append(x)
            for layer, skip in zip(self.up, kept[-2::-1]):
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
                x = layer(torch.cat([x, skip], dim=1))
            return self.head(x)[:, 0]

    class PretrainedHeatmapNet(nn.Module):
        """The same with the first three stages of a ResNet-18 that has seen pictures
        before, its first layer given nine channels. The last step up to every pixel is
        made with the frames themselves beside it."""

        def __init__(self) -> None:
            super().__init__()
            net = models.resnet18(weights=None)  # The weights come with the file
            first = nn.Conv2d(9, 64, 7, stride=2, padding=3, bias=False)
            self.stem = nn.Sequential(first, net.bn1, net.relu)
            self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3])
            self.up = nn.ModuleList([_block(256 + 128, 128), _block(128 + 64, 64)])
            self.fine = _block(64 + 9, 32)
            self.head = nn.Conv2d(32, 1, 1)
            self.register_buffer("mean", torch.zeros(1, 9, 1, 1))
            self.register_buffer("spread", torch.ones(1, 9, 1, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            normal = (x - self.mean) / self.spread
            kept, y = [], self.stem(normal)
            for stage in self.stages:
                y = stage(y)
                kept.append(y)
            for layer, skip in zip(self.up, kept[-2::-1]):
                y = nn.functional.interpolate(
                    y, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
                y = layer(torch.cat([y, skip], dim=1))
            y = nn.functional.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
            return self.head(self.fine(torch.cat([y, normal], dim=1)))[:, 0]


class DiscHeatmap:
    """The heat map of the frames of a video, one after the other."""

    def __init__(self, weights: Path, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is needed for the disc heat map")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(str(weights), map_location=self.device)
        model = PretrainedHeatmapNet() if saved.get("pretrained") else HeatmapNet()
        model.load_state_dict(saved["state"])
        self._model = model.to(self.device).eval()
        # On a graphics card in half precision, as the detectors run: a sixth faster than
        # leaving the choice to autocast, and the same heat to the fourth decimal
        self._half = self.device == "cuda"
        if self._half:
            self._model = self._model.half().to(memory_format=torch.channels_last)
        self.one_frame = bool(saved.get("one_frame", False))
        # How surely a peak is a disc, as chosen on the validation frames
        self.threshold = float(saved.get("threshold", 0.5))
        self.frame_size: Tuple[int, int] = tuple(saved.get("frame", FRAME_SIZE))
        self._rows = -(-self.frame_size[1] // 16) * 16
        # A TensorRT engine built for the network (scripts/export_tensorrt.py --heatmap)
        self._engine = None
        if self.device == "cuda":
            from .network_engines import load_engine

            # Set up while nobody else is on the GPU, as every engine is (model_lock.py)
            with GPU_SETUP_LOCK:
                self._engine = load_engine(weights, self.input_shape)
        # The frames at the network's size, newest last, and how the camera moved to each
        # from the one before (at that size)
        self._frames: Deque["torch.Tensor"] = deque()
        self._moves: Deque[np.ndarray] = deque()
        self._pixels: Any = None  # The pixels of a frame, for laying frames onto each other
        logger.info(f"Disc heat map: {weights} at {self.frame_size[0]} x {self.frame_size[1]}")

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """What the network takes: three frames of three colours, at its size."""
        return (1, 9, self._rows, self.frame_size[0])

    def reset(self) -> None:
        """Forget the frames before (new video, seek, cut)."""
        self._frames.clear()
        self._moves.clear()

    def update(
        self,
        frame: np.ndarray,
        motion: Optional[np.ndarray],
        frame_rate: float,
        look: bool = True,
    ) -> Optional[np.ndarray]:
        """Take in a frame and say where a disc is likely in it.

        Args:
            frame: The frame (BGR), as it came
            motion: How the camera moved from the frame before to this one (3x3, pixels
                of the frame), None if that is not known: the frames before are then
                of no use
            frame_rate: Frames a second of the video
            look: False: the frame is only kept for the frames after it, and the
                network, which is most of the work, is not asked

        Returns:
            For every pixel of the frame at the network's size (`frame_size`), how
            surely a disc's middle is there, 0 to 1; None until the frames before are
            in, and where the network was not asked
        """
        step = max(1, round(frame_rate * FRAME_GAP_SECONDS))
        height, width = frame.shape[:2]
        with torch.no_grad():
            # On the graphics card from here on: making the frame smaller and laying the
            # frames onto each other take longer on the processor than the network does
            small = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float()
            if (width, height) != self.frame_size:
                small = nn.functional.interpolate(
                    small,
                    size=(self.frame_size[1], self.frame_size[0]),
                    mode="area" if width > self.frame_size[0] else "bilinear",
                )
            if motion is None:
                self.reset()
            elif self._frames:
                # The same move between the frames at the network's size
                scale = np.diag([self.frame_size[0] / width, self.frame_size[1] / height, 1.0])
                self._moves.append(scale @ motion @ np.linalg.inv(scale))
            self._frames.append(small)
            while len(self._frames) > 2 * step + 1:
                self._frames.popleft()
                self._moves.popleft()
            if not look:
                return None
            if self.one_frame:
                three = [small, small, small]
            elif len(self._frames) < 2 * step + 1:
                return None
            else:
                moves = list(self._moves)  # moves[i]: from frame i to frame i + 1
                later = np.eye(3)
                for move in moves[step:]:
                    later = move @ later
                whole = later.copy()
                for move in reversed(moves[:step]):
                    whole = whole @ move
                three = [
                    self._laid_onto(self._frames[0], whole),
                    self._laid_onto(self._frames[step], later),
                    small,
                ]
            batch = torch.cat(three, dim=1) / 255.0
            batch = nn.functional.pad(batch, (0, 0, 0, self._rows - self.frame_size[1]))
            with GPU_SETUP_LOCK.work():  # Not while an engine is set up on another thread
                if self._engine is not None:
                    heat = torch.sigmoid(self._engine(batch)[0].float())
                    return heat[: self.frame_size[1]].cpu().numpy()
                if self._half:
                    # Only the network: half precision is too coarse for where a pixel was
                    batch = batch.half().contiguous(memory_format=torch.channels_last)
                heat = torch.sigmoid(self._model(batch)[0].float())
                return heat[: self.frame_size[1]].cpu().numpy()

    def _laid_onto(self, picture: "torch.Tensor", move: np.ndarray) -> "torch.Tensor":
        """An earlier frame (1 x 3 x rows x columns) as it lies in the frame the camera
        has moved to since (`move`: pixel then -> pixel now); black where it shows
        nothing."""
        columns, rows = self.frame_size
        if self._pixels is None:
            ys, xs = torch.meshgrid(
                torch.arange(rows, dtype=torch.float32, device=self.device),
                torch.arange(columns, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            self._pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
        back = torch.tensor(np.linalg.inv(move), dtype=torch.float32, device=self.device)
        then = self._pixels @ back.T  # Where each pixel of now was then
        then = then[..., :2] / then[..., 2:]
        # As grid_sample counts: -1 to 1 from the one edge of the picture to the other
        then[..., 0] = (then[..., 0] + 0.5) * (2.0 / columns) - 1.0
        then[..., 1] = (then[..., 1] + 0.5) * (2.0 / rows) - 1.0
        return nn.functional.grid_sample(
            picture, then[None], mode="bilinear", padding_mode="zeros", align_corners=False
        )


# The spots of a heat map: no nearer to each other than this share of its width, at
# least this sure, and so many of them at most
PEAK_APART = 0.01
PEAK_LEAST = 0.05
PEAKS = 3


def peaks_of(heat: Optional[np.ndarray], frame_size: Tuple[int, int]) -> list:
    """The spots a heat map is surest of: (x, y, how sure) in the pixels of a frame of
    `frame_size` (width, height), the surest first."""
    if heat is None:
        return []
    rows, columns = heat.shape
    reach = max(1, round(PEAK_APART * columns))
    # Only where there is any heat to speak of: nearly everywhere there is none
    ys, xs = np.nonzero(heat >= PEAK_LEAST)
    sure = heat[ys, xs]
    across, down = frame_size[0] / columns, frame_size[1] / rows
    peaks: list = []
    while len(sure) and len(peaks) < PEAKS:
        top = int(np.argmax(sure))
        x, y = int(xs[top]), int(ys[top])
        peaks.append(((x + 0.5) * across, (y + 0.5) * down, float(sure[top])))
        apart = (np.abs(xs - x) > reach) | (np.abs(ys - y) > reach)
        ys, xs, sure = ys[apart], xs[apart], sure[apart]
    return peaks


def load_disc_heatmap(weights: Any) -> Optional[DiscHeatmap]:
    """The heat map of a network's file, or None if there is no such file or it does
    not load (said in the log)."""
    path = Path(str(weights))
    if not str(weights) or not path.is_file():
        logger.warning(f"No disc heat-map network at '{weights}'")
        return None
    try:
        return DiscHeatmap(path)
    except Exception as error:
        logger.exception(f"The disc heat-map network did not load: {error}")
        return None
