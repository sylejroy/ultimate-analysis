#!/usr/bin/env python3
"""Train the network behind the disc heat map (processing/disc_heatmap.py).

It learns from a dataset of scripts/build_disc_frames.py: each labelled frame with the
two frames before it, laid onto it by the camera's motion. The target is a picture that
is dark everywhere and has a small bright spot on the middle of every labelled disc.

    python scripts/train_disc_heatmap.py train_disc_frames_full three_frames_full --windows

writes data/models/disc_heatmap/<run name>/best.pt and log.csv, with a line per epoch
in the terminal. The network kept is the one with the best score on the validation
frames: of the surest spot of each frame, how often it is on a labelled disc (F1).

With --windows the network learns from windows of 960 x 544 cut out of the frames,
three in four with a disc in them, and is scored on whole frames: for datasets at full
size, where a whole frame is too large to learn from in batches. --one-frame gives it
the labelled frame three times (nothing to see moving), --scratch starts without the
encoder that has seen pictures before; both are for comparison.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import models  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.disc_heatmap import (  # noqa: E402
    HeatmapNet,
    PretrainedHeatmapNet,
)

WINDOW = (960, 544)  # Of a frame, to learn from (--windows)
WITH_DISC = 0.75  # Share of the windows cut so that a disc is in them
SPREAD = 2.5  # Of the spot around a disc's middle in the target, in pixels
NEAR = 8.0  # A spot this near a labelled disc is that disc, in pixels of a frame 960 wide
IMAGENET_MEAN = (0.406, 0.456, 0.485)  # Blue, green, red
IMAGENET_SPREAD = (0.225, 0.224, 0.229)


def new_network(scratch: bool) -> nn.Module:
    """A network to train: with the encoder of a ResNet-18 trained on ImageNet, its
    first layer given nine channels (the weights of three, three times, a third each),
    or learned from nothing."""
    model = HeatmapNet() if scratch else PretrainedHeatmapNet()
    nn.init.constant_(model.head.bias, -4.0)  # Nearly everywhere there is no disc
    if scratch:
        return model
    seen = models.resnet18(weights="DEFAULT")
    with torch.no_grad():
        model.stem[0].weight.copy_(seen.conv1.weight.repeat(1, 3, 1, 1) / 3.0)
        model.mean.copy_(torch.tensor(IMAGENET_MEAN * 3).view(1, 9, 1, 1))
        model.spread.copy_(torch.tensor(IMAGENET_SPREAD * 3).view(1, 9, 1, 1))
    model.stem[1].load_state_dict(seen.bn1.state_dict())
    for stage, theirs in zip(model.stages, (seen.layer1, seen.layer2, seen.layer3)):
        stage.load_state_dict(theirs.state_dict())
    return model


class Frames:
    """The frames of a dataset, as the network is given them."""

    def __init__(self, data: Path, one_frame: bool):
        self.data, self.one_frame = data, one_frame
        self.index = json.loads((data / "index.json").read_text())
        first = self.index[0]
        picture = cv2.imread(str(data / first["split"] / f"{first['name']}_2.jpg"))
        self.frame = (picture.shape[1], picture.shape[0])
        # What the network sees: the rows filled up to a multiple of 16
        self.size = (self.frame[0], -(-self.frame[1] // 16) * 16)
        self.near = NEAR * self.frame[0] / 960.0

    def split(self, name: str) -> list:
        return [row for row in self.index if row["split"] == name]

    def window_of(self, row: dict) -> tuple:
        """Where to cut a window out of a frame: mostly with one of its discs in it."""
        reach = (max(self.frame[0] - WINDOW[0], 0), max(self.frame[1] - WINDOW[1], 0))
        if row["discs"] and random.random() < WITH_DISC:
            cx, cy = random.choice(row["discs"])[:2]
            x = cx - random.uniform(0.1, 0.9) * WINDOW[0]
            y = cy - random.uniform(0.1, 0.9) * WINDOW[1]
        else:
            x, y = random.uniform(0, reach[0]), random.uniform(0, reach[1])
        return int(np.clip(x, 0, reach[0])), int(np.clip(y, 0, reach[1]))

    def load(self, row: dict, flip: bool = False, jitter=None, window=None) -> tuple:
        """The three frames of a row (9 x rows x columns, 0 to 1) and their target."""
        numbers = (2, 2, 2) if self.one_frame else (0, 1, 2)
        frames = [
            cv2.imread(str(self.data / row["split"] / f"{row['name']}_{n}.jpg")) for n in numbers
        ]
        picture = np.concatenate(frames, axis=2)
        discs, size = row["discs"], self.size
        if window is not None:
            picture = picture[window[1] : window[1] + WINDOW[1], window[0] : window[0] + WINDOW[0]]
            discs = [(cx - window[0], cy - window[1], w, h) for cx, cy, w, h in discs]
            size = WINDOW
        picture = picture.astype(np.float32) / 255.0
        if jitter is not None:  # Lighter or darker, more or less contrast
            picture = np.clip(picture * jitter[0] + jitter[1], 0.0, 1.0)
        padded = np.zeros((size[1], size[0], 9), dtype=np.float32)
        padded[: picture.shape[0]] = picture
        target = np.zeros((size[1], size[0]), dtype=np.float32)
        for cx, cy, w, h in discs:
            spread = max(SPREAD, 0.25 * max(w, h))
            far = int(4 * spread) + 1  # Beyond this the spot is nothing
            x1, x2 = max(0, int(cx) - far), min(size[0], int(cx) + far + 1)
            y1, y2 = max(0, int(cy) - far), min(size[1], int(cy) + far + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            ys, xs = np.mgrid[y1:y2, x1:x2]
            spot = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * spread**2))
            target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], spot)
        if flip:
            padded, target = padded[:, ::-1], target[:, ::-1]
        return np.ascontiguousarray(padded.transpose(2, 0, 1)), np.ascontiguousarray(target)


def loss_of(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The focal loss of heat maps (CenterNet): the few pixels of a disc count as much
    as all the others, and the pixels beside a middle are not pushed hard to nothing."""
    sure = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    at = target > 0.99
    positive = -((1 - sure) ** 2 * torch.log(sure))[at].sum()
    negative = -(((1 - target) ** 4) * sure**2 * torch.log(1 - sure))[~at].sum()
    return (positive + negative) / max(int(at.sum()), 1)


def surest_spot(heat: torch.Tensor) -> tuple:
    """(x, y, how sure) of the highest point of a heat map."""
    place = int(torch.argmax(heat))
    y, x = divmod(place, heat.shape[1])
    return float(x), float(y), float(heat[y, x])


def score(model: nn.Module, frames: Frames, rows: list, device: str) -> tuple:
    """(F1, recall, precision, from how sure) over some frames: the surest spot of each
    is right if it is on a labelled disc, and counts from the sureness that gives the
    best F1."""
    model.eval()
    found = []  # (how sure, right) of each frame's surest spot
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for row in rows:
            picture, _ = frames.load(row)
            heat = torch.sigmoid(model(torch.from_numpy(picture)[None].to(device))[0]).float()
            x, y, sure = surest_spot(heat)
            right = any(
                np.hypot(x - d[0], y - d[1]) <= max(frames.near, 0.6 * max(d[2], d[3]))
                for d in row["discs"]
            )
            found.append((sure, right))
    with_disc = sum(1 for row in rows if row["discs"])
    best = (0.0, 0.0, 0.0, 0.0)
    for threshold in np.linspace(0.1, 0.9, 17):
        said = [right for sure, right in found if sure >= threshold]
        hits = sum(said)
        recall, precision = hits / max(with_disc, 1), hits / max(len(said), 1)
        f1 = 2 * recall * precision / max(recall + precision, 1e-9)
        if f1 > best[0]:
            best = (f1, recall, precision, float(threshold))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Folder of scripts/build_disc_frames.py")
    parser.add_argument("name", help="New folder in data/models/disc_heatmap")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--windows", action="store_true", help="Learn from windows of the frames")
    parser.add_argument("--one-frame", action="store_true", help="The labelled frame three times")
    parser.add_argument("--scratch", action="store_true", help="No pretrained encoder")
    parser.add_argument("--few", type=int, default=0, help="So many frames a split: a try-out")
    args = parser.parse_args()

    out = REPO / "data" / "models" / "disc_heatmap" / args.name
    if out.exists():
        sys.exit(f"{out} exists: nothing is overwritten")
    frames = Frames(REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset, args.one_frame)
    train, valid, test = (frames.split(name) for name in ("train", "valid", "test"))
    if args.few:
        train, valid, test = train[: args.few], valid[: args.few], test[: args.few]
    out.mkdir(parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    model = new_network(args.scratch).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = len(train) // args.batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=1e-3, total_steps=args.epochs * batches, pct_start=0.1
    )
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    print(
        f"{len(train)} frames to learn from ({sum(len(r['discs']) for r in train)} discs), "
        f"{len(valid)} to choose by, {len(test)} to test on; "
        f"{'one frame' if args.one_frame else 'three frames'} of "
        f"{frames.frame[0]} x {frames.frame[1]}{', learnt from windows' if args.windows else ''}",
        flush=True,
    )
    (out / "log.csv").write_text("epoch,loss,f1,recall,precision,threshold,seconds\n")
    best_f1, began = -1.0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        total = 0.0
        for start in range(0, batches * args.batch, args.batch):
            batch = [
                frames.load(
                    row,
                    flip=random.random() < 0.5,
                    jitter=(random.uniform(0.8, 1.2), random.uniform(-0.08, 0.08)),
                    window=frames.window_of(row) if args.windows else None,
                )
                for row in train[start : start + args.batch]
            ]
            pictures = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            targets = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(pictures)
            loss = loss_of(logits.float(), targets)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += float(loss)
        f1, recall, precision, threshold = score(model, frames, valid, device)
        seconds = time.perf_counter() - began
        kept = f1 > best_f1
        if kept:
            best_f1 = f1
            torch.save(
                {
                    "state": model.state_dict(),
                    "one_frame": args.one_frame,
                    "pretrained": not args.scratch,
                    "threshold": threshold,
                    "size": frames.size,
                    "frame": frames.frame,
                },
                out / "best.pt",
            )
        mean = total / max(batches, 1)
        with open(out / "log.csv", "a") as file:
            file.write(
                f"{epoch},{mean:.4f},{f1:.4f},{recall:.4f},{precision:.4f},"
                f"{threshold:.2f},{seconds:.0f}\n"
            )
        print(
            f"epoch {epoch:3d} of {args.epochs}  loss {mean:7.3f}  validation: F1 {f1:.3f} "
            f"(recall {recall:.3f}, precision {precision:.3f} at {threshold:.2f})  "
            f"{seconds / 60:5.1f} min{'  <- kept' if kept else ''}",
            flush=True,
        )
    model.load_state_dict(torch.load(out / "best.pt")["state"])
    for name, rows in (("validation", valid), ("test", test)):
        f1, recall, precision, threshold = score(model, frames, rows, device)
        print(
            f"best network on {name}: F1 {f1:.3f}, recall {recall:.3f}, precision "
            f"{precision:.3f} at {threshold:.2f} ({sum(1 for r in rows if r['discs'])} frames "
            f"with a disc of {len(rows)})",
            flush=True,
        )


if __name__ == "__main__":
    main()
