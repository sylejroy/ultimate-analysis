"""Small pictures of players made larger for somebody to look at.

A player is some 40 by 90 pixels in the drone footage, and the video is compressed.
Blown up as they are, the pixels and the blocks of the compression show. A
super-resolution network (Swin2SR) draws the picture at four times the size. The one
used was trained on pictures spoiled the way real ones are (blur, noise, compression):
it takes the blocks out and draws clean edges. The light one used before was trained
on clean pictures made smaller, and drew the blocks sharper (10 Oct 2026: side by side
on crops of four games, with one made for compressed pictures, which painted over what
was there, and with the frames before and after laid over the crop, which changed
little). It is for looking at only: nothing that is learned from or measured goes
through it. What it draws is its guess: a thing of a few pixels, a disc, is as well
looked for in the picture of the surroundings, which is shown as it is.

The network comes from the Hugging Face hub and is fetched the first time it is used;
after that it is read from this machine and the hub is not asked again.
Without it (no connection, no GPU) the crop is enlarged plainly.
"""

from typing import Any, Optional

import cv2
import numpy as np

from .logger import get_logger

logger = get_logger("ENLARGING")

NETWORK = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
FACTOR = 4  # What the network enlarges by
_network: Optional[Any] = None
_failed = False


def _load() -> Optional[Any]:
    global _network, _failed
    if _network is None and not _failed:
        try:
            import torch
            from transformers import Swin2SRForImageSuperResolution

            if not torch.cuda.is_available():
                raise RuntimeError("no GPU")
            try:
                # From this machine if it was fetched before: nothing is asked of the hub
                network = Swin2SRForImageSuperResolution.from_pretrained(
                    NETWORK, local_files_only=True
                )
            except OSError:
                network = Swin2SRForImageSuperResolution.from_pretrained(NETWORK)
            _network = network.eval().cuda().half()
        except Exception as error:
            logger.warning(f"Crops are enlarged without the super-resolution network: {error}")
            _failed = True
    return _network


def enlarged(picture: np.ndarray) -> np.ndarray:
    """A BGR picture at four times its size, cleaner and sharper than stretched where
    the network is there."""
    network = _load()
    height, width = picture.shape[:2]
    if network is not None:
        try:
            import torch

            # The network takes sides that are multiples of eight
            padded = cv2.copyMakeBorder(
                picture, 0, (-height) % 8, 0, (-width) % 8, cv2.BORDER_REFLECT
            )
            given = torch.from_numpy(padded[:, :, ::-1].copy()).permute(2, 0, 1)[None]
            with torch.no_grad():
                drawn = network(pixel_values=given.cuda().half() / 255.0).reconstruction[0]
            drawn = (drawn.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(
                np.uint8
            )
            return np.ascontiguousarray(drawn[: FACTOR * height, : FACTOR * width, ::-1])
        except Exception as error:
            logger.warning(f"The super-resolution network failed on a crop: {error}")
    return cv2.resize(picture, (FACTOR * width, FACTOR * height), interpolation=cv2.INTER_CUBIC)
