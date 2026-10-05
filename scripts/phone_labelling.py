#!/usr/bin/env python3
"""Label discs from a phone.

Starts a small web server on this PC. Open the address it prints in the phone's browser:
the page shows one frame of a game at a time and asks whether the disc model found the
disc, lets you tap the disc where it did not, or takes "no disc visible". Every answer
adds a frame to a disc-only dataset in data/raw/training_data, at full resolution.

The PC has to stay on; it holds the videos and runs the model.

From outside your home network, use a private link between phone and PC such as
Tailscale (install it on both, sign in with the same account) and open the Tailscale
address printed below. Do not forward the port on your router: anybody with the address
could then try to guess the key.

    python scripts/phone_labelling.py
    python scripts/phone_labelling.py --dataset labelled_discs_v2 --port 8765
"""

import argparse
import secrets
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.inference import FP16_KWARGS, load_detection_model  # noqa: E402
from ultimate_analysis.utils.model_files import default_model_path  # noqa: E402
from ultimate_analysis.web.label_server import create_server  # noqa: E402
from ultimate_analysis.web.label_session import LabelSession  # noqa: E402

KEY_FILE = REPO / "data" / "cache" / "phone_labelling.key"
# Lower than the app's threshold: a doubtful suggestion is a quick "no", a missing one costs
# three taps
SUGGESTION_CONFIDENCE = 0.1


def game_videos() -> Dict[str, int]:
    """Full game videos and their number of frames."""
    videos = {}
    for path in sorted((REPO / DEFAULT_PATHS["RAW_VIDEOS"]).glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frames > 0:
            videos[str(path)] = frames
    return videos


def make_frame_reader():
    captures: Dict[str, cv2.VideoCapture] = {}

    def read_frame(video: str, index: int) -> Optional[np.ndarray]:
        capture = captures.setdefault(video, cv2.VideoCapture(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None

    return read_frame


def make_disc_finder():
    loaded = load_detection_model(default_model_path("disc_detection"))
    if loaded is None:
        sys.exit("The default disc model could not be loaded")
    model, image_size = loaded
    disc_classes = [index for index, name in dict(model.names).items() if name == "disc"]

    def find_discs(frame: np.ndarray) -> List[Tuple[tuple, float]]:
        boxes = model.predict(
            frame,
            imgsz=image_size,
            conf=SUGGESTION_CONFIDENCE,
            classes=disc_classes or None,
            verbose=False,
            **FP16_KWARGS,
        )[0].boxes
        found = zip(boxes.xyxy.cpu().numpy().tolist(), boxes.conf.cpu().numpy().tolist())
        return sorted(((tuple(box), conf) for box, conf in found), key=lambda item: -item[1])

    return find_discs


def access_key() -> str:
    """The key the phone has to present; kept between starts so a bookmark keeps working."""
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(24)
    KEY_FILE.write_text(key)
    return key


def addresses() -> List[Tuple[str, str]]:
    """(description, IP address) under which this PC can be reached."""
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))  # No packet is sent
            found.append(("Home network", probe.getsockname()[0]))
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            found.append(("Tailscale (from anywhere)", result.stdout.split()[0]))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="labelled_discs_v1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address to listen on; 127.0.0.1 keeps the page to this PC (for trying it out)",
    )
    args = parser.parse_args()

    videos = game_videos()
    if not videos:
        sys.exit(f"No videos found in {DEFAULT_PATHS['RAW_VIDEOS']}")
    dataset_dir = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    session = LabelSession(dataset_dir, videos, make_frame_reader(), make_disc_finder())
    key = access_key()
    server = create_server(session, key, args.host, args.port)

    print(f"\nLabelling discs from {len(videos)} games into {dataset_dir}")
    print(f"{session.labelled_count()} frames labelled so far\n")
    print("Open one of these on the phone:")
    for description, address in addresses():
        print(f"  {description}: http://{address}:{args.port}/?key={key}")
    print("\nStop with Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
