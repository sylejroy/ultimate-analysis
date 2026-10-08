"""Render a stretch of a game as the app shows it, to a video file.

Every frame goes through the analysis pipeline, as in the main tab. The camera view with
its overlays, the top-down view, and a strip with who holds the disc are put together
into one picture per frame. Nothing is skipped to keep up, unlike live playback, so the
video shows what the analysis does when it gets every frame.

OpenCV writes very large files. With imageio-ffmpeg installed
(`pip install --no-deps imageio-ffmpeg`) the video is compressed afterwards to a size
that can be passed on.

Usage:
    python scripts/render_demo.py VIDEO --start 12:30 --seconds 60 --output data/demo/demo.mp4
"""

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis import pipeline as pipeline_module  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.homography import load_default_matrix  # noqa: E402
from ultimate_analysis.processing.player_id import discard_pending_readings  # noqa: E402
from ultimate_analysis.rendering.tracks import UNKNOWN_TEAM_COLOUR  # noqa: E402

WIDTH, HEIGHT = 1920, 1080
CAMERA_SIZE = (1440, 810)  # The camera view, top left
STRIP_SECONDS = 20.0  # How far back the strip of who held the disc reaches
BACKGROUND = (30, 30, 30)
TEXT = (235, 235, 235)
FAINT = (150, 150, 150)


def seconds_of(text: str) -> float:
    """Seconds from "90", "1:30" or "0:01:30"."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def put_text(picture, text, place, scale=0.7, colour=TEXT, thickness=1):
    cv2.putText(
        picture, text, place, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA
    )


def is_number(read) -> bool:
    """Whether what the jersey reader gave for a player is a number (not "Unknown")."""
    return bool(read) and str(read).isdigit()


def compose(result, held: deque, frames_per_second: float, title: str, clock: str) -> np.ndarray:
    """One picture of the demo: camera view, top-down view, and the possession strip.

    `held`: for each frame written, (colour of the holder's team, the holder's ID), or
    None for nobody.
    """
    picture = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    camera_width, camera_height = CAMERA_SIZE
    picture[:camera_height, :camera_width] = cv2.resize(
        result.main_view, CAMERA_SIZE, interpolation=cv2.INTER_AREA
    )

    # The top-down view fills the column on the right
    column_left, column_width = camera_width + 12, WIDTH - camera_width - 24
    put_text(picture, "Top-down view", (column_left, 30), 0.7)
    top = 44
    if result.top_down_view is not None:
        view = result.top_down_view
        scale = min(column_width / view.shape[1], (HEIGHT - top - 12) / view.shape[0])
        size = (int(view.shape[1] * scale), int(view.shape[0] * scale))
        view = cv2.resize(view, size, interpolation=cv2.INTER_AREA)
        left = column_left + (column_width - size[0]) // 2
        picture[top : top + size[1], left : left + size[0]] = view
    else:
        put_text(picture, result.top_down_message[:40], (column_left, top + 40), 0.5, FAINT)

    # Below the camera view: what is going on
    base = camera_height
    put_text(picture, title, (20, base + 40), 0.9, TEXT, 2)
    put_text(picture, clock, (camera_width - 150, base + 40), 0.8, FAINT)
    players = len(result.tracks)
    numbers = sum(1 for number, _ in result.player_ids.values() if is_number(number))
    put_text(
        picture,
        f"{players} players followed, {numbers} jersey numbers read",
        (20, base + 80),
        0.7,
        FAINT,
    )
    holder = result.holder_id
    if holder is None:
        put_text(picture, "Disc: in the air or not seen", (20, base + 125), 0.8, FAINT)
    else:
        number = result.player_ids.get(holder, ("", None))[0]
        who = f"number {number}" if is_number(number) else f"player {holder}"
        cv2.circle(picture, (34, base + 117), 12, held[-1][0], -1)
        put_text(picture, f"Disc: {who}", (60, base + 125), 0.8)

    # Who held the disc over the last seconds, newest on the right
    strip_left, strip_right, strip_top = 20, camera_width - 20, base + 160
    put_text(picture, "Possession", (strip_left, strip_top + 60), 0.5, FAINT)
    put_text(picture, "now", (strip_right - 34, strip_top + 60), 0.5, FAINT)
    cv2.rectangle(picture, (strip_left, strip_top), (strip_right, strip_top + 36), (55, 55, 55), -1)
    length = int(STRIP_SECONDS * frames_per_second)
    step = (strip_right - strip_left) / length
    runs = []  # [first position, last position, (colour, holder)] of one holder's stretch
    for position, earlier in enumerate(held):
        if earlier is None:
            continue
        x = strip_left + (length - len(held) + position) * step
        cv2.rectangle(
            picture,
            (int(x), strip_top),
            (int(x + step) + 1, strip_top + 36),
            earlier[0],
            -1,
        )
        if runs and runs[-1][1] == position - 1 and runs[-1][2] == earlier:
            runs[-1][1] = position
        else:
            runs.append([position, position, earlier])
    # The holder's jersey number on their stretch, where it has been read and fits
    for first, last, (colour, holder) in runs:
        number = result.player_ids.get(holder, ("", None))[0]
        if not is_number(number):
            continue
        text = f"#{number}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        left = strip_left + (length - len(held) + first) * step
        right = strip_left + (length - len(held) + last + 1) * step
        if text_width + 8 > right - left:
            continue
        ink = (20, 20, 20) if sum(colour) > 380 else (245, 245, 245)
        place = (int((left + right - text_width) / 2), strip_top + 18 + text_height // 2)
        put_text(picture, text, place, 0.65, ink, 2)
    return picture


def open_writer(path: Path, frames_per_second: float) -> cv2.VideoWriter:
    """A writer for H.264 if this OpenCV can write it, else MPEG-4."""
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*code), frames_per_second, (WIDTH, HEIGHT)
        )
        if writer.isOpened():
            return writer
        writer.release()
    sys.exit(f"No video writer could be opened for {path}")


def compress(written: Path, output: Path) -> None:
    """Compress what OpenCV wrote, if ffmpeg is there; else keep it as it is."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        written.replace(output)
        return
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(written)]
    command += ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]
    command += ["-movflags", "+faststart", str(output)]
    if subprocess.run(command).returncode == 0:
        written.unlink()
    else:
        written.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", default="0", help="Where to start: seconds or m:ss")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lead-in",
        type=float,
        default=10.0,
        help="Seconds analysed before the start and not shown, so that players are "
        "followed and numbers read when the video begins",
    )
    parser.add_argument(
        "--top-down",
        choices=("field", "calibration"),
        default="field",
        help="What the top-down view is made from",
    )
    parser.add_argument(
        "--every",
        type=int,
        help="Write every n-th frame; by default as many as give about 30 frames per second",
    )
    parser.add_argument("--title", default="Ultimate Analysis")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        sys.exit(f"Cannot open {args.video}")
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(seconds_of(args.start) * frames_per_second)
    first = max(0, start - int(args.lead_in * frames_per_second))
    last = start + int(args.seconds * frames_per_second)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    every = args.every or max(1, round(frames_per_second / 30.0))
    # The processing rate shown in the app says nothing here: every frame is waited for
    pipeline_module.draw_fps_overlay = lambda *_, **__: None
    pipeline = AnalysisPipeline()
    pipeline.new_video(str(args.video))
    pipeline.set_frame_rate(frames_per_second)
    if args.top_down == "calibration":
        pipeline.homography_matrix = load_default_matrix()
    options = PipelineOptions(top_down_source=args.top_down)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_rate = frames_per_second / every
    written = args.output.with_name(args.output.stem + ".uncompressed.mp4")
    writer = open_writer(written, out_rate)
    held: deque = deque(maxlen=int(STRIP_SECONDS * out_rate))
    began = time.perf_counter()
    try:
        for index in range(first, last):
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process(frame, index, options)
            if index < start or (index - start) % every:
                continue
            holder = (
                None
                if result.holder_id is None
                else (result.possession_colour or UNKNOWN_TEAM_COLOUR, result.holder_id)
            )
            if held and held[-1] != holder:
                # A change is confirmed a moment after it happened: the strip is put right
                for back in range(
                    1, min(len(held), (index - result.possession_since) // every) + 1
                ):
                    held[-back] = holder
            held.append(holder)
            seconds = index / frames_per_second
            clock = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
            writer.write(compose(result, held, out_rate, args.title, clock))
    finally:
        writer.release()
        capture.release()
        discard_pending_readings()
    compress(written, args.output)
    megabytes = args.output.stat().st_size / 1e6
    print(
        f"{args.output}: {args.seconds:.0f} s at {out_rate:.0f} frames per second, "
        f"{megabytes:.0f} MB, rendered in {time.perf_counter() - began:.0f} s"
    )


if __name__ == "__main__":
    main()
