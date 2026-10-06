#!/usr/bin/env python3
"""Download the newest videos of a YouTube channel into data/raw/videos.

The app lists every video in that folder. Only the picture is downloaded (the app has no
use for sound), as H.264 at up to 1080p, which OpenCV reads and which needs no ffmpeg.
Videos that are already there are skipped.

    python scripts/download_videos.py                      # the 3 newest of Flatball Club
    python scripts/download_videos.py --count 5
    python scripts/download_videos.py --channel https://www.youtube.com/@OtherChannel
    python scripts/download_videos.py --list               # only show what would be taken

Needs yt-dlp (`pip install --no-deps yt-dlp`). Only download footage you may use.
"""

import argparse
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed: pip install --no-deps yt-dlp")

REPO = Path(__file__).resolve().parents[1]
VIDEOS = REPO / "data" / "raw" / "videos"
DEFAULT_CHANNEL = "https://www.youtube.com/@FlatballClub"
# H.264 first: OpenCV does not read every build of AV1 or VP9
VIDEO_FORMAT = "bv*[vcodec^=avc1][height<=1080]/bv*[ext=mp4][height<=1080]"


def newest_videos(channel: str, count: int) -> list:
    """The newest uploads of a channel: [{"id", "title", "duration"}]."""
    options = {"extract_flat": True, "playlistend": count, "quiet": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        listing = downloader.extract_info(f"{channel.rstrip('/')}/videos", download=False)
    return list(listing.get("entries") or [])[:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--count", type=int, default=3, help="how many of the newest videos")
    parser.add_argument("--list", action="store_true", help="show the videos, download nothing")
    args = parser.parse_args()

    videos = newest_videos(args.channel, args.count)
    for video in videos:
        minutes = (video.get("duration") or 0) / 60
        print(f"{video['id']}  {minutes:5.1f} min  {video.get('title')}")
    if args.list or not videos:
        return

    VIDEOS.mkdir(parents=True, exist_ok=True)
    options = {
        "format": VIDEO_FORMAT,
        # Names without spaces or special characters; the id keeps them apart
        "outtmpl": str(VIDEOS / "%(title).60s_%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "download_archive": str(VIDEOS / ".downloaded.txt"),  # Skips what is already there
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([f"https://www.youtube.com/watch?v={video['id']}" for video in videos])
    print(f"Videos are in {VIDEOS}")


if __name__ == "__main__":
    main()
