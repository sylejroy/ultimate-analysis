#!/usr/bin/env python3
"""Download the newest videos of a YouTube channel into data/raw/videos.

The app lists every video in that folder. Only the picture is downloaded (the app has no
use for sound), as H.264 at up to 1080p, which OpenCV reads and which needs no ffmpeg.
Videos that are already there are skipped.

    python scripts/download_videos.py                      # the 3 newest of Flatball Club
    python scripts/download_videos.py --count 5
    python scripts/download_videos.py --random --count 3   # 3 games picked at random
    python scripts/download_videos.py --title "Truck Stop VS Chain Lightning" --title "Pacmen VS"
    python scripts/download_videos.py --channel https://www.youtube.com/@OtherChannel
    python scripts/download_videos.py --list               # only show what would be taken

Needs yt-dlp (`pip install --no-deps yt-dlp`). Only download footage you may use.
"""

import argparse
import random
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
# A random pick is made among this many of the newest uploads, and only among those at
# least this long: whole games, not highlights
RANDOM_POOL = 150
# A video asked for by its title is looked for among this many of the newest uploads
TITLE_POOL = 600
MIN_GAME_MINUTES = 15.0


def newest_videos(channel: str, count: int) -> list:
    """The newest uploads of a channel: [{"id", "title", "duration"}]."""
    options = {"extract_flat": True, "playlistend": count, "quiet": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        listing = downloader.extract_info(f"{channel.rstrip('/')}/videos", download=False)
    return list(listing.get("entries") or [])[:count]


def titled(channel: str, wanted: list) -> list:
    """The upload whose title holds each of the given texts (the newest, if several do)."""
    uploads = newest_videos(channel, TITLE_POOL)
    found = []
    for text in wanted:
        words = text.lower().split()
        match = next(
            (
                video
                for video in uploads
                if all(w in (video.get("title") or "").lower() for w in words)
            ),
            None,
        )
        if match is None:
            print(f"No upload among the newest {TITLE_POOL} has a title with: {text}")
        elif match not in found:
            found.append(match)
    return found


def random_games(channel: str, count: int) -> list:
    """Whole games picked at random among the channel's newer uploads, none already here."""
    archive = VIDEOS / ".downloaded.txt"
    have = set(archive.read_text().split()) if archive.exists() else set()
    games = [
        video
        for video in newest_videos(channel, RANDOM_POOL)
        if (video.get("duration") or 0) >= MIN_GAME_MINUTES * 60 and video["id"] not in have
    ]
    return random.sample(games, min(count, len(games)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--count", type=int, default=3, help="how many videos")
    parser.add_argument(
        "--random",
        action="store_true",
        help="whole games picked at random instead of the newest uploads",
    )
    parser.add_argument("--list", action="store_true", help="show the videos, download nothing")
    parser.add_argument(
        "--title",
        action="append",
        help="a video whose title holds all these words (any case); may be given several times",
    )
    args = parser.parse_args()

    if args.title:
        videos = titled(args.channel, args.title)
    else:
        pick = random_games if args.random else newest_videos
        videos = pick(args.channel, args.count)
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
