"""What is asked and what is answered in the two labelling tabs that take a yes or no.

"Same player?": two tracks of a dataset of player crops (scripts/build_reid_dataset.py)
are shown side by side and somebody says whether they show one player. The answers are
labels for telling players apart by their looks, and for the tracker as well: a track
that shows several people is one the tracker got wrong. A picture of a track that
shows somebody else than the rest of it can be taken out before answering: the answer is
then about the pictures left, and every picture shown is written down as being of the
track's player or not. Most pairs asked about are ones the network that tells players
apart is unsure of, the rest are drawn at random; which it was is written down too.

"Holding the disc?": a player of a frame of a holder dataset
(scripts/build_holder_sighted_dataset.py) is shown and somebody says whether they hold
the disc. One disc is in play: a yes for one player is a no for every other player of
that frame. A box that shows two players (the thrower and the mark, mostly) is neither
a holder's nor another's: it is written down as such, with the place in the frame of
the one who holds the disc if somebody pointed at them, and is no crop to learn from.

The source datasets are only read. The answers are added, a line each, to a file in a
dataset folder of their own.
"""

import csv
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# What can be said of two tracks. The two in the middle are no longer asked for (a track
# of several people has its pictures taken out one by one) and stand in older answers.
PAIR_ANSWERS = ("same", "different", "first_mixed", "second_mixed", "unsure")
PAIR_COLUMNS = (
    "source",
    "video",
    "first_stretch",
    "first_player",
    "second_stretch",
    "second_player",
    "answer",
    "time",
)
# Why a pair was asked about: the network that tells players apart is unsure of it, or
# it was drawn at random
UNSURE, RANDOM = "unsure", "random"
ASKED_COLUMNS = (
    "video",
    "first_stretch",
    "first_player",
    "second_stretch",
    "second_player",
    "why",
    "likeness",
    "time",
)
UNSURE_SHARE = 0.7  # Of the pairs asked about, where the network's vectors are there
# How alike (cosine of their vectors) two tracks are that the network is unsure of. Of
# the pairs answered by hand on 10 Oct it put the same players at 0.85 and others at
# 0.1 to 0.3, in the middle case
UNSURE_LIKENESS = (0.4, 0.8)
# A picture of a track that was shown: of the player the track is mostly of (1), or
# of somebody else (0)
PICTURE_COLUMNS = ("source", "video", "stretch", "player", "frame", "of_the_player", "time")
HOLDER_COLUMNS = ("source", "video", "stretch", "frame", "player", "holding", "how", "time")
# How a player of a frame came by their answer
SAID, BY_ANOTHER = "said", "another player holds the disc in this frame"
# Boxes that show two players, beside the file of the answers: x and y are where in
# the frame the one who holds the disc is, empty if neither does or nobody pointed
TWO_COLUMNS = ("source", "video", "stretch", "frame", "player", "x", "y", "time")
# Holders clicked who are none of the dataset's players of the frame: the box that
# was clicked and where in it, in frame pixels; two = 1 if the box shows two players
CLICKED_COLUMNS = (
    "source",
    "video",
    "stretch",
    "frame",
    "x1",
    "y1",
    "x2",
    "y2",
    "x",
    "y",
    "two",
    "time",
)
# Answers that were looked at again, so that they are not asked about a third time
RECHECKED_COLUMNS = ("video", "stretch", "frame", "player", "time")
# A box this wide for its height may show two players: half of all boxes are narrower
# than 0.5, one in four is wider than 0.6
WIDE_BOX = 0.6
# From when a box could be said to show two players: answers given before are the ones
# to look at again, those since were given with that to choose from
TWO_SAID_FROM = "2026-10-10T18:05:35"
MIN_TRACK_CROPS = 4  # A track of fewer crops shows too little to judge

TrackKey = Tuple[str, str, str]  # Video, stretch, player
FrameKey = Tuple[str, str, str]  # Video, stretch, frame


@dataclass
class Track:
    """The crops of one track of a stretch."""

    key: TrackKey
    light_shirt: str
    crops: List[Tuple[int, Path]] = field(default_factory=list)  # (frame, file), in order
    numbers: Counter = field(default_factory=Counter)  # Jersey numbers read on it

    @property
    def number(self) -> str:
        """The number read most often on the track, or "" if none was."""
        return self.numbers.most_common(1)[0][0] if self.numbers else ""

    def chosen(self, count: int) -> List[Tuple[int, Path]]:
        """Crops (frame, file) from the whole length of the track, as many as asked for
        at most."""
        if len(self.crops) <= count:
            return list(self.crops)
        step = (len(self.crops) - 1) / (count - 1)
        return [self.crops[round(index * step)] for index in range(count)]

    def spread(self, count: int) -> List[Path]:
        """The files of `chosen`."""
        return [path for _, path in self.chosen(count)]


def _append(path: Path, columns: Sequence[str], rows: List[dict]) -> None:
    """Add lines to a file of answers, which is begun with its heading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, columns)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def stretch_folder(crops: Path, video: str, stretch: str) -> Path:
    """Where the crops of a stretch are: the folder is the stretch's number, written
    with two digits in some datasets and six in others."""
    for digits in (2, 6):
        folder = crops / video / f"{int(stretch):0{digits}d}"
        if folder.exists():
            return folder
    return crops / video / stretch


# ------------------------------------------------------------------ same player?


def load_tracks(dataset: Path) -> Dict[TrackKey, Track]:
    """The tracks of a dataset of player crops that are long enough to judge."""
    tracks: Dict[TrackKey, Track] = {}
    folders: Dict[Tuple[str, str], Path] = {}
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            key = (row["video"], row["stretch"], row["player"])
            track = tracks.get(key)
            if track is None:
                track = tracks[key] = Track(key, row.get("light_shirt", ""))
            place = key[:2]
            if place not in folders:
                folders[place] = stretch_folder(dataset / "crops", *place)
            frame = int(row["frame"])
            track.crops.append((frame, folders[place] / f"{row['player']}_{frame}.jpg"))
            if row.get("number"):
                track.numbers[row["number"]] += 1
    for track in tracks.values():
        track.crops.sort()
    return {key: track for key, track in tracks.items() if len(track.crops) >= MIN_TRACK_CROPS}


def pair_of(first: TrackKey, second: TrackKey) -> frozenset:
    return frozenset((first, second))


def judged_pairs(answers: Path) -> Set[frozenset]:
    """The pairs of tracks that have an answer already."""
    return {
        pair_of(
            (row["video"], row["first_stretch"], row["first_player"]),
            (row["video"], row["second_stretch"], row["second_player"]),
        )
        for row in _read(answers)
    }


def choose_pair(
    tracks: Dict[TrackKey, Track],
    judged: Set[frozenset],
    rng: random.Random,
    vectors: Optional[Dict[TrackKey, np.ndarray]] = None,
    tries: int = 200,
) -> Optional[Tuple[Track, Track, str]]:
    """Two tracks to ask about and why they are asked about (`UNSURE` or `RANDOM`), or
    None if none are found.

    Both are of the same game and the same kind of shirt. Two drawn at random are
    different players nearly every time, and a network that tells players apart knows
    that already. So where its vectors of the tracks are given, most pairs
    (`UNSURE_SHARE`) are ones it is unsure of: neither alike enough to be one player
    nor unlike enough to be two (`UNSURE_LIKENESS`). The rest are drawn at random: a
    fair sample, which the others are not.
    """
    by_game: Dict[Tuple[str, str], List[Track]] = defaultdict(list)
    for track in tracks.values():
        by_game[track.key[0], track.light_shirt].append(track)
    everyone = list(tracks.values())
    unsure = bool(vectors) and rng.random() < UNSURE_SHARE
    for _ in range(tries if everyone else 0):
        first = rng.choice(everyone)
        others = [
            other
            for other in by_game[first.key[0], first.light_shirt]
            if other is not first and pair_of(first.key, other.key) not in judged
        ]
        if unsure and first.key in vectors:
            low, high = UNSURE_LIKENESS
            others = [
                other
                for other in others
                if other.key in vectors
                and low <= float(vectors[first.key] @ vectors[other.key]) <= high
            ]
        elif unsure:
            continue
        if others:
            return first, rng.choice(others), UNSURE if unsure else RANDOM
    if unsure:
        # Nothing left that the network is unsure of
        return choose_pair(tracks, judged, rng, None, tries)
    return None


def save_pair(answers: Path, source: str, first: Track, second: Track, answer: str) -> None:
    """Write down what was said of two tracks."""
    if answer not in PAIR_ANSWERS:
        raise ValueError(f"'{answer}' is not one of {PAIR_ANSWERS}")
    row = {
        "source": source,
        "video": first.key[0],
        "first_stretch": first.key[1],
        "first_player": first.key[2],
        "second_stretch": second.key[1],
        "second_player": second.key[2],
        "answer": answer,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    _append(answers, PAIR_COLUMNS, [row])


def save_asked(
    asked: Path, first: Track, second: Track, why: str, likeness: Optional[float]
) -> None:
    """Write down why a pair that was answered had been asked about, and how alike the
    network took the two tracks to be (None: not known)."""
    row = {
        "video": first.key[0],
        "first_stretch": first.key[1],
        "first_player": first.key[2],
        "second_stretch": second.key[1],
        "second_player": second.key[2],
        "why": why,
        "likeness": "" if likeness is None else round(likeness, 3),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    _append(asked, ASKED_COLUMNS, [row])


def what_was_said(folder: Path) -> Tuple[List[Tuple[TrackKey, TrackKey]], Set[TrackKey], Set]:
    """The answers of a folder as training takes them: the pairs of tracks said to be
    one player, the tracks said to show several people, and the pictures taken out of
    their track (video, stretch, player, frame). The last answer for a pair counts."""
    last = {}
    for row in _read(folder / "pairs.csv"):
        first = (row["video"], row["first_stretch"], row["first_player"])
        second = (row["video"], row["second_stretch"], row["second_player"])
        last[pair_of(first, second)] = (first, second, row["answer"])
    same = [(first, second) for first, second, answer in last.values() if answer == "same"]
    mixed = {first for first, _, answer in last.values() if answer == "first_mixed"}
    mixed |= {second for _, second, answer in last.values() if answer == "second_mixed"}
    others = {
        (row["video"], row["stretch"], row["player"], int(row["frame"]))
        for row in _read(folder / "pictures.csv")
        if row["of_the_player"] == "0"
    }
    return same, mixed, others


def save_pictures(
    pictures: Path, source: str, track: Track, shown: Sequence[int], others: Set[int]
) -> None:
    """Write down, for the pictures of a track that were shown (their frames), which
    are of the track's player and which (`others`) are of somebody else."""
    now = datetime.now().isoformat(timespec="seconds")
    video, stretch, player = track.key
    _append(
        pictures,
        PICTURE_COLUMNS,
        [
            {
                "source": source,
                "video": video,
                "stretch": stretch,
                "player": player,
                "frame": frame,
                "of_the_player": int(frame not in others),
                "time": now,
            }
            for frame in shown
        ],
    )


# ------------------------------------------------------------------ holding the disc?


def load_holder_frames(dataset: Path, without: bool = False) -> Dict[FrameKey, List[dict]]:
    """The players of every frame of a holder dataset in which the dataset has a holder;
    with `without`, of the frames in which it has none as well (the disc flies, by its
    marks): a network that is only ever shown frames with a holder never learns that
    there may be none.

    The player the dataset takes for the holder comes first, then the one nearest to
    them, then the others: the order to ask in.
    """
    frames: Dict[FrameKey, List[dict]] = defaultdict(list)
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            frames[row["video"], row["stretch"], row["frame"]].append(row)
    asked = {}
    for key, rows in frames.items():
        if without or any(row["holding"] == "1" for row in rows):
            rows.sort(key=lambda row: (row["holding"] != "1", row.get("nearest") != "1"))
            asked[key] = rows
    return asked


# Of the frames the "Click the holder" page asks about, this share are frames in which
# the dataset has no holder
NO_HOLDER_SHARE = 0.35


# What scripts/find_flying_disc_frames.py writes of the frames a holder dataset has no
# holder in: what each shows of the disc, and where the disc is. The dataset's own mark
# "the disc flies" is right for a third of them
FLYING = "flying"
FLYING_COLUMNS = ("video", "stretch", "frame", "kind", "x", "y")
FLYING_FOLDER = Path("data") / "cache" / "flying_frames"


def flying_frames(source: str, folder: Path = FLYING_FOLDER) -> Dict[FrameKey, Tuple[float, float]]:
    """The frames of a holder dataset in which the disc was found to fly, each with
    where the disc is in it (frame pixels); none if they have not been looked for."""
    return {
        (row["video"], row["stretch"], row["frame"]): (float(row["x"]), float(row["y"]))
        for row in _read(folder / f"{source}.csv")
        if row["kind"] == FLYING and row["x"] and row["y"]
    }


def said_of_the_frame(answers: Path) -> Set[FrameKey]:
    """The frames of which it was said at once that nobody holds the disc (the N key of
    the "Click the holder" page, which shows the whole frame): every player of them a
    no, said at the same moment."""
    last: Dict[Tuple[str, str, str, str], dict] = {}
    for line in _read(answers):
        last[_player_key(line)] = line
    by_frame: Dict[FrameKey, List[dict]] = defaultdict(list)
    for key, line in last.items():
        by_frame[key[:3]].append(line)
    return {
        frame
        for frame, lines in by_frame.items()
        if len(lines) >= 2
        and all(line["holding"] == "0" and line["how"] == SAID for line in lines)
        and len({line["time"] for line in lines}) == 1
    }


CROWDED_SHARE = 0.7  # Of the frames asked about, while there are such frames left


def crowded(rows: List[dict]) -> bool:
    """Whether another player of the frame stands in the box of the one the dataset takes
    for the holder: the mark in front of the thrower, a defender at the catch. These are
    the frames in which the disc is given to the wrong one of two, and the ones worth
    most to have an answer for."""

    def box(row: dict) -> Tuple[float, ...]:
        return tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))

    try:
        holders = [box(row) for row in rows if row["holding"] == "1"]
        others = [box(row) for row in rows if row["holding"] != "1"]
    except (KeyError, ValueError):
        return False
    return any(
        a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3] for a in holders for b in others
    )


def _player_key(row: dict) -> Tuple[str, str, str, str]:
    return (row["video"], row["stretch"], row["frame"], row["player"])


def answered_players(answers: Path) -> Set[Tuple[str, str, str, str]]:
    """Video, stretch, frame and player of everyone who has an answer already."""
    return {_player_key(row) for row in _read(answers)} | set(boxes_of_two(answers))


def _beside(answers: Path, name: str) -> Path:
    return answers.with_name(name)


def boxes_of_two(answers: Path) -> Dict[Tuple[str, str, str, str], Optional[Tuple[float, float]]]:
    """The boxes said to show two players: video, stretch, frame and player -> where in
    the frame the one who holds the disc is, None if that was not said."""
    found = {}
    for row in _read(_beside(answers, "two_in_box.csv")):
        point = (float(row["x"]), float(row["y"])) if row["x"] and row["y"] else None
        found[_player_key(row)] = point
    return found


def two_in_box_answers(
    rows: List[dict], player: str, point: Optional[Tuple[float, float]]
) -> List[Tuple[dict, int, str]]:
    """What a holder pointed at in a box of two says of the frame's other players:
    whoever's box the place is not in does not hold the disc. Nothing of a box the
    place is in as well (it may be the holder's own), and nothing without a place."""
    if point is None:
        return []
    entries = []
    for row in rows:
        try:
            x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
        except (KeyError, ValueError):
            continue
        inside = x1 <= point[0] <= x2 and y1 <= point[1] <= y2
        if row["player"] != player and not inside:
            entries.append((row, 0, BY_ANOTHER))
    return entries


def save_two_in_box(
    answers: Path,
    source: str,
    row: dict,
    point: Optional[Tuple[float, float]],
    entries: List[Tuple[dict, int, str]],
) -> str:
    """Write down that a box shows two players, where the holder is in the frame if
    that was pointed at, and what follows for the others; the time written with them."""
    now = save_holders(answers, source, entries)
    _append(
        _beside(answers, "two_in_box.csv"),
        TWO_COLUMNS,
        [
            {
                "source": source,
                **{name: row[name] for name in ("video", "stretch", "frame", "player")},
                "x": "" if point is None else round(point[0], 1),
                "y": "" if point is None else round(point[1], 1),
                "time": now,
            }
        ],
    )
    return now


def _rewrite(path: Path, columns: Sequence[str], kept: List[dict]) -> None:
    """Write a file anew beside the old one and put it in its place, so that it is
    whole at every moment."""
    anew = path.with_suffix(".csv.new")
    with open(anew, "w", newline="") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(kept)
    anew.replace(path)


def take_back_two_in_box(answers: Path, row: dict, time: str) -> int:
    """Take out again that a box shows two players, as written at that time."""
    path = _beside(answers, "two_in_box.csv")
    lines = _read(path)
    kept = [line for line in lines if line["time"] != time or _player_key(line) != _player_key(row)]
    if len(kept) < len(lines):
        _rewrite(path, TWO_COLUMNS, kept)
    return len(lines) - len(kept)


def keep_a_copy(answers: Path) -> None:
    """Before answers are looked at again and replaced: the file as it was, once."""
    copy = answers.with_suffix(".csv.before_recheck")
    if answers.exists() and not copy.exists():
        shutil.copyfile(answers, copy)


def save_rechecked(answers: Path, row: dict) -> None:
    """Write down that an answer was looked at again."""
    now = datetime.now().isoformat(timespec="seconds")
    line = {name: row[name] for name in ("video", "stretch", "frame", "player")}
    _append(_beside(answers, "rechecked.csv"), RECHECKED_COLUMNS, [{**line, "time": now}])


def wide_answers(
    answers: Path,
    frames: Dict[FrameKey, List[dict]],
    least: float = WIDE_BOX,
    before: str = TWO_SAID_FROM,
) -> List[dict]:
    """The answers given for boxes wide enough to show two players, the widest first,
    without those looked at again already: to be asked about once more, now that a box
    of two can be said to be one. Only answers given `before` that could be said."""
    done = {_player_key(row) for row in _read(_beside(answers, "rechecked.csv"))}
    found = []
    for answer in said_answers(answers):
        key = (*answer["frame"], answer["player"])
        row = next((r for r in frames.get(answer["frame"], []) if r["player"] == key[3]), None)
        if row is None or key in done:
            continue
        done.add(key)  # The last answer for a player is the one that counts
        if answer["time"] >= before:
            continue
        try:
            x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
        except (KeyError, ValueError):
            continue
        ratio = (x2 - x1) / max(y2 - y1, 1.0)
        if ratio >= least:
            found.append({**answer, "ratio": ratio})
    return sorted(found, key=lambda answer: -answer["ratio"])


def holders_said(answers: Path) -> Dict[Tuple[str, str, str, str], int]:
    """Who holds the disc by the answers given: video, stretch, frame and player ->
    1 or 0. The last answer for a player counts."""
    return {
        (row["video"], row["stretch"], row["frame"], row["player"]): int(row["holding"])
        for row in _read(answers)
    }


def holder_answers(rows: List[dict], player: str, holding: bool) -> List[Tuple[dict, int, str]]:
    """What an answer for one player of a frame says of the frame's players: (row,
    holding, how it is known). A yes is a no for all the others: one disc is in play."""
    said = [(row, int(holding), SAID) for row in rows if row["player"] == player]
    if not holding:
        return said
    return said + [(row, 0, BY_ANOTHER) for row in rows if row["player"] != player]


def nobody_holds(rows: List[dict]) -> List[Tuple[dict, int, str]]:
    """What "nobody shown holds the disc" says of a frame's players: a no for each."""
    return [(row, 0, SAID) for row in rows]


def others_hold(rows: List[dict]) -> List[Tuple[dict, int, str]]:
    """What a holder who is none of a frame's players says of them: a no for each."""
    return [(row, 0, BY_ANOTHER) for row in rows]


def save_clicked(
    answers: Path,
    source: str,
    frame: FrameKey,
    box: Sequence[float],
    point: Tuple[float, float],
    two: bool,
    time: str,
) -> None:
    """Write down a holder who is none of the dataset's players of a frame: the box
    clicked, where in it, and whether the box shows two players."""
    line = {"source": source, "video": frame[0], "stretch": frame[1], "frame": frame[2]}
    line.update(zip(("x1", "y1", "x2", "y2"), (round(float(value), 1) for value in box)))
    line.update(x=round(point[0], 1), y=round(point[1], 1), two=int(two), time=time)
    _append(_beside(answers, "clicked_holders.csv"), CLICKED_COLUMNS, [line])


def take_back_clicked(answers: Path, frame: FrameKey, time: str) -> int:
    """Take a holder written by `save_clicked` at that time out again."""
    path = _beside(answers, "clicked_holders.csv")
    lines = _read(path)
    kept = [
        line
        for line in lines
        if line["time"] != time or (line["video"], line["stretch"], line["frame"]) != frame
    ]
    if len(kept) < len(lines):
        _rewrite(path, CLICKED_COLUMNS, kept)
    return len(lines) - len(kept)


def save_holders(answers: Path, source: str, entries: List[Tuple[dict, int, str]]) -> str:
    """Write down who holds the disc and who does not; the time written with them, by
    which the answer can be taken back."""
    now = datetime.now().isoformat(timespec="seconds")
    if not entries:
        return now
    _append(
        answers,
        HOLDER_COLUMNS,
        [
            {
                "source": source,
                "video": row["video"],
                "stretch": row["stretch"],
                "frame": row["frame"],
                "player": row["player"],
                "holding": holding,
                "how": how,
                "time": now,
            }
            for row, holding, how in entries
        ],
    )
    return now


def last_holder_answers(answers: Path, count: int) -> List[dict]:
    """The last answers given, the newest first (see `said_answers`)."""
    return said_answers(answers, count)


def said_answers(answers: Path, count: Optional[int] = None) -> List[dict]:
    """The answers given, the newest first: for each the frame, the player it was
    given for, what was said, when, and every line written with it (player, holding,
    how), as `take_back_holders` needs them."""
    lines = _read(answers)
    found = []
    for line in reversed(lines):
        if line["how"] != SAID:
            continue
        frame = (line["video"], line["stretch"], line["frame"])
        with_it = [
            (other["player"], int(other["holding"]), other["how"])
            for other in lines
            if other["time"] == line["time"]
            and (other["video"], other["stretch"], other["frame"]) == frame
            and (other["player"] == line["player"] or other["how"] == BY_ANOTHER)
        ]
        found.append(
            {
                "frame": frame,
                "player": line["player"],
                "holding": line["holding"] == "1",
                "time": line["time"],
                "lines": with_it,
            }
        )
        if len(found) == count:
            break
    return found


def take_back_holders(answers: Path, entries: List[Tuple[dict, int, str]], time: str) -> int:
    """Take an answer out of the file again: the lines `save_holders` wrote for it at
    that time. The file is written anew beside the old one and put in its place, so that
    it is whole at every moment. Returns how many lines were taken out."""
    written = {
        (row["video"], row["stretch"], row["frame"], row["player"], str(holding), how)
        for row, holding, how in entries
    }
    lines = _read(answers)
    kept = [
        line
        for line in lines
        if line["time"] != time
        or tuple(line[name] for name in ("video", "stretch", "frame", "player", "holding", "how"))
        not in written
    ]
    if len(kept) < len(lines):
        _rewrite(answers, HOLDER_COLUMNS, kept)
    return len(lines) - len(kept)
