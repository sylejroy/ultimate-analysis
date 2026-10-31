#!/usr/bin/env python3
"""Start the app, visit every tab, play the default video for a while, close it, and
print what the log said at WARNING or above.

The check the guidelines ask for after a change to the GUI, driven by timers instead
of a hand. Needs a real window (not QT_QPA_PLATFORM=offscreen), the default video of
configs/default.yaml in data/raw/videos, and the models. Reads nothing else and writes
nothing.

    python scripts/check_app.py --play-seconds 30
"""

import argparse
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# A tab is visited every few seconds, the video is played after the last, and the app
# is closed a moment after it is paused
TAB_SECONDS = 6.0
SETTLE_SECONDS = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--play-seconds", type=float, default=30.0, help="How long to play")
    args = parser.parse_args()

    os.chdir(REPO)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from PyQt5.QtCore import QTimer

    from ultimate_analysis.gui.main_app import UltimateAnalysisApp, create_application

    said = []

    class Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                said.append(self.format(record))

    for name in ["", *logging.Logger.manager.loggerDict]:
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.addHandler(Catch())

    app = create_application()
    from ultimate_analysis.gui.main import main_tab  # noqa: F401  (the slow import)

    window = UltimateAnalysisApp()
    window.show()

    def step(what: str) -> None:
        print(f"[{what}]", flush=True)

    def visit(index: int) -> None:
        window.tab_widget.setCurrentIndex(index)
        app.processEvents()
        step(f"tab {index}: {window.tab_widget.tabText(index)}")

    def play() -> None:
        info = window.main_tab.video_info or {}
        step(f"video: {info.get('path')}")
        window.main_tab._toggle_play_pause()
        step("playing")

    def pause() -> None:
        window.main_tab._toggle_play_pause()
        step("paused")

    def finish() -> None:
        step("closing")
        window.close()
        app.processEvents()
        print("--- log at WARNING or above:")
        for line in said:
            print(line)
        print(f"--- {len(said)} lines")
        app.quit()

    tabs = window.tab_widget.count()
    plan = [(TAB_SECONDS * (n + 1), (lambda i=i: visit(i))) for n, i in enumerate(range(1, tabs))]
    back = TAB_SECONDS * (tabs + 1)
    plan += [
        (back, lambda: visit(0)),
        (back + SETTLE_SECONDS, play),
        (back + SETTLE_SECONDS + args.play_seconds, pause),
        (back + 2 * SETTLE_SECONDS + args.play_seconds, finish),
    ]
    for seconds, what in plan:
        QTimer.singleShot(int(seconds * 1000), what)
    QTimer.singleShot(int((plan[-1][0] + 60) * 1000), app.quit)  # However it goes
    code = app.exec_()
    sys.exit(code or (1 if said else 0))


if __name__ == "__main__":
    main()
