"""Phone labelling: what each answer stores, and that the server asks for the key."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from support import load_module

VIDEO = "games/final.mp4"


class PhoneLabellingTests(unittest.TestCase):
    def setUp(self):
        self.session_module = load_module("web.label_session")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dataset = Path(folder.name) / "discs"
        self.suggestions = [((100.0, 200.0, 118.0, 214.0), 0.3)]
        self.session = self.session_module.LabelSession(
            self.dataset,
            {VIDEO: 5000},
            read_frame=lambda video, index: np.full((1080, 1920, 3), index % 255, dtype=np.uint8),
            find_discs=lambda frame: list(self.suggestions),
            seed=1,
        )

    def label_lines(self, task):
        name = f"final_frame_{task['frame']:06d}"
        return (self.dataset / "labels" / f"{name}.txt").read_text().split("\n")[:-1]

    def test_confirmed_moved_and_missing_discs_are_stored_and_can_be_undone(self):
        session = self.session

        # Suggestion confirmed
        first = session.next_task()
        self.assertEqual(first["box"], [100.0, 200.0, 118.0, 214.0])
        self.assertTrue(session.save(first["task"], first["box"]))
        self.assertEqual(len(self.label_lines(first)), 1)

        # The disc is somewhere else
        second = session.next_task()
        session.save(second["task"], [300, 400, 318, 414])
        (line,) = self.label_lines(second)
        np.testing.assert_allclose(
            [float(value) for value in line.split()],
            [0, 309 / 1920, 407 / 1080, 18 / 1920, 14 / 1080],
            atol=1e-5,
        )

        # No disc to be seen: the frame is stored without a box
        third = session.next_task()
        session.save(third["task"], None)
        self.assertEqual(self.label_lines(third), [])
        self.assertEqual(session.next_task()["labelled"], 3)

        # A disc-only dataset, and an answer cannot be given twice
        names = yaml.safe_load((self.dataset / "data.yaml").read_text())["names"]
        self.assertEqual(names, ["disc"])
        self.assertFalse(session.save(third["task"], None))

        self.assertEqual(session.undo(), f"final_frame_{third['frame']:06d}")
        self.assertEqual(session.labelled_count(), 2)

    def test_skipped_and_labelled_frames_are_not_offered_again(self):
        session = self.session_module.LabelSession(
            self.dataset,
            {VIDEO: 6},
            read_frame=lambda video, index: np.zeros((1080, 1920, 3), dtype=np.uint8),
            find_discs=lambda frame: [],
            seed=3,
        )
        seen = set()
        for step in range(6):
            task = session.next_task()
            self.assertIsNone(task["box"])
            self.assertNotIn(task["frame"], seen)
            seen.add(task["frame"])
            if step % 2:
                session.skip(task["task"])
            else:
                session.save(task["task"], None)
        self.assertIsNone(session.next_task())

    def test_suggestions_the_model_is_sure_of_are_mostly_passed_over(self):
        self.suggestions = [((100.0, 200.0, 118.0, 214.0), 0.9)]
        reads = []
        session = self.session_module.LabelSession(
            self.dataset,
            {VIDEO: 100000},
            read_frame=lambda video, index: reads.append(index) or np.zeros((8, 8, 3), np.uint8),
            find_discs=lambda frame: list(self.suggestions),
            seed=5,
        )
        offered = sum(session.next_task() is not None for _ in range(20))
        self.assertEqual(offered, 20)
        self.assertGreater(len(reads), 40)  # Most frames were read and passed over

    def test_the_box_is_fitted_to_the_disc_at_the_tapped_spot(self):
        import cv2

        grass = np.full((1080, 1920, 3), (60, 140, 70), dtype=np.uint8)
        # A disc seen from the side (a thin line), one from above (round), and a shirt
        cv2.ellipse(grass, (500, 300), (11, 2), 0, 0, 360, (235, 235, 235), -1)
        cv2.circle(grass, (900, 600), 8, (235, 235, 235), -1)
        cv2.rectangle(grass, (1400, 200), (1500, 400), (240, 240, 240), -1)
        session = self.session_module.LabelSession(
            self.dataset, {VIDEO: 10}, lambda video, index: grass, lambda frame: [], seed=1
        )
        task = session.next_task()["task"]

        side_on = session.fit_box(task, 503, 302)  # Tapped slightly off its middle
        np.testing.assert_allclose(side_on, [488, 297, 513, 304], atol=2)
        from_above = session.fit_box(task, 900, 600)
        np.testing.assert_allclose(from_above, [891, 591, 910, 610], atol=2)

        # Nothing disc-like at the tap, or something far too big: a box of typical size
        for x, y in ((200, 900), (1450, 300)):
            box = session.fit_box(task, x, y)
            np.testing.assert_allclose(box, [x - 9, y - 7, x + 9, y + 7])
        self.assertIsNone(session.fit_box("unknown", 10, 10))

    def test_pictures_are_cut_inside_the_frame(self):
        session = self.session
        task = session.next_task()
        self.assertEqual(session.view(1920, 1080, -50, 1000, 400, 400), (0, 680, 400, 400))
        self.assertEqual(session.view(1920, 1080, 0, 0, 5000, 5000), (0, 0, 1920, 1080))

        picture = session.picture(task["task"], 1800, 900, 220, 220, 480)
        self.assertEqual(picture[:2], b"\xff\xd8")  # A JPEG
        self.assertIsNone(session.picture("unknown", 0, 0, 10, 10, 100))

    def test_server_needs_the_key_and_carries_a_labelling_round(self):
        server_module = load_module("web.label_server")
        server = server_module.create_server(self.session, "secret", "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def get(path):
            with urllib.request.urlopen(base + path) as response:
                return response.read()

        def post(path, body):
            request = urllib.request.Request(base + path, data=json.dumps(body).encode())
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read())

        for path in ("/", "/api/next", "/api/next?key=wrong"):
            with self.assertRaises(urllib.error.HTTPError) as refused:
                get(path)
            self.assertEqual(refused.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as refused:
            post("/api/save", {"task": "0", "box": None})
        self.assertEqual(refused.exception.code, 403)

        self.assertIn(b"Is this the disc?", get("/?key=secret"))
        task = json.loads(get("/api/next?key=secret"))
        picture = get(f"/api/picture?task={task['task']}&x=0&y=0&w=1920&h=1080&out=640&key=secret")
        self.assertEqual(picture[:2], b"\xff\xd8")
        self.assertEqual(
            post("/api/save?key=secret", {"task": task["task"], "box": task["box"]}), {"ok": True}
        )
        self.assertEqual(self.session.labelled_count(), 1)
        self.assertEqual(
            post("/api/undo?key=secret", {})["undone"], f"final_frame_{task['frame']:06d}"
        )

        with self.assertRaises(urllib.error.HTTPError) as bad:
            post("/api/save?key=secret", {"task": "1", "box": [1, 2]})
        self.assertEqual(bad.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
