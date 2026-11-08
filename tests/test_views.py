"""The views of a frame, drawn for real from a Scene: nothing mocked in the drawing."""

import unittest
from types import SimpleNamespace

import numpy as np
from support import load_module


class ViewsTest(unittest.TestCase):
    def setUp(self):
        self.views = load_module("pipeline.views")
        tracking = load_module("processing.tracking")
        template = load_module("utils.field_template").TEMPLATES["usau"]
        self.records = {}
        player = tracking.Track(7, [900.0, 400.0, 940.0, 490.0], 1, 0.9, "player")
        player.team_colour = (40, 40, 200)
        disc = tracking.Track(100001, [1000.0, 300.0, 1010.0, 310.0], 0, 0.8, "disc")
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mask[200:1000, 100:1800] = 1
        # Pixels to field units: 40 yards across the frame, 110 along it
        to_field = np.diag([40.0 / 1920, 110.0 / 1080, 1.0])
        self.scene = self.views.Scene(
            template=template,
            frame_size=(1920, 1080),
            fps=30.0,
            tracks=[player, disc],
            trails={7: np.array([[880, 480], [900, 485], [920, 490]], dtype=np.int32)},
            numbers={7: ("17", None)},
            read_numbers=True,
            holder_id=7,
            possession_colour=(40, 40, 200),
            field_results=[object()],
            field_mask=mask,
            image_to_field=to_field,
            places={7: (19.0, 45.0)},
            disc_path=[np.array([[19.0, 45.0], [20.0, 50.0]])],
        )
        self.options = SimpleNamespace(
            detection=True, tracking=True, player_id=True, field_segmentation=True
        )

    def record(self, stage, start):
        self.records[stage] = start
        return 0.0

    def test_the_camera_view_is_drawn_on_the_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        drawn = self.views.draw_camera_view(frame, self.scene, self.options, self.record)
        self.assertIs(drawn, frame)
        self.assertTrue(frame[200, 100:1800].any())  # The outline of the field
        self.assertTrue(frame[400:490, 900:940].any())  # The player
        self.assertIn("Visualization", self.records)

    def test_the_field_is_drawn_from_above(self):
        view, message = self.views.draw_top_down_view(self.scene, self.options, self.record)
        self.assertEqual(message, "")
        self.assertEqual(view.ndim, 3)
        self.assertTrue(view.any())
        self.assertIn("Top-down view", self.records)

    def test_where_the_next_throw_goes_is_shaded_over_the_field(self):
        plain, _ = self.views.draw_top_down_view(self.scene, self.options, self.record)
        chances = np.zeros((55, 20))
        chances[52, 2] = 1.0  # Near the far back line, on the left
        self.scene.field_map = chances
        shaded, _ = self.views.draw_top_down_view(self.scene, self.options, self.record)
        rows, columns = np.nonzero((shaded != plain).any(axis=2))
        self.assertTrue(len(rows))
        self.assertLess(rows.mean(), 0.15 * shaded.shape[0])  # The far end is at the top
        self.assertLess(columns.mean(), 0.5 * shaded.shape[1])

    def test_where_the_next_throw_goes_is_shaded_on_the_camera_view_too(self):
        def drawn():
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            return self.views.draw_camera_view(frame, self.scene, self.options, self.record)

        plain = drawn()
        chances = np.zeros((55, 20))
        chances[52, 2] = 1.0  # Around (5, 105) on the field: pixel (240, 1030) here
        self.scene.field_map = chances
        rows, columns = np.nonzero((drawn() != plain).any(axis=2))
        self.assertTrue(len(rows))
        self.assertAlmostEqual(rows.mean(), 1030, delta=40)
        self.assertAlmostEqual(columns.mean(), 240, delta=60)

    def test_without_the_field_there_is_no_top_down_view(self):
        self.scene.image_to_field = None
        view, message = self.views.draw_top_down_view(self.scene, self.options, self.record)
        self.assertIsNone(view)
        self.assertIn("not been found", message)

    def test_a_close_up_is_said_and_not_drawn_over(self):
        self.scene.wide = False
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.views.draw_camera_view(frame, self.scene, self.options, self.record)
        self.assertFalse(frame[400:490, 900:940].any())
        self.assertEqual(
            self.views.draw_top_down_view(self.scene, self.options, self.record)[0], None
        )


if __name__ == "__main__":
    unittest.main()
