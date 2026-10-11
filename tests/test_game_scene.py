"""The scene in three dimensions: where things are put, and how an eye sees them."""

import unittest
from types import SimpleNamespace

import numpy as np
from support import load_module

SIZE = (1280, 720)


def result_of(to_field, tracks, **given):
    values = {
        "frame_index": 10,
        "image_to_field": to_field,
        "wide_shot": True,
        "tracks": tracks,
        "player_ids": {},
        "numbers_by_looks": {},
        "holder_id": None,
        "disc_state": "air",
        "disc_place": None,
        "detections": [],
        "main_view": None,
    }
    return SimpleNamespace(**{**values, **given})


def player(track_id, feet, colour=None):
    x, y = feet
    return SimpleNamespace(
        track_id=track_id, class_name="player", bbox=[x - 1, y - 4, x + 1, y], team_colour=colour
    )


class SceneOfTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("rendering.game_scene")
        self.template = load_module("utils.field_template").TEMPLATES["usau"]
        # A picture in which a pixel is half a field unit
        self.to_field = np.diag([0.5, 0.5, 1.0])

    def test_players_stand_where_the_lower_edge_of_their_box_is(self):
        result = result_of(
            self.to_field,
            [player(1, (40, 100), (0, 0, 255)), player(2, (20, 60))],
            player_ids={1: ("12", None)},
            numbers_by_looks={2: "7"},
        )
        scene = self.module.scene_of(result, self.template)
        first, second = scene.players
        self.assertEqual(
            (first.place, first.number, first.colour), ((20.0, 50.0), "12", (0, 0, 255))
        )
        self.assertEqual((second.place, second.number, second.colour), ((10.0, 30.0), "7", None))

    def test_who_stands_far_from_the_field_is_left_out(self):
        result = result_of(self.to_field, [player(1, (40, 100)), player(2, (400, 100))])
        scene = self.module.scene_of(result, self.template)
        self.assertEqual([p.player_id for p in scene.players], [1])

    def test_a_held_disc_is_with_its_holder_and_a_flying_one_over_its_place(self):
        tracks = [player(1, (40, 100))]
        held = result_of(self.to_field, tracks, holder_id=1, disc_state="held")
        scene = self.module.scene_of(held, self.template)
        self.assertTrue(scene.players[0].holding)
        self.assertEqual(scene.disc[:2], (20.0, 50.0))
        self.assertGreater(scene.disc[2], 0.0)

        flying = result_of(self.to_field, tracks, disc_place=(15.0, 70.0))
        disc = self.module.scene_of(flying, self.template).disc
        self.assertEqual(disc[:2], (15.0, 70.0))
        self.assertGreater(disc[2], 0.0)

    def test_a_disc_that_is_only_seen_lies_where_it_is_seen(self):
        seen = result_of(
            self.to_field,
            [],
            detections=[
                {"class_name": "disc", "bbox": [38, 78, 42, 82], "confidence": 0.8},
                {"class_name": "disc", "bbox": [0, 0, 4, 4], "confidence": 0.3},
            ],
        )
        self.assertEqual(self.module.scene_of(seen, self.template).disc, (20.0, 40.0, 0.0))

    def test_observers_are_in_the_scene_in_orange_and_are_no_players(self):
        result = result_of(
            self.to_field,
            [player(1, (40, 100), (200, 30, 30))],
            observers={9: (5.0, 60.0), 10: (400.0, 60.0)},
        )
        scene = self.module.scene_of(result, self.template)
        watching = [p for p in scene.players if p.observer]
        self.assertEqual([(p.player_id, p.place) for p in watching], [(9, (5.0, 60.0))])
        self.assertEqual(watching[0].colour, self.module.OBSERVER_COLOUR)
        self.assertFalse(scene.players[0].observer)

    def test_what_the_camera_shows_is_in_the_scene(self):
        outline = np.array([[0.0, 10.0], [40.0, 10.0], [60.0, 90.0], [-20.0, 90.0]])
        scene = self.module.scene_of(
            result_of(self.to_field, [], seen_by_camera=outline), self.template
        )
        np.testing.assert_allclose(scene.seen, outline)
        self.assertIsNone(self.module.scene_of(result_of(self.to_field, []), self.template).seen)

    def test_no_scene_without_the_field_or_on_a_close_up(self):
        self.assertIsNone(self.module.scene_of(result_of(None, []), self.template))
        close_up = result_of(self.to_field, [], wide_shot=False)
        self.assertIsNone(self.module.scene_of(close_up, self.template))

    def test_the_camera_is_found_where_it_films_from(self):
        camera = load_module("utils.field_camera")
        position = np.array([20.0, -25.0, 12.0])
        # Looking along the field and 25 degrees down
        tilt = np.radians(90.0 + 25.0)
        to_image = camera.camera_mapping(1000.0, (tilt, 0.0, 0.0), position, SIZE)
        result = result_of(
            np.linalg.inv(to_image), [], main_view=np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        )
        scene = self.module.scene_of(result, self.template)
        np.testing.assert_allclose(scene.camera, position, atol=0.5)
        self.assertAlmostEqual(scene.looks_at[0], 20.0, places=3)
        self.assertGreater(scene.looks_at[1], position[1])


class ViewPointTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("rendering.game_scene")
        self.view = self.module.ViewPoint((20.0, 55.0, 0.0), 0.0, 30.0, 80.0)

    def test_the_spot_looked_at_is_in_the_middle_of_the_picture(self):
        seen = self.view.seen([self.view.target])
        np.testing.assert_allclose(seen[0], [0.0, 0.0, 80.0], atol=1e-9)
        np.testing.assert_allclose(self.view.on_picture(seen, SIZE)[0], [640.0, 360.0], atol=1e-9)

    def test_up_is_up_and_right_is_right(self):
        # The eye is beside the field, on the side of large x, looking back across it
        above, along = self.view.on_picture(
            self.view.seen([(20.0, 55.0, 5.0), (20.0, 65.0, 0.0)]), SIZE
        )
        self.assertLess(above[1], 360.0)
        self.assertGreater(along[0], 640.0)

    def test_an_eye_placed_somewhere_is_there(self):
        view = self.module.ViewPoint.from_eye((20.0, -25.0, 12.0), (20.0, 40.0, 0.0))
        np.testing.assert_allclose(view.eye, [20.0, -25.0, 12.0], atol=1e-9)

    def test_what_lies_behind_the_eye_is_cut_off(self):
        # A strip of ground from in front of the eye to far behind it
        eye = self.view.eye
        strip = [
            (eye[0] - 30, 50, 0),
            (eye[0] - 30, 60, 0),
            (eye[0] + 50, 60, 0),
            (eye[0] + 50, 50, 0),
        ]
        left = self.module.clip_ahead(self.view.seen(strip))
        self.assertGreaterEqual(len(left), 3)
        self.assertTrue(np.all(left[:, 2] >= self.module.NEAR - 1e-9))
        behind = [(eye[0] + 100, 50, 0), (eye[0] + 100, 60, 0), (eye[0] + 150, 60, 0)]
        self.assertEqual(len(self.module.clip_ahead(self.view.seen(behind))), 0)

    def test_a_polygon_is_cut_to_a_rectangle(self):
        outline = np.array([[-10.0, 5.0], [50.0, 5.0], [50.0, 300.0], [-10.0, 300.0]])
        inside = self.module.within_rectangle(outline, (0.0, 0.0), (40.0, 110.0))
        self.assertEqual(
            sorted(map(tuple, inside.round(6))),
            [(0.0, 5.0), (0.0, 110.0), (40.0, 5.0), (40.0, 110.0)],
        )
        beside = outline + [100.0, 0.0]
        self.assertEqual(len(self.module.within_rectangle(beside, (0.0, 0.0), (40.0, 110.0))), 0)

    def test_a_line_keeps_its_ends_where_they_are_ahead(self):
        line = self.view.seen([(20, 50, 0), (20, 60, 0)])
        np.testing.assert_allclose(self.module.clip_ahead(line, closed=False), line)


if __name__ == "__main__":
    unittest.main()
