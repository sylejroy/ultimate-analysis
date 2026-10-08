"""Tests for the camera fitted to the field and the estimate made from the field model's masks."""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from ultimate_analysis.processing.field_registration import (
    FieldEstimate,
    FieldFollower,
    estimate_field,
    field_to_canvas,
    implausible,
)
from ultimate_analysis.utils.field_camera import (
    camera_mapping,
    fit_camera,
    focal_of,
    move_camera,
)
from ultimate_analysis.utils.field_template import TEMPLATES

SIZE = (1920, 1080)
FOCAL = 1500.0
POSITION = (18.0, 30.0, 9.0)


def looking_down_the_field(pitch_degrees: float = 22.0) -> np.ndarray:
    """Rotation vector of a camera that looks along the field, tilted down."""
    pitch = np.radians(pitch_degrees)
    turn = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -np.sin(pitch), -np.cos(pitch)],
            [0.0, np.cos(pitch), -np.sin(pitch)],
        ]
    )
    return cv2.Rodrigues(turn)[0].ravel()


def pixel_of(mapping: np.ndarray, place) -> np.ndarray:
    mapped = mapping @ [place[0], place[1], 1.0]
    return mapped[:2] / mapped[2]


class FieldCameraTest(unittest.TestCase):
    def setUp(self):
        self.template = TEMPLATES["usau"]
        self.mapping = camera_mapping(FOCAL, looking_down_the_field(), POSITION, SIZE)
        self.far_corners = {
            name: tuple(pixel_of(self.mapping, self.template.points[name]))
            for name in ("far_back_left", "far_back_right", "far_goal_left", "far_goal_right")
        }

    def test_mapping_gives_its_focal_length(self):
        self.assertAlmostEqual(focal_of(self.mapping, SIZE), FOCAL, delta=1e-3)

    def test_camera_is_found_from_the_far_corners(self):
        fit = fit_camera(self.template, {}, self.far_corners, SIZE)
        self.assertIsNotNone(fit)
        self.assertLess(fit.error, 0.01)
        self.assertAlmostEqual(fit.focal, FOCAL, delta=2.0)
        np.testing.assert_allclose(fit.position, POSITION, atol=0.05)

    def test_known_focal_length_holds_the_near_end_with_imprecise_corners(self):
        rng = np.random.default_rng(0)
        near_goal_left = self.template.points["near_goal_left"]
        truth = pixel_of(self.mapping, near_goal_left)
        free_errors, known_errors = [], []
        for _ in range(10):
            corners = {
                name: (x + rng.normal(0, 2.0), y + rng.normal(0, 2.0))
                for name, (x, y) in self.far_corners.items()
            }
            free = fit_camera(self.template, {}, corners, SIZE)
            known = fit_camera(self.template, {}, corners, SIZE, focal=FOCAL)
            free_errors.append(
                np.linalg.norm(pixel_of(free.field_to_image, near_goal_left) - truth)
            )
            known_errors.append(
                np.linalg.norm(pixel_of(known.field_to_image, near_goal_left) - truth)
            )
        self.assertLess(np.median(known_errors), np.median(free_errors))

    def test_moved_camera_puts_the_marks_on_their_pixels(self):
        # One mark dragged takes the field along; a second one stays where the first was put
        first = tuple(np.add(self.far_corners["far_back_left"], (30.0, 12.0)))
        moved = move_camera(self.template, self.mapping, {"far_back_left": first}, SIZE, FOCAL)
        np.testing.assert_allclose(
            pixel_of(moved, self.template.points["far_back_left"]), first, atol=0.5
        )
        other = pixel_of(moved, self.template.points["far_back_right"])
        shift = other - self.far_corners["far_back_right"]
        self.assertGreater(np.linalg.norm(shift), 10.0)  # The rest followed

        second = tuple(other + (-20.0, 4.0))
        both = {"far_back_left": first, "far_back_right": second}
        moved = move_camera(self.template, moved, both, SIZE, FOCAL)
        for name, pixel in both.items():
            np.testing.assert_allclose(pixel_of(moved, self.template.points[name]), pixel, atol=0.5)

    def test_too_few_elements_give_no_camera(self):
        corners = dict(list(self.far_corners.items())[:2])
        self.assertIsNone(fit_camera(self.template, {}, corners, SIZE))


def masks_as_the_field_model_gives_them(template, mapping, resolution=640):
    """The central field and the end zones as seen by a camera, as segmentation results."""
    grid_x, grid_y = np.meshgrid(
        (np.arange(resolution) + 0.5) * SIZE[0] / resolution,
        (np.arange(resolution) + 0.5) * SIZE[1] / resolution,
    )
    pixels = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    places = pixels @ np.linalg.inv(mapping).T
    in_front = (places[..., :2] / places[..., 2:3]) @ mapping[2, :2] + mapping[2, 2] > 0
    across, along = places[..., 0] / places[..., 2], places[..., 1] / places[..., 2]
    on_field = in_front & (across >= 0) & (across <= template.width)
    central = (
        on_field & (along >= template.end_zone) & (along <= template.length - template.end_zone)
    )
    far_zone = on_field & (along > template.length - template.end_zone) & (along <= template.length)
    near_zone = on_field & (along >= 0) & (along < template.end_zone)
    masks, classes = [central, far_zone], [0, 1]
    if near_zone.sum() > 500:
        masks.append(near_zone)
        classes.append(1)
    return [
        SimpleNamespace(
            masks=SimpleNamespace(data=np.array(masks, dtype=np.float32)),
            boxes=SimpleNamespace(
                cls=np.array(classes, dtype=np.float32),
                conf=np.linspace(0.95, 0.85, len(masks)).astype(np.float32),
            ),
        )
    ]


class FieldEstimateTest(unittest.TestCase):
    def setUp(self):
        self.template = TEMPLATES["usau"]
        self.mapping = camera_mapping(FOCAL, looking_down_the_field(), POSITION, SIZE)
        self.results = masks_as_the_field_model_gives_them(self.template, self.mapping)
        self.shape = (SIZE[1], SIZE[0])

    def far_corner_error(self, field_to_image: np.ndarray) -> float:
        return max(
            np.linalg.norm(
                pixel_of(field_to_image, self.template.points[name])
                - pixel_of(self.mapping, self.template.points[name])
            )
            for name in ("far_back_left", "far_back_right", "far_goal_left", "far_goal_right")
        )

    def test_field_is_found_from_the_masks(self):
        estimate = estimate_field(self.results, self.shape, self.template, focal=FOCAL)
        self.assertIsNotNone(estimate)
        self.assertIn("far_goal_line", estimate.lines)
        self.assertLess(self.far_corner_error(estimate.field_to_image), 8.0)
        self.assertAlmostEqual(estimate.position[2], POSITION[2], delta=1.5)

    def test_no_field_without_masks(self):
        self.assertIsNone(estimate_field([], self.shape, self.template))

    def test_follower_moves_with_the_camera(self):
        follower = FieldFollower(self.template)
        follower.update(self.results, self.shape)
        self.assertIsNotNone(follower.image_to_field)
        pixel = np.array([900.0, 700.0, 1.0])
        before = follower.image_to_field @ pixel
        shift = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]])
        follower.move(shift)
        after = follower.image_to_field @ (shift @ pixel)
        np.testing.assert_allclose(after[:2] / after[2], before[:2] / before[2], atol=1e-6)
        follower.reset()
        self.assertIsNone(follower.image_to_field)

    def as_estimate(self, mapping: np.ndarray) -> FieldEstimate:
        return FieldEstimate(mapping, np.linalg.inv(mapping), {}, 0.0, FOCAL, np.array(POSITION))

    def test_an_estimate_that_puts_the_players_off_the_field_cannot_be_right(self):
        places = [(x, y) for x in (8.0, 20.0, 32.0) for y in (60.0, 75.0, 95.0)]
        feet = np.array([pixel_of(self.mapping, place) for place in places])
        mask = np.asarray(self.results[0].masks.data).max(axis=0)
        mask = cv2.resize(mask, SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        self.assertEqual(implausible(self.as_estimate(self.mapping), self.template, mask, feet), "")

        # The same view taken for a field 45 yards to the side
        aside = self.mapping @ np.array([[1.0, 0.0, 45.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertIn("players", implausible(self.as_estimate(aside), self.template, None, feet))
        self.assertIn("field model", implausible(self.as_estimate(aside), self.template, mask))

    def test_one_estimate_far_from_the_followed_field_is_not_taken_two_are(self):
        follower = FieldFollower(self.template)
        follower.new_video(FOCAL)
        follower.update(self.results, self.shape)
        first = follower.image_to_field.copy()

        elsewhere = camera_mapping(FOCAL, looking_down_the_field(), (18.0, 5.0, 9.0), SIZE)
        other = masks_as_the_field_model_gives_them(self.template, elsewhere)
        follower.update(other, self.shape)
        np.testing.assert_allclose(follower.image_to_field, first)
        self.assertIn("waiting", follower.left_out)

        follower.update(other, self.shape)
        self.assertGreater(np.abs(follower.image_to_field - first).max(), 1e-6)
        self.assertEqual(follower.left_out, "")

        # A lone outlier between two estimates that agree with the followed field is forgotten
        follower.update(self.results, self.shape)
        self.assertIn("waiting", follower.left_out)

    def test_someone_standing_beside_the_field_is_told_from_one_who_steps_out(self):
        from types import SimpleNamespace

        from ultimate_analysis.processing.field_registration import OffFieldWatcher

        watcher = OffFieldWatcher(self.template)
        mapping = np.eye(3)  # A pixel is a field unit: feet at (x, y) stand at (x, y)

        def player(track_id, x, y):
            return SimpleNamespace(
                track_id=track_id, class_name="player", to_ltrb=lambda: [x - 1, y - 4, x + 1, y]
            )

        middle = self.template.width / 2
        for frame in range(60):
            # 1 on the field, 2 five units beside it, 3 on the field and out for a moment,
            # 4 a unit over the line: within the margin
            steps_out = -5.0 if 30 <= frame < 40 else 5.0
            off = watcher.update(
                mapping,
                [
                    player(1, middle, 30),
                    player(2, -5, 30),
                    player(3, steps_out, 50),
                    player(4, -1, 60),
                ],
            )
        self.assertEqual(off, {2})
        # Without knowing where the field is, what was learned holds
        self.assertEqual(watcher.update(None, [player(2, middle, 30)]), {2})
        # Coming on to play, a track is on the field again after a while
        for frame in range(300):
            off = watcher.update(mapping, [player(2, middle, 30)])
        self.assertEqual(off, set())

    def test_canvas_shows_the_whole_field_with_the_far_end_on_top(self):
        on_canvas = field_to_canvas(self.template, (400, 1200))
        near_left = pixel_of(on_canvas, (0.0, 0.0))
        far_right = pixel_of(on_canvas, (self.template.width, self.template.length))
        self.assertLess(far_right[1], near_left[1])
        self.assertLess(near_left[0], far_right[0])
        for corner in (near_left, far_right):
            self.assertTrue(0 <= corner[0] <= 400 and 0 <= corner[1] <= 1200)


if __name__ == "__main__":
    unittest.main()
