"""Mapping a picture onto the field from labelled lines and marks."""

import unittest

import numpy as np
from support import load_module


def camera():
    """A plausible view from behind the near end zone: field (yards) -> pixels."""
    # Looks down the field; far things are higher in the picture and smaller
    return np.array([[30.0, -4.0, 360.0], [0.0, -3.0, 1000.0], [0.0, 0.011, 0.6]])


def pixel(field_to_image, x, y):
    mapped = field_to_image @ [x, y, 1.0]
    return (mapped[0] / mapped[2], mapped[1] / mapped[2])


class FieldTemplateTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.field_template")
        self.field = self.module.TEMPLATES["usau"]
        self.camera = camera()

    def on_line(self, name, *shares):
        """Pixels at the given shares of the way along a line of the field."""
        start, end = np.array(self.field.lines[name][0]), np.array(self.field.lines[name][1])
        return [pixel(self.camera, *(start + share * (end - start))) for share in shares]

    def test_four_lines_give_the_mapping(self):
        lines = {
            "left_sideline": self.on_line("left_sideline", 0.5, 0.9),
            "right_sideline": self.on_line("right_sideline", 0.4, 0.95),
            "far_back_line": self.on_line("far_back_line", 0.2, 0.8),
            "far_goal_line": self.on_line("far_goal_line", 0.1, 0.7),
        }
        fit = self.module.fit_field(self.field, lines, {})
        self.assertEqual(fit.statements, 8)
        self.assertLess(fit.error, 1e-6)
        # A place nobody labelled: the near brick mark
        mark = pixel(self.camera, *self.field.points["near_brick"])
        np.testing.assert_allclose(
            self.module.to_field(fit.image_to_field, [mark])[0],
            self.field.points["near_brick"],
            atol=1e-5,
        )

    def test_lines_and_marks_together(self):
        lines = {
            "left_sideline": self.on_line("left_sideline", 0.5, 0.9),
            "far_goal_line": self.on_line("far_goal_line", 0.1, 0.7),
        }
        points = {
            name: pixel(self.camera, *self.field.points[name]) for name in ("far_brick", "midfield")
        }
        fit = self.module.fit_field(self.field, lines, points)
        self.assertEqual(fit.statements, 8)
        self.assertLess(fit.error, 1e-6)

    def test_too_little_or_lines_that_say_the_same_give_no_mapping(self):
        sidelines = {
            "left_sideline": self.on_line("left_sideline", 0.5, 0.9),
            "right_sideline": self.on_line("right_sideline", 0.4, 0.95),
        }
        self.assertIsNone(self.module.fit_field(self.field, sidelines, {}))
        # Eight statements, but all about lines across the field: nothing fixes where
        # along those lines a pixel is
        across = {
            name: self.on_line(name, 0.1, 0.3, 0.6, 0.9)
            for name in ("far_back_line", "far_goal_line")
        }
        self.assertIsNone(self.module.fit_field(self.field, across, {}))

    def test_a_badly_placed_element_shows_as_the_worst(self):
        lines = {
            name: self.on_line(name, 0.3, 0.8)
            for name in ("left_sideline", "right_sideline", "far_back_line", "far_goal_line")
        }
        points = {
            name: pixel(self.camera, *self.field.points[name])
            for name in ("far_brick", "midfield", "near_brick")
        }
        # The near goal line labelled 30 pixels too low
        lines["near_goal_line"] = [(u, v + 30) for u, v in self.on_line("near_goal_line", 0.2, 0.8)]
        fit = self.module.fit_field(self.field, lines, points)
        self.assertGreater(fit.error, 1.0)  # Pixels
        # The misfit shows in the near part of the field, where the line and the mark next
        # to it disagree; which of the two is off the fit cannot know
        self.assertIn(fit.worst[0], ("near_goal_line", "near_brick"))
        self.assertGreater(fit.worst[1], 5.0)

    def test_a_line_is_drawn_only_where_it_is_in_front_of_the_camera(self):
        lines = {
            name: self.on_line(name, 0.3, 0.8)
            for name in ("left_sideline", "right_sideline", "far_back_line", "far_goal_line")
        }
        fit = self.module.fit_field(self.field, lines, {})
        # The sideline carried on far behind the camera
        parts = self.module.field_segment_in_image(fit, (0.0, -400.0), (0.0, 110.0))
        self.assertEqual(len(parts), 1)
        first_place = self.module.to_field(fit.image_to_field, [tuple(parts[0][0])])[0]
        self.assertGreater(first_place[1], -150.0)  # Starts in front, not at -400
        np.testing.assert_allclose(parts[0][-1], pixel(self.camera, 0.0, 110.0), atol=1e-4)

    def test_both_rulesets_have_the_same_elements(self):
        usau, wfdf = self.module.TEMPLATES["usau"], self.module.TEMPLATES["wfdf"]
        self.assertEqual(set(usau.lines), set(wfdf.lines))
        self.assertEqual(set(usau.points), set(wfdf.points))


if __name__ == "__main__":
    unittest.main()
