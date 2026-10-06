"""Field segmentation caching, line fitting, and field drawing."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.field_segmentation")
        self.module.reset_segmentation_cache()

    def test_empty_results_are_cached_but_rewind_and_resize_invalidate(self):
        model = Mock()
        model.predict.return_value = []
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            patch.multiple(
                self.module, _field_model=model, _model_imgsz=8, ULTRALYTICS_AVAILABLE=True
            ),
            patch.object(self.module, "get_setting", side_effect=lambda key, default=None: default),
        ):
            for index in (10, 11, 12, 13, 14):
                self.module.run_field_segmentation(frame, index)
            self.assertEqual(model.predict.call_count, 1)
            self.module.run_field_segmentation(frame, 2)
            self.assertEqual(model.predict.call_count, 2)
            self.module.run_field_segmentation(np.zeros((6, 8, 3), dtype=np.uint8), 3)
            self.assertEqual(model.predict.call_count, 3)
            self.module.reset_segmentation_cache()
            self.module.run_field_segmentation(frame, 4)
            self.assertEqual(model.predict.call_count, 4)


class FieldGeometryTests(unittest.TestCase):
    def test_main_outline_survives_alternating_warped_masks(self):
        module = load_module("rendering.field")
        module._mask_outline_cache.clear()
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        original_find = module.cv2.findContours
        with patch.object(module.cv2, "findContours", wraps=original_find) as find:
            first = module._get_mask_outline(mask)
            for _ in range(5):
                module._get_mask_outline(mask.copy())  # A new warped mask each frame.
                self.assertIs(module._get_mask_outline(mask), first)
        self.assertEqual(find.call_count, 6)
        self.assertEqual(len(module._mask_outline_cache), 2)

    def test_numpy_ransac_fits_vertical_line_and_rejects_outliers(self):
        module = load_module("processing.field_analysis")
        line = np.column_stack([np.full(50, 100.0), np.linspace(0, 490, 50)])
        outliers = np.array([[300.0, 10.0], [400.0, 250.0], [20.0, 480.0]])
        points = np.vstack([line, outliers]).astype(np.float32)
        with patch.object(module, "get_setting", side_effect=lambda key, default=None: default):
            (start, end), rejected, inliers, confidence = module._fit_line_ransac_with_outliers(
                points, distance_threshold=5.0, min_samples=2, max_trials=50
            )
        self.assertEqual(len(inliers), 50)
        self.assertEqual(len(rejected), 3)
        self.assertAlmostEqual(confidence, 50 / 53)
        np.testing.assert_allclose([start[0], end[0]], [100.0, 100.0], atol=1e-3)
        np.testing.assert_allclose(sorted([start[1], end[1]]), [0.0, 490.0], atol=1e-3)

    def test_drawing_with_a_cached_fit_does_not_refit_or_copy(self):
        module = load_module("rendering.field")

        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        lines = [np.array([[10.0, 10.0], [29.0, 10.0]])]
        fit = (lines, [np.empty((0, 2))], [lines[0]], np.empty((0, 2)), {}, {"line_0": 1})
        settings = {"models.segmentation.contour.ransac.enabled": True}
        with (
            patch.object(module, "get_setting", side_effect=lambda k, d=None: settings.get(k, d)),
            patch.object(module, "fit_field_lines_ransac") as refit,
        ):
            result, _, all_lines = module.draw_unified_field_mask(
                frame, mask, ransac_fit=fit, in_place=True
            )
        refit.assert_not_called()
        self.assertIs(result, frame)
        self.assertEqual(all_lines, {"line_0": 1})
        self.assertTrue(frame.any())


if __name__ == "__main__":
    unittest.main()
