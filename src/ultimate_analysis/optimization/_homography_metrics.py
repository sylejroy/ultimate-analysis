"""Homography evaluation metrics used by the genetic optimizer.

Split from `homography_optimizer.py` to keep files within size limits and improve clarity.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting


def evaluate_line_alignment(
    h_matrix: np.ndarray,
    ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
    confidences: List[float],
    output_size: Tuple[int, int],
) -> float:
    if not ransac_lines:
        return 0.0

    output_width, output_height = output_size
    vertical_tolerance = get_setting("optimization.ga_line_alignment.vertical_tolerance", 5.0)
    horizontal_tolerance = get_setting("optimization.ga_line_alignment.horizontal_tolerance", 5.0)
    min_line_length = get_setting("optimization.ga_line_alignment.min_line_length", 10)

    total_score = 0.0
    total_weight = 0.0

    for i, (start_point, end_point) in enumerate(ransac_lines):
        if i >= len(confidences):
            continue
        confidence = confidences[i]

        start_homo = np.array([start_point[0], start_point[1], 1.0])
        end_homo = np.array([end_point[0], end_point[1], 1.0])
        start_transformed = h_matrix @ start_homo
        end_transformed = h_matrix @ end_homo

        if start_transformed[2] != 0 and end_transformed[2] != 0:
            start_2d = start_transformed[:2] / start_transformed[2]
            end_2d = end_transformed[:2] / end_transformed[2]
            dx = end_2d[0] - start_2d[0]
            dy = end_2d[1] - start_2d[1]
            length = np.sqrt(dx * dx + dy * dy)
            if length < min_line_length:
                continue
            angle = np.degrees(np.arctan2(dy, dx))

            if (
                0 <= start_2d[0] < output_width
                and 0 <= start_2d[1] < output_height
                and 0 <= end_2d[0] < output_width
                and 0 <= end_2d[1] < output_height
            ):
                vertical_deviation = min(abs(90 - abs(angle)), abs(270 - abs(angle)))
                horizontal_deviation = min(abs(angle), abs(180 - abs(angle)))

                if vertical_deviation <= horizontal_deviation:
                    alignment_score = max(0.0, 1.0 - vertical_deviation / vertical_tolerance)
                else:
                    alignment_score = max(0.0, 1.0 - horizontal_deviation / horizontal_tolerance)

                weight = confidence * (length / 100.0)
                total_score += alignment_score * weight
                total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def evaluate_field_coverage(warped_frame: np.ndarray) -> float:
    black_threshold = get_setting("optimization.ga_field_coverage.black_pixel_threshold", 10)
    optimal_coverage = get_setting("optimization.ga_field_coverage.optimal_coverage_ratio", 0.75)
    gray_warped = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    non_black_pixels = np.count_nonzero(gray_warped > black_threshold)
    total_pixels = gray_warped.shape[0] * gray_warped.shape[1]
    current_coverage = non_black_pixels / total_pixels

    if current_coverage <= optimal_coverage:
        coverage_score = current_coverage / optimal_coverage
    else:
        excess = current_coverage - optimal_coverage
        max_excess = 1.0 - optimal_coverage
        penalty = excess / max_excess if max_excess > 0 else 0
        coverage_score = 1.0 - (penalty * 0.5)
    return max(0.0, min(1.0, coverage_score))


def evaluate_line_visibility(
    h_matrix: np.ndarray,
    ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
    output_size: Tuple[int, int],
) -> float:
    if not ransac_lines:
        return 0.0
    output_width, output_height = output_size
    visible_lines = 0
    for start_point, end_point in ransac_lines:
        start_homo = np.array([start_point[0], start_point[1], 1.0])
        end_homo = np.array([end_point[0], end_point[1], 1.0])
        start_transformed = h_matrix @ start_homo
        end_transformed = h_matrix @ end_homo
        if start_transformed[2] != 0 and end_transformed[2] != 0:
            start_2d = start_transformed[:2] / start_transformed[2]
            end_2d = end_transformed[:2] / end_transformed[2]
            if (0 <= start_2d[0] < output_width and 0 <= start_2d[1] < output_height) or (
                0 <= end_2d[0] < output_width and 0 <= end_2d[1] < output_height
            ):
                visible_lines += 1
    return visible_lines / len(ransac_lines)


def evaluate_field_proportions(
    h_matrix: np.ndarray,
    ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
    output_size: Tuple[int, int],
) -> float:
    expected_ratio = get_setting("optimization.ga_field_proportion.expected_ratio", 2.7)
    ratio_tolerance = get_setting("optimization.ga_field_proportion.ratio_tolerance", 0.5)
    output_width, output_height = output_size
    actual_ratio = output_height / output_width
    ratio_deviation = abs(actual_ratio - expected_ratio)
    proportion_score = max(0.0, 1.0 - ratio_deviation / ratio_tolerance)
    return proportion_score


def evaluate_perspective_distortion(h_matrix: np.ndarray) -> float:
    try:
        det = np.linalg.det(h_matrix)
        if abs(det) < 1e-10:
            return 0.0
        h20, h21 = h_matrix[2, 0], h_matrix[2, 1]
        perspective_magnitude = np.sqrt(h20 * h20 + h21 * h21)
        max_perspective = get_setting("homography.slider_range_perspective", [-0.2, 0.2])[1]
        if perspective_magnitude > abs(max_perspective):
            return 0.0
        scale_x = np.sqrt(h_matrix[0, 0] ** 2 + h_matrix[0, 1] ** 2)
        scale_y = np.sqrt(h_matrix[1, 0] ** 2 + h_matrix[1, 1] ** 2)
        if scale_x < 0.1 or scale_x > 10.0 or scale_y < 0.1 or scale_y > 10.0:
            return 0.5
        return 1.0
    except Exception:
        return 0.0


def evaluate_line_orientation_balance(
    h_matrix: np.ndarray,
    ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
    output_size: Tuple[int, int],
) -> float:
    if not ransac_lines:
        return 0.0
    output_width, output_height = output_size
    vertical_tolerance = get_setting("optimization.ga_line_orientation.vertical_tolerance", 15.0)
    horizontal_tolerance = get_setting(
        "optimization.ga_line_orientation.horizontal_tolerance", 15.0
    )
    min_line_length = get_setting("optimization.ga_line_orientation.min_line_length", 20)

    vertical_lines = 0
    horizontal_lines = 0

    for start_point, end_point in ransac_lines:
        start_homo = np.array([start_point[0], start_point[1], 1.0])
        end_homo = np.array([end_point[0], end_point[1], 1.0])
        start_transformed = h_matrix @ start_homo
        end_transformed = h_matrix @ end_homo
        if start_transformed[2] != 0 and end_transformed[2] != 0:
            start_2d = start_transformed[:2] / start_transformed[2]
            end_2d = end_transformed[:2] / end_transformed[2]
            x1, y1 = start_2d
            x2, y2 = end_2d
            if (
                min(x1, x2) > output_width
                or max(x1, x2) < 0
                or min(y1, y2) > output_height
                or max(y1, y2) < 0
            ):
                continue
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length < min_line_length:
                continue
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle > 90:
                angle = 180 - angle
            if angle >= (90 - vertical_tolerance):
                vertical_lines += 1
            elif angle <= horizontal_tolerance:
                horizontal_lines += 1

    base_score = 0.0
    missing_penalty = get_setting(
        "optimization.ga_line_orientation.missing_orientation_penalty", 0.5
    )
    if vertical_lines == 0 or horizontal_lines == 0:
        base_score -= missing_penalty
    if vertical_lines > 0 and horizontal_lines > 0:
        base_score += 0.6
        vertical_bonus = min(vertical_lines * 0.1, 0.2)
        horizontal_bonus = min(horizontal_lines * 0.1, 0.2)
        base_score += vertical_bonus + horizontal_bonus
    return max(0.0, min(1.0, base_score))
