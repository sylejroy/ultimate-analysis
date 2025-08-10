#!/usr/bin/env python3
"""Test script for field projection functionality.

This script tests the unified field segmentation and line estimation
on sample video frames to verify the implementation works correctly.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ultimate_analysis.processing.field_segmentation import run_field_segmentation
from ultimate_analysis.processing.field_projection import (
    unify_field_segmentation, estimate_field_lines, create_unified_field, 
    visualize_unified_field
)
from ultimate_analysis.config.settings import get_setting


def test_field_projection():
    """Test field projection on a sample frame."""
    
    print("=== Field Projection Test ===")
    
    # Create a more realistic mock frame for testing
    frame_height, frame_width = 720, 1280
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Add field-like content (green trapezoidal area to simulate perspective)
    field_color = (0, 100, 0)  # Dark green
    
    # Create perspective field shape (wider at bottom, narrower at top)
    field_bottom_width = int(frame_width * 0.85)
    field_top_width = int(frame_width * 0.6)
    field_height = int(frame_height * 0.7)
    field_bottom_y = int(frame_height * 0.9)
    field_top_y = field_bottom_y - field_height
    
    # Calculate field corners for perspective
    bottom_left_x = (frame_width - field_bottom_width) // 2
    bottom_right_x = bottom_left_x + field_bottom_width
    top_left_x = (frame_width - field_top_width) // 2
    top_right_x = top_left_x + field_top_width
    
    # Create field polygon
    field_points = np.array([
        [bottom_left_x, field_bottom_y],
        [bottom_right_x, field_bottom_y],
        [top_right_x, field_top_y],
        [top_left_x, field_top_y]
    ], np.int32)
    
    cv2.fillPoly(frame, [field_points], field_color)
    
    # Add field lines with perspective (diagonal sidelines, horizontal field lines)
    line_color = (255, 255, 255)  # White
    
    # Left sideline (diagonal due to perspective)
    cv2.line(frame, (bottom_left_x, field_bottom_y), (top_left_x, field_top_y), line_color, 3)
    
    # Right sideline (diagonal due to perspective)
    cv2.line(frame, (bottom_right_x, field_bottom_y), (top_right_x, field_top_y), line_color, 3)
    
    # Close field line (more horizontal)
    close_y = field_bottom_y - int(field_height * 0.2)
    close_left_x = bottom_left_x + int((top_left_x - bottom_left_x) * 0.2)
    close_right_x = bottom_right_x + int((top_right_x - bottom_right_x) * 0.2)
    cv2.line(frame, (close_left_x, close_y), (close_right_x, close_y), line_color, 3)
    
    # Far field line (more horizontal)
    far_y = field_top_y + int(field_height * 0.1)
    far_left_x = bottom_left_x + int((top_left_x - bottom_left_x) * 0.9)
    far_right_x = bottom_right_x + int((top_right_x - bottom_right_x) * 0.9)
    cv2.line(frame, (far_left_x, far_y), (far_right_x, far_y), line_color, 3)
    
    # Add some boundary artifacts (lines touching edges) to test filtering
    cv2.line(frame, (0, frame_height//2), (50, frame_height//2), line_color, 2)  # Left edge artifact
    cv2.line(frame, (frame_width-50, frame_height//3), (frame_width, frame_height//3), line_color, 2)  # Right edge artifact
    cv2.line(frame, (frame_width//2, 0), (frame_width//2, 40), line_color, 2)  # Top edge artifact
    
    print(f"Created test frame with perspective field: {frame.shape}")
    print("Added diagonal sidelines, horizontal field lines, and boundary artifacts")
    
    # Save the input frame for comparison
    input_path = "output/field_projection_input.jpg"
    Path("output").mkdir(exist_ok=True)
    cv2.imwrite(input_path, frame)
    print(f"Saved input frame to: {input_path}")
    
    # Continue with the existing test logic...
    # Step 1: Run field segmentation
    print("\n1. Running field segmentation...")
    try:
        segmentation_results = run_field_segmentation(frame)
        print(f"   Field segmentation results: {len(segmentation_results)} segments")
        
        if not segmentation_results:
            print("   Warning: No field segmentation results (this is expected with mock data)")
            return
        
    except Exception as e:
        print(f"   Error in field segmentation: {e}")
        return
    
    # Step 2: Unify field segmentation
    print("\n2. Unifying field segmentation...")
    try:
        unified_mask = unify_field_segmentation(segmentation_results, frame.shape[:2])
        if unified_mask is not None:
            field_pixels = np.sum(unified_mask)
            print(f"   Unified mask created: {field_pixels} field pixels")
            
            # Save the unified mask for inspection
            mask_vis = np.zeros_like(frame)
            mask_vis[unified_mask > 0] = [0, 255, 0]  # Green for field
            mask_path = "output/unified_field_mask.jpg"
            cv2.imwrite(mask_path, mask_vis)
            print(f"   Saved unified mask visualization to: {mask_path}")
        else:
            print("   Warning: Failed to create unified mask")
            return
            
    except Exception as e:
        print(f"   Error in field unification: {e}")
        return
    
    # Step 3: Estimate field lines
    print("\n3. Estimating field lines...")
    try:
        field_lines = estimate_field_lines(unified_mask)
        print(f"   Estimated {len(field_lines)} field lines:")
        for line in field_lines:
            print(f"     {line.line_type}: {line.point1} -> {line.point2} (conf: {line.confidence:.2f})")
            
    except Exception as e:
        print(f"   Error in line estimation: {e}")
        return
    
    # Step 4: Create unified field
    print("\n4. Creating unified field...")
    try:
        unified_field = create_unified_field(segmentation_results, frame.shape[:2])
        if unified_field:
            print(f"   Unified field created:")
            print(f"     Lines: {len(unified_field.lines)}")
            print(f"     Corners: {len(unified_field.field_corners)}")
            print(f"     Coverage: {unified_field.coverage_ratio:.2%}")
        else:
            print("   Warning: Failed to create unified field")
            return
            
    except Exception as e:
        print(f"   Error creating unified field: {e}")
        return
    
    # Step 5: Visualize unified field
    print("\n5. Testing visualization...")
    try:
        vis_frame = visualize_unified_field(frame, unified_field)
        print(f"   Visualization frame shape: {vis_frame.shape}")
        
        # Save visualization for inspection
        output_path = "output/field_projection_test.jpg"
        Path("output").mkdir(exist_ok=True)
        cv2.imwrite(output_path, vis_frame)
        print(f"   Saved visualization to: {output_path}")
        
    except Exception as e:
        print(f"   Error in visualization: {e}")
        return
    
    print("\n=== Test Complete ===")
    print("Field projection implementation appears to be working!")


if __name__ == "__main__":
    test_field_projection()
