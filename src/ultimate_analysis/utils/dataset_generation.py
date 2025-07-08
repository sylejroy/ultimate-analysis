"""
Dataset generation utilities for Ultimate Analysis.

Contains functions for creating training datasets from video footage.
"""
import cv2
import os
import random
import glob
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger("ultimate_analysis.utils.dataset_generation")


def make_test_videos(
    input_folder: str = 'input',
    snippet_duration: int = 7,
    num_snippets: int = 4,
    dev_data_folder: Optional[str] = None
) -> None:
    """
    Create test video snippets from full videos.
    
    Args:
        input_folder: Path to folder containing input videos
        snippet_duration: Duration of each snippet in seconds
        num_snippets: Number of snippets to create per video
        dev_data_folder: Output folder for snippets (defaults to input/dev_data)
    """
    if dev_data_folder is None:
        dev_data_folder = os.path.join(input_folder, "dev_data")
    
    if not os.path.exists(dev_data_folder):
        os.makedirs(dev_data_folder)

    video_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video {video_file}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        snippet_frames = int(fps * snippet_duration)
        video_name = os.path.splitext(video_file)[0]

        for i in range(num_snippets):
            if total_frames <= snippet_frames:
                logger.warning(f"Video {video_file} is too short for a {snippet_duration}s snippet")
                break
                
            start_frame = random.randint(0, total_frames - snippet_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(dev_data_folder, f"{video_name}_snippet_{i+1}_{start_frame}.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
            
            for _ in range(snippet_frames):
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Could not read frame from {video_file}")
                    break
                out.write(frame)
                
            out.release()
            logger.info(f"Saved snippet {i+1} for {video_file} to {output_path}")
            
        cap.release()


def make_test_screenshot(
    video_path: str = 'input/portland_vs_san_francisco_2024.mp4',
    output_folder: str = 'input'
) -> Optional[str]:
    """
    Create a random screenshot from a video.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save the screenshot
        
    Returns:
        Path to the saved screenshot, or None if failed
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None
        
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate a random frame number
    random_frame_number = random.randint(0, total_frames - 1)
    
    # Set the video to the random frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        logger.error("Could not read frame")
        cap.release()
        return None

    # Save the frame as a screenshot with frame number in the filename
    cap.release()
    screenshot_path = os.path.join(output_folder, f'screenshot_frame_{random_frame_number}.jpg')

    cv2.imwrite(screenshot_path, frame)
    logger.info(f"Screenshot saved to {screenshot_path}")
    
    return screenshot_path


def sample_video_frames(
    video_path: str,
    output_folder: str,
    interval: int = 5
) -> None:
    """
    Sample frames from a video at regular intervals.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save the frames
        interval: Interval in seconds between frames
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video or error reading frame")
            break
            
        # Get a shorthand for the video name (without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Save the frame every `frame_interval` frames with video name shorthand
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
        
        # Print progress based on total frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Progress: {progress:.2f}%")

        frame_count += 1

    cap.release()


def sample_all_videos(
    input_folder: str = 'input',
    output_folder: str = 'input/dataset',
    interval: int = 20
) -> None:
    """
    Sample frames from all videos in a folder.
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save the sampled frames
        interval: Interval in seconds between frames
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        logger.info(f"Sampling frames from {video_file}...")
        sample_video_frames(video_path, output_folder, interval)
        
    logger.info("Frame sampling completed")


def crop_images_in_folder(
    input_folder: str,
    output_folder: str,
    crop_size: Tuple[int, int] = (640, 640)
) -> None:
    """
    Crop all images in a folder to a specified size.
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save cropped images
        crop_size: Target crop size as (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            logger.error(f"Error reading {image_file}. Skipping.")
            continue

        height, width = image.shape[:2]
        
        # Calculate the center crop
        start_x = (width - crop_size[0]) // 2
        start_y = (height - crop_size[1]) // 2
        cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]

        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, cropped_image)
        logger.info(f"Cropped and saved {output_path}")


def augment_dataset(
    input_folder: str,
    output_folder: str,
    augmentation_factor: int = 3
) -> None:
    """
    Apply data augmentation to images in a folder.
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save augmented images
        augmentation_factor: Number of augmented versions per image
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            logger.error(f"Error reading {image_file}. Skipping.")
            continue

        base_name = os.path.splitext(os.path.basename(image_file))[0]
        
        for i in range(augmentation_factor):
            # Apply random transformations
            augmented = apply_random_augmentation(image)
            
            output_path = os.path.join(output_folder, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(output_path, augmented)
            
        logger.info(f"Created {augmentation_factor} augmented versions of {image_file}")


def apply_random_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentation to an image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Augmented image
    """
    # Random brightness adjustment
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Random horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    
    return image


# Main execution for testing
if __name__ == "__main__":
    # Example usage
    logger.info("Starting dataset generation utilities...")
    
    # Uncomment the function you want to run
    # sample_all_videos(input_folder='input', output_folder='input/dataset', interval=20)
    # crop_images_in_folder(input_folder='input/dataset', output_folder='input/cropped_dataset', crop_size=(640, 640))
    make_test_videos()
    
    logger.info("Dataset generation completed.")
