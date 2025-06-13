# sample the video input\portland_vs_san_francisco_2024.mp4 once every 5 seconds and output to a folder called "training data"
import cv2
import os
import random
from ultralytics.data.augment import Compose, RandomPerspective, RandomFlip
import glob
import yaml
import numpy as np


def make_test_videos():
    # Loop through all videos in the input folder and make 4 snippets of 7 seconds each from each video        
    input_folder = 'input'
    snippet_duration = 7  # seconds
    num_snippets = 4
    dev_data_folder = os.path.join(input_folder, "dev_data")
    if not os.path.exists(dev_data_folder):
        os.makedirs(dev_data_folder)

    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        snippet_frames = int(fps * snippet_duration)
        video_name = os.path.splitext(video_file)[0]

        for i in range(num_snippets):
            if total_frames <= snippet_frames:
                print(f"Video {video_file} is too short for a {snippet_duration}s snippet.")
                break
            start_frame = random.randint(0, total_frames - snippet_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(dev_data_folder, f"{video_name}_snippet_{i+1}_{start_frame}.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
            for _ in range(snippet_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Could not read frame from {video_file}.")
                    break
                out.write(frame)
            out.release()
            print(f"Saved snippet {i+1} for {video_file} to {output_path}")
        cap.release()
 
def make_test_screenshot():
    # Create an image randomly selected from the video
    video_path = 'input/portland_vs_san_francisco_2024.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return  
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Generate a random frame number
    random_frame_number = random.randint(0, total_frames - 1)
    # Set the video to the random frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    # Save the frame as a screenshot with frame number in the filename
    cap.release()
    screenshot_path = f'input/screenshot_frame_{random_frame_number}.jpg'

    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved to {screenshot_path}")
    # Display the screenshot
    cv2.imshow('Screenshot', frame)
    cv2.waitKey(0)

def sample_video_frames(video_path, output_folder, interval=5):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        # Get a shorthand for the video name (without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Save the frame every `frame_interval` frames with video name shorthand
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
        
        # Print progress based on total frames
        print(f"\rProgress: {frame_count/total_frames*100:.2f}%", end="")

        frame_count += 1

    cap.release()

def sample_all_videos(input_folder='input', output_folder='input/dataset', interval=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Sampling frames from {video_file}...")
        sample_video_frames(video_path, output_folder, interval)
    print("Frame sampling completed.")

def crop_images_in_folder(input_folder, output_folder, crop_size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error reading {image_file}. Skipping.")
            continue

        height, width = image.shape[:2]
        # Calculate the center crop
        start_x = (width - crop_size[0]) // 2
        start_y = (height - crop_size[1]) // 2
        cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]

        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped and saved {output_path}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    #sample_all_videos(input_folder='input', output_folder='input/dataset', interval=20)
    #crop_images_in_folder(input_folder='input/dataset', output_folder='input/cropped_dataset', crop_size=(640, 640))
    make_test_videos()

