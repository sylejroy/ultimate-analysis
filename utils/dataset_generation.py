# sample the video input\portland_vs_san_francisco_2024.mp4 once every 5 seconds and output to a folder called "training data"
import cv2
import os

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

        # Save the frame every `frame_interval` frames
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)

        # Print progress based on total frames
        print(f"\rProgress: {frame_count/total_frames*100:.2f}%", end="")

        frame_count += 1

    cap.release()

def make_test_video():
    # Create a shorter version of the video - random 10 seconds snippet
    video_path = 'input/portland_vs_san_francisco_2024.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    # Get the total number of frames and the frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames for 10 seconds
    snippet_frames = int(fps * 7)  # 10 seconds snippet
    # Generate a random starting frame number
    start_frame = random.randint(0, total_frames - snippet_frames)
    # Set the video to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # Create a VideoWriter object to save the snippet
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    output_path = f'input/portland_vs_san_francisco_snippet_{start_frame}.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    # Read and write the frames for the snippet
    for _ in range(snippet_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        out.write(frame)
    # Release resources
    cap.release()
    out.release()

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

# Usage
video_path = 'input/portland_vs_san_francisco_2024.mp4'
output_folder = 'training_data'
sample_video_frames(video_path, output_folder)
