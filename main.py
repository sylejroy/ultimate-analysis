import cv2
import random

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
    snippet_frames = int(fps * 10)  # 10 seconds snippet
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

def main():
    #read video file portland_vs_san_francisco_2024 in input folder and display it

    video_path = 'input/portland_vs_san_francisco_2024.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    #make_test_screenshot()
    make_test_video()



