import os
import argparse
import cv2
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Extract Frames from a Video.")

    parser.add_argument(
        "--video-file", 
        type=str, 
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save video frames."
    )
    parser.add_argument(
        "--skip-frame", 
        type=int, 
        default=15, 
        help="Number of frames to skip between processing. (default: 15)"
    )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(args.video_file)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # If no frame is read, the video has ended
        if not ret:
            break

        if frame_count%15==0:
            # Save the frame as an image
            frame_filename = os.path.join(args.output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"\nTotal {frame_count//args.skip_frame} frames saved to: {args.output_dir}")