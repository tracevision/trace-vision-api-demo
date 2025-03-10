#
#  Copyright Alpinereplay Inc., 2025. All rights reserved.
#  Authors: Daniel Furman
#
"""
Resizes the video to 1280x720 or 720x1280 (depending on the orientation) and 
changes the fps to 12. This is to fit the format that TraceVision API requires.

Allowed resolutions:
- 1280x720 (landscape)
- 640x480 (landscape)
- 640x360 (landscape)
- 720x1280 (portrait)
- 480x640 (portrait)
- 360x640 (portrait)

Allowed FPS range: 4.9 - 12.5 fps

Requires ffmpeg and opencv-python to be installed.

Example usage:
python resize_and_resample.py --input_path /path/to/video.mp4 --output_path /path/to/output/video.mp4
"""
import argparse
import glob
import os
import subprocess
import sys
import cv2

def get_video_dimensions(video_path):
    """
    Get the width, height and fps of a video using OpenCV.
    Args:
        video_path (str): The path to the video file.
    Returns:
        tuple: A tuple containing the width, height and fps of the video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    # Get width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    cap.release()
    
    return width, height, fps

def resize_video(input_path, output_path, target_width, target_height, target_fps=12):
    """
    Resize a video to the specified dimensions using ffmpeg, optionally changing fps.
    Args:
        input_path (str): The path to the input video file.
        output_path (str): The path to the output video file.
        target_width (int): The width to resize the video to.
        target_height (int): The height to resize the video to.
        target_fps (int): The fps to change the video to.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare the filter string
    filter_string = f'scale={target_width}:{target_height}'
    if target_fps is not None:
        filter_string += f',fps={target_fps}'
        print(f"Resizing video to {target_width}x{target_height} and changing FPS to {target_fps}...")
    else:
        print(f"Resizing video to {target_width}x{target_height}...")
    
    # Base command for resizing
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', filter_string,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-c:a', 'copy',
        output_path, 
        '-y'
    ]
    
    # Run ffmpeg with output displayed in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Error: ffmpeg exited with code {return_code}")
        sys.exit(1)
    
    print(f"Successfully processed video to {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="The path to the input video file.")
    ap.add_argument("--output_path", required=True, help="The path to the output video file.")
    args = ap.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    output_path = input_path.replace('60_min_chunks_right', '60_min_12fps_resized_right')
    print(f"Resizing video {input_path} to {output_path}")

    # Get current dimensions and fps
    width, height, fps = get_video_dimensions(input_path)
    print(f"Original video dimensions: {width}x{height}, FPS: {fps}")
    
    # Determine target dimensions based on whether width or height is larger
    if width > height:
        # Landscape orientation
        target_width, target_height = 1280, 720
    else:
        # Portrait orientation
        target_width, target_height = 720, 1280
    
    # Resize the video and optionally change FPS
    resize_video(input_path, output_path, target_width, target_height)

if __name__ == "__main__":
    main() 