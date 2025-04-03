#
#  Copyright Alpinereplay Inc., 2025. All rights reserved.
#  Authors: Claire Roberts-Thomson
#
"""
Draw line and polygon counters on a sample frame image.
"""
import argparse
import os

import cv2

from vision_api_interface import VisionAPIInterface


def main():
    """
    Main function to retrieve and use results from a Trace Vision session.
    """
    # Define command line arguments:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer_id", required=True, type=int, help="Customer (division) ID"
    )
    ap.add_argument("--api_key", required=True, type=str, help="API key")
    ap.add_argument("--api_url", required=True, type=str, help="API URL")
    ap.add_argument("--camera_id", required=True, type=int, help="Camera ID")
    ap.add_argument(
        "--sample_frame_filepath",
        required=True,
        type=str,
        help="Path to sample frame image",
    )
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    camera_id = args["camera_id"]
    sample_frame_filepath = args["sample_frame_filepath"]

    # Create an interface to help with the API calls:
    vision_api_interface = VisionAPIInterface(customer_id, api_key, api_url)

    # Get shapes:
    shapes = vision_api_interface.get_shapes(camera_id)
    print(shapes)

    # Draw shapes on sample frame:
    frame = cv2.imread(sample_frame_filepath)
    for shape in shapes:
        print(shape)
        for i, cur_coords in enumerate(shape["coordinates"]):
            if i == 0:
                text_string = f"Name: {shape['name']}, ID: {shape['shape_id']}, Type: {shape['shape_type']}, Enabled: {shape['enabled']}"
                cv2.putText(
                    frame,
                    text_string,
                    (cur_coords["x"], cur_coords["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            if i == len(shape["coordinates"]) - 1:
                break
            next_coords = shape["coordinates"][i + 1]
            cv2.line(
                frame,
                (cur_coords["x"], cur_coords["y"]),
                (next_coords["x"], next_coords["y"]),
                (0, 0, 255),
                2,
            )

    out_dir, out_filename = os.path.split(sample_frame_filepath)
    out_filename_base, out_file_extension = os.path.splitext(out_filename)
    out_filepath = os.path.join(
        out_dir, f"{out_filename_base}_with_shapes{out_file_extension}"
    )
    cv2.imwrite(out_filepath, frame)


if __name__ == "__main__":
    main()
