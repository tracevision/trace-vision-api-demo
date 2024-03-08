#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Claire Roberts-Thomson
#
"""
Create a Trace Vision sesion using the GraphQL API and upload a video.

Demonstrate:
- Find all existing sessions
- Create a new session
- Upload a video to the new session

Since soccer games are supported, we'll create a session for soccer game
footage.

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  Find a video containing soccer footage.
3.  Create a session input JSON file containing session input data like this:
{
    "type": "soccer_game",
    "game_info": {
        "home_team": {
            "name": "Red team",
            "score": 0,
            "color": "#ff0000"
        },
        "away_team": {
            "name": "Blue team",
            "score": 0,
            "color": "#0000ff"
        },
        "start_time": "2024-01-01T12:00:00Z"
    },
    "capabilities": ["tracking", "highlights"]
}
    *   Note that it's important to set the team colors to match the jersey
        colors of the teams in the video.
4.  Run the script using the command line:
        python create_trace_vision_session.py \
            --customer_id 1234 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --session_input_json_path "path/to/session_input.json" \
            --video_filepath "path/to/video.mp4"

You may want to run this script in a debugger to step through the code and
examine API responses.
"""
import argparse
import json
import os

from vision_api_interface import VisionAPIInterface


def main(
    customer_id, api_key, api_url, session_input_json_path, video_filepath
):
    """
    Create a new Trace Vision session and upload a video to it.

    :param customer_id: Customer (division) ID
    :param api_key: API key
    :param api_url: API URL
    :param video_filepath: Path to video file
    :param out_dir: Path to directory to save results
    """
    # Create an interface to help with the API calls:
    vision_api_interface = VisionAPIInterface(customer_id, api_key, api_url)

    # Get all available vision sessions for your customer ID:
    get_sessions_response = (
        vision_api_interface.get_all_available_vision_sessions()
    )

    # List all available sessions:
    vision_api_interface.list_sessions(get_sessions_response)

    # Load session input data from JSON file:
    with open(session_input_json_path, "r") as f:
        session_input = json.load(f)

    # Create a session:
    create_session_response = vision_api_interface.create_new_session(
        session_input
    )

    # Get the new session ID from the response:
    session_id = vision_api_interface.get_session_id(create_session_response)

    print(f"Created session with ID: {session_id}")

    # Find the video file size:
    # Note that AWS S3 has a limit of 5 GB for the size of files that can be
    # uploaded with a single PUT request. If the video file is larger than 5
    # GB, use multipart upload.
    video_filesize_bytes = os.path.getsize(video_filepath)
    print(f"Video file size: {video_filesize_bytes} bytes")
    max_single_upload_bytes = 4.9 * 1024 * 1024 * 1024  # 4.9 (just under 5) GB
    if video_filesize_bytes < max_single_upload_bytes:
        # Upload video in a single PUT request
        (
            upload_video_response,
            put_video_response,
        ) = vision_api_interface.upload_video(
            session_id, session_input, video_filepath
        )
    else:
        # Use multi-part upload to upload the video
        print("Using multi-part upload")
        # Calculate the number of parts to split the video into:
        n_parts = int(
            round(video_filesize_bytes / max_single_upload_bytes + 0.5)
        )
        (
            upload_video_response,
            put_video_responses,
            complete_multipart_upload_response,
        ) = vision_api_interface.upload_video_multipart(
            session_id,
            session_input,
            video_filepath,
            n_parts,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer_id", required=True, type=int, help="Customer (division) ID"
    )
    ap.add_argument("--api_key", required=True, type=str, help="API key")
    ap.add_argument("--api_url", required=True, type=str, help="API URL")
    ap.add_argument(
        "--session_input_json_path",
        required=True,
        type=str,
        help="Path to session input JSON file",
    )
    ap.add_argument(
        "--video_filepath", required=True, type=str, help="Path to video file"
    )
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    session_input_json_path = args["session_input_json_path"]
    video_filepath = args["video_filepath"]

    main(
        customer_id, api_key, api_url, session_input_json_path, video_filepath
    )
