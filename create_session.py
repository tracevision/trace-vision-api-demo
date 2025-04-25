"""
Create sessions using the Trace Vision GraphQL API.

This script demonstrates how to create video sessions for tracking and analytics purposes.

Usage:
    python create_session.py \
        --customer_id 42 \
        --api_key "your_api_key" \
        --api_url "api_url" \
        --video_filepath "/path/to/video.mp4" \
        --facility_id 232 \
        --camera_id 911 \
        --video_start_time "2023-09-13T03:02:30Z" \
        --soccer

Required Arguments:
    --customer_id: Your customer ID
    --api_key: Your API key
    --api_url: The API URL
    --video_filepath: Path to the video file to be uploaded
    --facility_id: ID of the facility
    --camera_id: ID of the camera

Optional Arguments:
    --video_start_time: Start time of the video in ISO 8601 format. If not provided, current time will be used
    --soccer: Flag to create a soccer session instead of a general video session
"""

import argparse
import datetime
import json
from vision_api_interface import VisionAPIInterface

# Session configuration
session_input = {
    "type": "general_video",
    "game_info": {
        "home_team": {"name": "dummy_home_team", "score": 0, "color": "#000000"},
        "away_team": {"name": "dummy_away_team", "score": 0, "color": "#ffffff"},
        "start_time": "2023-09-13T03:02:30Z",
    },
    "capabilities": ["tracking", "highlights", "cross-cam-tracking", "videos"],
}

def main():
    ap = argparse.ArgumentParser()
    # API credentials:
    ap.add_argument("--customer_id", type=int, required=True, help="Customer ID")
    ap.add_argument("--api_key", type=str, required=True, help="API key")
    ap.add_argument("--api_url", type=str, required=True, help="API URL")

    # Session arguments:
    ap.add_argument(
        "--video_filepath",
        type=str,
        required=True,
        help="Path to the video file to be uploaded",
    )
    ap.add_argument(
        "--facility_id",
        type=int,
        required=True,
        help="ID of the facility",
    )
    ap.add_argument(
        "--camera_id",
        type=int,
        required=True,
        help="ID of the camera",
    )
    ap.add_argument(
        "--video_start_time",
        type=str,
        required=False,
        help="Start time of the video in ISO 8601 format. Example: 2023-09-13T03:02:30Z. If not provided, the current time will be used.",
    )
    ap.add_argument(
        "--soccer",
        action="store_true",
        help="Create a soccer session instead of a general video session",
    )

    args = ap.parse_args()

    vision_api_interface = VisionAPIInterface(
        args.customer_id, args.api_key, args.api_url
    )

    if args.soccer:
        session_input["type"] = "soccer_game"
    
    session_input["camera_id"] = args.camera_id
    session_input["game_info"]["start_time"] = (
        args.video_start_time
        if args.video_start_time is not None
        else datetime.datetime.now().isoformat()
    )
    
    session_input_path = "tmp_session_input.json"
    with open(session_input_path, "w") as f:
        json.dump(session_input, f)
    
    session_id = vision_api_interface.create_session_from_json_and_video_file(
        session_input_path, args.video_filepath
    )
    print(f"Created session ID: {session_id}")
    return session_id

if __name__ == "__main__":
    main() 
