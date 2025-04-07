"""
Create facilities, cameras, and sessions using the Trace Vision GraphQL API.

This script demonstrates how to create and manage facilities, cameras, and video sessions
for tracking and analytics purposes.

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  Run the script using the command line with the appropriate arguments:

    Create a new facility:
        python create_facility_camera_and_session.py \
            --customer_id 42 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --facility_name "facility" \
            --facility_latitude 100 \
            --facility_longitude -100

    Create a new camera in an existing facility:
        python create_facility_camera_and_session.py \
            --customer_id 42 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --camera_name "camera_name" \
            --facility_id 232 \
            --camera_group_id "camera_group"

    Create a new session with a video file:
        python create_facility_camera_and_session.py \
            --customer_id 42 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --video_filepath "/path/to/video.mp4" \
            --video_start_time "2023-09-13T03:02:30Z" \
            --facility_id 232 \
            --camera_id 911

Required Arguments:
    --customer_id: Your customer ID
    --api_key: Your API key
    --api_url: The API URL

Optional Arguments:
    --facility_id: ID of an existing facility (if not provided, a new facility will be created)
    --facility_name: Name of the new facility (required if facility_id is not provided)
    --facility_latitude: Latitude of the new facility (required if facility_id is not provided)
    --facility_longitude: Longitude of the new facility (required if facility_id is not provided)
    --camera_id: ID of an existing camera (if not provided, a new camera will be created)
    --camera_name: Name of the new camera (required if camera_id is not provided)
    --camera_group_id: Group ID of the new camera (if not provided, a random ID will be generated)
    --video_filepath: Path to the video file to be uploaded (if provided, a new session will be created)
    --video_start_time: Start time of the video in ISO 8601 format (if not provided, current time will be used)
    --indoor: Whether the camera is indoors (1) or outdoors (0). Default is 0 (outdoors).
    --scene_type: Scene type of the camera. Default is 'drive_through'.
"""

import argparse
import datetime
import json
import random
import string

from vision_api_interface import VisionAPIInterface


# Facility configuration
facility_input = {
    "type": "retail",
    "metadata": {}  # Optional additional metadata
}

camera_input = {
    "model": "model",
    "indoor": False,
    "scene_type": "drive_through",
    "metadata": {},
    "enabled": True,
}

session_input = {
    "type": "general_video",
    "game_info": {
        "home_team": {"name": "dummy_home_team", "score": 0, "color": "#000000"},
        "away_team": {"name": "dummy_away_team", "score": 0, "color": "#ffffff"},
        "start_time": "2023-09-13T03:02:30Z",
    },
    "capabilities": ["tracking", "highlights", "cross-cam-tracking"],
}


def main():
    ap = argparse.ArgumentParser()
    # API credentials:
    ap.add_argument("--customer_id", type=int, required=True, help="Customer ID")
    ap.add_argument("--api_key", type=str, required=True, help="API key")
    ap.add_argument("--api_url", type=str, required=True, help="API URL")

    # Soccer arguments:
    ap.add_argument(
        "--soccer",
        action="store_true",
        help="Create a soccer session",
    )

    # Facility arguments:
    ap.add_argument(
        "--facility_id",
        type=int,
        required=False,
        help="If not provided, a new facility will be created",
    )
    ap.add_argument(
        "--facility_name",
        type=str,
        required=False,
        help="Name of the new facility. Only required if facility_id is not provided.",
    )
    ap.add_argument(
        "--facility_latitude",
        type=float,
        required=False,
        help="Latitude of the new facility, Example: '37.774929'. Only required if facility_id is not provided.",
    )
    ap.add_argument(
        "--facility_longitude",
        type=float,
        required=False,
        help="Longitude of the new facility, Example: '-122.419418'. Only required if facility_id is not provided.",
    )

    # Camera arguments:
    ap.add_argument(
        "--camera_id",
        type=int,
        required=False,
        help="If not provided, a new camera will be created",
    )
    ap.add_argument(
        "--camera_name",
        type=str,
        required=False,
        help="Name of the new camera. Only required if camera_id is not provided.",
    )
    ap.add_argument(
        "--camera_group_id",
        type=str,
        required=False,
        help="Group ID of the new camera. If not provided, a new group ID will be created.",
    )
    ap.add_argument(
        "--indoor",
        type=int,
        choices=[0, 1],
        default=0,
        required=False,
        help="Whether the camera is indoors (1) or outdoors (0). Default is 0 (outdoors).",
    )
    ap.add_argument(
        "--scene_type",
        type=str,
        choices=['drive_through', 'parking_lot', 'entryway'],
        default='drive_through',
        required=False,
        help="Scene type of the camera. One of: 'drive_through', 'parking_lot', 'entryway'. Default is 'drive_through'.",
    )

    # Session arguments:
    ap.add_argument(
        "--video_filepath",
        type=str,
        required=False,
        help="Path to the video file to be uploaded. If provided, a new session will be created.",
    )
    ap.add_argument(
        "--video_start_time",
        type=str,
        required=False,
        help="Start time of the video in ISO 8601 format. Example: 2023-09-13T03:02:30Z. If not provided, the current time will be used. Only required if video_filepath is provided.",
    )
    args = ap.parse_args()

    vision_api_interface = VisionAPIInterface(
        args.customer_id, args.api_key, args.api_url
    )

    if args.facility_id is None:
        assert (
            args.facility_name is not None
        ), "facility_name must be provided if facility_id is not provided"
        assert (
            args.facility_latitude is not None and args.facility_longitude is not None
        ), "facility_latitude and facility_longitude must be provided if facility_id is not provided"
        facility_input["latitude"] = args.facility_latitude
        facility_input["longitude"] = args.facility_longitude
        facility_input["name"] = args.facility_name
        facility_response = vision_api_interface.create_new_facility(facility_input)
        facility_id = facility_response.json()["data"]["createFacility"]["facility"][
            "facility_id"
        ]
        print(f"Created facility ID: {facility_id}")
    else:
        facility_id = args.facility_id

    if args.camera_id is None:
        assert (
            args.camera_name is not None
        ), "Camera name must be provided if camera ID is not provided"
        camera_input["facility_id"] = facility_id
        camera_input["name"] = args.camera_name
        camera_input["indoor"] = bool(args.indoor)
        
        # Set scene_type to None if camera is indoor, otherwise use the provided or default value
        if camera_input["indoor"]:
            camera_input["scene_type"] = None
        else:
            camera_input["scene_type"] = args.scene_type
            
        if args.camera_group_id is not None:
            camera_input["group_id"] = args.camera_group_id
        else:
            camera_input["group_id"] = "".join(
                random.choices(string.ascii_letters + string.digits, k=16)
            )
        camera_response = vision_api_interface.create_new_camera(camera_input)
        camera_id = camera_response.json()["data"]["createCamera"]["camera"][
            "camera_id"
        ]
        print(f"Created camera ID: {camera_id}")
    else:
        camera_id = args.camera_id

    if args.video_filepath is not None:
        if args.soccer:
            session_input["type"] = "soccer_game"
        session_input["camera_id"] = camera_id
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


if __name__ == "__main__":
    main()
