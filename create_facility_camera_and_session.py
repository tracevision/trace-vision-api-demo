"""
Example usage:
python vision_api/setup_everything.py --facility_id 220 --camera_id 889 --video_filepath /home/ubuntu/scratch2/60_min_12fps_resized/1696557665.mp4
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
