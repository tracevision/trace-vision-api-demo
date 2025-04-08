"""
Create cameras using the Trace Vision GraphQL API.

This script demonstrates how to create cameras for tracking and analytics purposes.

Usage:
    python create_camera.py \
        --customer_id 42 \
        --api_key "your_api_key" \
        --api_url "api_url" \
        --camera_name "camera_name" \
        --facility_id 232 \
        --camera_group_id "camera_group" \
        --indoor 0 \
        --scene_type "drive_through"

Required Arguments:
    --customer_id: Your customer ID
    --api_key: Your API key
    --api_url: The API URL
    --camera_name: Name of the new camera
    --facility_id: ID of the facility to add the camera to

Optional Arguments:
    --camera_group_id: Group ID of the new camera (if not provided, Group ID will be NULL)
    --indoor: Whether the camera is indoors (1) or outdoors (0). Default is 0 (outdoors)
    --scene_type: Scene type of the camera. One of: 'drive_through', 'parking_lot', 'entryway'. Default is None
"""

import argparse
from vision_api_interface import VisionAPIInterface

# Camera configuration
camera_input = {
    "model": "model",
    "indoor": False,
    "scene_type": None,
    "metadata": {},
    "enabled": True,
}

def main():
    ap = argparse.ArgumentParser()
    # API credentials:
    ap.add_argument("--customer_id", type=int, required=True, help="Customer ID")
    ap.add_argument("--api_key", type=str, required=True, help="API key")
    ap.add_argument("--api_url", type=str, required=True, help="API URL")

    # Camera arguments:
    ap.add_argument(
        "--camera_name",
        type=str,
        required=True,
        help="Name of the new camera",
    )
    ap.add_argument(
        "--facility_id",
        type=int,
        required=True,
        help="ID of the facility to add the camera to",
    )
    ap.add_argument(
        "--camera_group_id",
        type=str,
        default=None,
        required=False,
        help="Group ID of the new camera. If not provided, group ID will be NULL.",
    )
    ap.add_argument(
        "--indoor",
        type=int,
        choices=[0, 1],
        default=None,
        required=False,
        help="Whether the camera is indoors (1) or outdoors (0). Default is NULL.",
    )
    ap.add_argument(
        "--scene_type",
        type=str,
        choices=['drive_through', 'parking_lot', 'entryway'],
        default=None,
        required=False,
        help="Scene type of the camera. One of: 'drive_through', 'parking_lot', 'entryway'. Default is NULL",
    )

    args = ap.parse_args()

    vision_api_interface = VisionAPIInterface(
        args.customer_id, args.api_key, args.api_url
    )

    camera_input["facility_id"] = args.facility_id
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
        camera_input["group_id"] = None
    
    camera_response = vision_api_interface.create_new_camera(camera_input)
    camera_id = camera_response.json()["data"]["createCamera"]["camera"]["camera_id"]
    print(f"Created camera ID: {camera_id}")
    return camera_id

if __name__ == "__main__":
    main() 
    