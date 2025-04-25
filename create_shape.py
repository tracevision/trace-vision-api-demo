#!/usr/bin/env python3
#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Ryan Hu
#
"""
Create shapes (line counters and polygons) using the Trace Vision GraphQL API.

This script demonstrates how to create shapes like line counters and polygons
for tracking and analytics purposes.

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  Run the script using the command line with the appropriate arguments:

    Create a line counter in image coordinates:
        python create_shape.py \
            --customer_id 42 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --shape_type "line" \
            --name "entry_counter" \
            --coordinate_type "image" \
            --camera_id 232 \
            --object_type "person" \
            --coordinates "[[30.7, 30.7], [50.7, 50.7]]"

    Create a polygon in world coordinates:
        python create_shape.py \
            --customer_id 42 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --shape_type "polygon" \
            --name "parking_area" \
            --coordinate_type "world" \
            --facility_id 232 \
            --object_type "vehicle" \
            --coordinates "[[37.774929, -122.419418], [37.774929, -122.419418], [37.774929, -122.419418]]"

Required Arguments:
    --customer_id: Your customer ID
    --api_key: Your API key
    --api_url: The API URL
    --shape_type: Type of shape to create (line or polygon)
    --name: Name of the shape
    --coordinate_type: Type of coordinates (image or world)
    --object_type: List of object types to track (e.g., "person", "vehicle")
    --coordinates: Coordinates in format [[x1,y1], [x2,y2]]

Optional Arguments:
    --facility_id: ID of the facility (required for world coordinates)
    --camera_id: ID of the camera (required for image coordinates)
    --metadata: Optional metadata in JSON format
    --enabled: Whether the shape is enabled (default: True)
"""
import argparse
import json
from vision_api_interface import VisionAPIInterface


def parse_coordinates(coord_str):
    """
    Parse coordinate string into list of points.
    
    :param coord_str: String representation of coordinates in format "[[x1,y1], [x2,y2]]"
    :return: List of coordinate dictionaries
    """
    try:
        coords = json.loads(coord_str)
        return [{"x": x, "y": y} for x, y in coords]
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid coordinate format. Expected [[x1,y1], [x2,y2]]. Error: {e}")


def main():
    """
    Main function to create a shape.
    """
    ap = argparse.ArgumentParser()
    # API credentials:
    ap.add_argument("--customer_id", type=int, required=True, help="Customer ID")
    ap.add_argument("--api_key", type=str, required=True, help="API key")
    ap.add_argument("--api_url", type=str, required=True, help="API URL")

    # Shape arguments:
    ap.add_argument(
        "--shape_type", 
        type=str, 
        choices=["line", "polygon"],
        required=True,
        help="Type of shape to create (line or polygon)"
    )
    ap.add_argument(
        "--name", 
        type=str, 
        required=True, 
        help="Name of the shape"
    )
    ap.add_argument(
        "--coordinate_type", 
        type=str, 
        choices=["image", "world"],
        required=True,
        help="Type of coordinates (image or world)"
    )
    ap.add_argument(
        "--facility_id", 
        type=int, 
        required=False, 
        help="Facility ID (required for world coordinates)"
    )
    ap.add_argument(
        "--camera_id", 
        type=int, 
        required=False, 
        help="Camera ID (required for image coordinates)"
    )
    ap.add_argument(
        "--object_type", 
        type=str, 
        nargs="+",
        required=True,
        help="List of object types to track (e.g., person car)"
    )
    ap.add_argument(
        "--coordinates", 
        type=str,
        required=True,
        help="Coordinates in format [[x1,y1], [x2,y2]]"
    )
    ap.add_argument(
        "--metadata", 
        type=str,
        required=False,
        help="Optional metadata in JSON format"
    )
    ap.add_argument(
        "--enabled", 
        type=bool, 
        default=True,
        required=True,
        help="Whether the shape is enabled (default: True)"
    )
    args = ap.parse_args()

    # Validate that the correct ID is provided based on coordinate_type
    if args.coordinate_type == "image" and args.camera_id is None:
        raise ValueError("camera_id is required when coordinate_type is 'image'")
    elif args.coordinate_type == "world" and args.facility_id is None:
        raise ValueError("facility_id is required when coordinate_type is 'world'")

    vision_api_interface = VisionAPIInterface(
        args.customer_id, args.api_key, args.api_url
    )

    # Create the shape input
    shape_input = {
        "shape_type": args.shape_type,
        "name": args.name,
        "coordinates": parse_coordinates(args.coordinates),
        "coordinate_type": args.coordinate_type,
        "object_type": args.object_type,
        "metadata": json.loads(args.metadata) if args.metadata else {},
        "enabled": args.enabled
    }

    # Add the appropriate ID based on coordinate_type
    if args.coordinate_type == "image":
        shape_input["camera_id"] = args.camera_id
    else:  
        shape_input["facility_id"] = args.facility_id

    # Make the API call
    print(f"Creating {args.shape_type} shape '{args.name}'")
    response = vision_api_interface.create_shape(shape_input)
    response_data = json.loads(response.text)
    print("Create shape response:", json.dumps(response_data, indent=2))


if __name__ == "__main__":
    main() 
