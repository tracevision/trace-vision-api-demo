"""
Create facilities using the Trace Vision GraphQL API.

This script demonstrates how to create facilities for tracking and analytics purposes.

Usage:
    python create_facility.py \
        --customer_id 42 \
        --api_key "your_api_key" \
        --api_url "api_url" \
        --facility_name "facility" \
        --facility_latitude 100 \
        --facility_longitude -100

Required Arguments:
    --customer_id: Your customer ID
    --api_key: Your API key
    --api_url: The API URL
    --facility_name: Name of the new facility
    --facility_latitude: Latitude of the new facility (Example: '37.774929')
    --facility_longitude: Longitude of the new facility (Example: '-122.419418')
"""

import argparse
from vision_api_interface import VisionAPIInterface

# Facility configuration
facility_input = {
    "type": "retail",
    "metadata": {}  # Optional additional metadata
}

def main():
    ap = argparse.ArgumentParser()
    # API credentials:
    ap.add_argument("--customer_id", type=int, required=True, help="Customer ID")
    ap.add_argument("--api_key", type=str, required=True, help="API key")
    ap.add_argument("--api_url", type=str, required=True, help="API URL")

    # Facility arguments:
    ap.add_argument(
        "--facility_name",
        type=str,
        required=True,
        help="Name of the new facility",
    )
    ap.add_argument(
        "--facility_latitude",
        type=float,
        required=True,
        help="Latitude of the new facility, Example: '37.774929'",
    )
    ap.add_argument(
        "--facility_longitude",
        type=float,
        required=True,
        help="Longitude of the new facility, Example: '-122.419418'",
    )

    args = ap.parse_args()

    vision_api_interface = VisionAPIInterface(
        args.customer_id, args.api_key, args.api_url
    )

    facility_input["latitude"] = args.facility_latitude
    facility_input["longitude"] = args.facility_longitude
    facility_input["name"] = args.facility_name
    
    facility_response = vision_api_interface.create_new_facility(facility_input)
    facility_id = facility_response.json()["data"]["createFacility"]["facility"]["facility_id"]
    print(f"Created facility ID: {facility_id}")
    return facility_id

if __name__ == "__main__":
    main() 
    