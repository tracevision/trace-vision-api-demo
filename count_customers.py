"""
Retrieve line crossing events from one or more Trace Vision sessions and estimate
the number of customers.

This script assumes that you have a processed session ready to go. Use
create_trace_vision_session.py to first create a session and upload a video.

Demonstrate:
- Retrieve results from a session
- Extract line crossing events
- Estimate the number of customers based on entry and exit events

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  You will need to know the date for which you want to analyze sessions.
3.  Run the script using the command line:
        python count_customers.py \
            --customer_id 1234 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --date "2025-05-27"
"""
import argparse
import datetime
import json

import pandas as pd

from vision_api_interface import VisionAPIInterface


def get_entry_and_exit_events(session_ids, vision_api_interface):

    all_events_df = []
    for session_id in session_ids:
        # Get session results:
        print(f"Retrieving results for session {session_id}...")
        session_result_response = vision_api_interface.get_session_result(session_id, include_events=True)
        # Unpack objects into a Pandas DataFrame:
        events_df = vision_api_interface.get_events(
            session_result_response
        )
        if events_df.empty:
            continue

        # Filter out the overlap region which is the first 30 seconds
        session = vision_api_interface.get_session(session_id)
        start_time_str = json.loads(session.text)["data"]["session"]["videos"]["videos"][0]["start_time"]
        start_time = datetime.datetime.fromisoformat(start_time_str)
        start_time_without_overlap = start_time + datetime.timedelta(seconds=30)
        events_df["event_time"] = pd.to_datetime(events_df["event_time"])
        events_df = events_df[events_df["event_time"] > start_time_without_overlap]

        all_events_df.append(events_df)

    # Concatenate all objects into a single DataFrame
    events_df = pd.concat(all_events_df, ignore_index=True)

    # Filter to only line crossing events
    events_df = events_df[events_df["type"] == "line_crossing"]

    # Filter out lines that are not labeled as entryway
    entryway_shape_ids = []
    for shape_id in events_df["shape_id"].unique():
        shape = vision_api_interface.get_shape(shape_id)
        shape_metadata = json.loads(shape.text)["data"]["shape"]["metadata"]
        try:
            is_entryway = shape_metadata["retail"]["entryway"]
        except (KeyError, TypeError):
            is_entryway = False
        if is_entryway:
            entryway_shape_ids.append(shape_id)

    events_df = events_df[events_df["shape_id"].isin(entryway_shape_ids)]

    n_entry_events = len(events_df[events_df["direction"] == 1])
    n_exit_events = len(events_df[events_df["direction"] == -1])

    print(f"Number of entry events: {n_entry_events}")
    print(f"Number of exit events: {n_exit_events}")

    # Choose the maximum of entry and exit events as the estimated number of customers because either entries or exits
    # are usually easier to detect than the other, depending on the setup of the camera.
    estimated_n_customers = max(n_entry_events, n_exit_events)
    print(f"Estimated number of customers: {estimated_n_customers}")

    return n_entry_events, n_exit_events, estimated_n_customers


def main():
    """
    Main function to count customers in a given day. Can easily be adapted to count customers in an arbitrary time window.
    """
    # Define command line arguments:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer_id", required=True, type=int, help="Customer (division) ID"
    )
    ap.add_argument("--api_key", required=True, type=str, help="API key")
    ap.add_argument("--api_url", required=True, type=str, help="API URL")
    ap.add_argument("--date", required=True, type=str, help="Date to count customers. Format: YYYY-MM-DD")
    ap.add_argument("--utc_offset", required=True, type=int, help="UTC offset in hours. Example: -8 for PST")
    ap.add_argument("--facility_id", required=True, type=int, help="Facility ID")
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    date = args["date"]
    utc_offset = args["utc_offset"]
    facility_id = args["facility_id"]

    # Create an interface to help with the API calls:
    vision_api_interface = VisionAPIInterface(customer_id, api_key, api_url)

    available_facilities = vision_api_interface.get_facilities()
    facility = next((f for f in available_facilities if f["facility_id"] == facility_id), None)
    if not facility:
        raise ValueError(f"Facility ID {facility_id} not found. Available facility IDs: {[f['facility_id'] for f in available_facilities]}")
    
    camera_ids = [camera["camera_id"] for camera in facility["cameras"]]
    
    start_time = datetime.datetime.strptime(date, "%Y-%m-%d")
    start_time = start_time - datetime.timedelta(hours=utc_offset)
    end_time = start_time + datetime.timedelta(days=1)
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    session_ids = []
    for camera_id in camera_ids:
        get_sessions_response = vision_api_interface.get_sessions(
            camera_id=camera_id,
            start_time_min=start_time,
            start_time_max=end_time
        )
        session_ids.extend([session["session_id"] for session in get_sessions_response])

    get_entry_and_exit_events(session_ids, vision_api_interface)


if __name__ == "__main__":
    main()
