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
2.  You will need the session IDs of the sessions you want to analyze.
3.  Run the script using the command line:
        python count_customers.py \
            --customer_id 1234 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --session_ids 5678 91011
"""
import argparse
import os

import pandas as pd

from vision_api_interface import VisionAPIInterface


def main():
    """
    Main function to generate a similarity matrix from object feature vectors.
    """
    # Define command line arguments:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer_id", required=True, type=int, help="Customer (division) ID"
    )
    ap.add_argument("--api_key", required=True, type=str, help="API key")
    ap.add_argument("--api_url", required=True, type=str, help="API URL")
    ap.add_argument(
        "--session_ids",
        required=True,
        type=int,
        nargs="+",
        help="One or more session IDs",
    )
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    session_ids = args["session_ids"]

    # Create an interface to help with the API calls:
    vision_api_interface = VisionAPIInterface(customer_id, api_key, api_url)

    all_events_df = []
    for session_id in session_ids:
        # Get session results:
        print(f"Retrieving results for session {session_id}...")
        session_result_response = vision_api_interface.get_session_result(session_id, include_events=True)

        # Unpack objects into a Pandas DataFrame:
        events_df = vision_api_interface.get_events(
            session_result_response
        )
        print(f"Found {len(events_df)} events in session {session_id}.")
        if not events_df.empty:
            events_df["session_id"] = session_id

        all_events_df.append(events_df)

    # Concatenate all objects into a single DataFrame:
    events_df = pd.concat(all_events_df, ignore_index=True)
    events_df = events_df[events_df["type"] == "line_crossing"]

    n_entry_events = len(events_df[events_df["direction"] == 1])
    n_exit_events = len(events_df[events_df["direction"] == -1])

    print(f"Number of entry events: {n_entry_events}")
    print(f"Number of exit events: {n_exit_events}")
    estimated_n_customers = (n_entry_events + n_exit_events) // 2
    print(f"Estimated number of customers: {estimated_n_customers}")


if __name__ == "__main__":
    main()
