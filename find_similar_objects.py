
#
#  Copyright Alpinereplay Inc., 2025. All rights reserved.
#  Authors: Daniel Furman
#
"""
Download object feature vectors from a Trace Vision session and compute a
similarity matrix.

This script assumes that you have a processed session ready to go. Use
create_trace_vision_session.py to first create a session and upload a video.

Demonstrate:
- Find all existing sessions
- Retrieve results from a session
- Download object data from a session
- Compute a similarity matrix between objects based on their feature vectors
- For each object, find the top 10 most similar objects

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  You will need the session IDs of the sessions you want to analyze.
3.  Run the script using the command line:
        python find_similar_objects.py \
            --customer_id 1234 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --session_ids 5678 91011 \
            --working_dir "path/to/output/directory"
"""
import argparse
import os

import numpy as np
import pandas as pd

from vision_api_interface import VisionAPIInterface


def download_object_tracking_data(
    objects_df: pd.DataFrame,
    working_dir: str,
    vision_api_interface: "VisionAPIInterface",
) -> None:
    """Download tracking data for each object.

    Args:
        objects_df: DataFrame of objects.
        working_dir: Directory to save results.
        vision_api_interface: Interface for API calls.
    """
    # Download tracking data for each object:
    tracking_json_dir = os.path.join(working_dir, "tracking_results")
    print(f"Downloading object data to {tracking_json_dir}...")
    os.makedirs(tracking_json_dir, exist_ok=True)

    # Find which objects we already have data for
    existing_object_ids = set()
    if os.path.exists(tracking_json_dir):
        for filename in os.listdir(tracking_json_dir):
            if filename.endswith("_tracking_results.json"):
                object_id = filename.split("_tracking_results.json")[0]
                existing_object_ids.add(object_id)

    # Filter out objects that have already been downloaded
    objects_to_download_df = objects_df[
        ~objects_df["object_id"].isin(existing_object_ids)
    ]

    if not objects_to_download_df.empty:
        print(
            "Found "
            f"{len(objects_to_download_df)}/{len(objects_df)} "
            "objects to download."
        )
        vision_api_interface.download_all_object_tracking_jsons(
            objects_to_download_df, tracking_json_dir
        )
    else:
        print("All object data already downloaded.")


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
    ap.add_argument(
        "--working_dir",
        required=True,
        type=str,
        help="Path to directory to save results",
    )
    ap.add_argument(
        "--object_type",
        type=str,
        default="person",
        choices=["person", "vehicle"],
        help="Type of object to analyze ('person' or 'vehicle')",
    )
    ap.add_argument(
        "--download_tracking_jsons",
        action="store_true",
        help="If set, download object tracking JSONs for each object.",
    )
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    session_ids = args["session_ids"]
    working_dir = args["working_dir"]
    download_jsons = args["download_tracking_jsons"]
    object_type = args["object_type"]

    os.makedirs(working_dir, exist_ok=True)

    # Create an interface to help with the API calls:
    vision_api_interface = VisionAPIInterface(customer_id, api_key, api_url)

    all_objects_df = []
    for session_id in session_ids:
        # Get session results:
        print(f"Retrieving results for session {session_id}...")
        session_result_response = vision_api_interface.get_session_result(session_id, include_appearance_vectors=True)

        # Unpack objects into a Pandas DataFrame:
        (
            objects_df,
            _,
        ) = vision_api_interface.get_session_objects_highlights(
            session_result_response
        )
        print(f"Found {len(objects_df)} objects in session {session_id}.")
        # Add session_id to each object_id to make it unique across sessions:
        if not objects_df.empty:
            objects_df["object_id"] = (
                f"{session_id}_" + objects_df["object_id"].astype(str)
            )
        all_objects_df.append(objects_df)

    # Concatenate all objects into a single DataFrame:
    objects_df = pd.concat(all_objects_df, ignore_index=True)
    # Remove any duplicate objects that might appear across sessions:
    objects_df.drop_duplicates(subset=["object_id"], inplace=True)
    print(f"Found a total of {len(objects_df)} unique objects across all sessions.")

    # Filter objects by the specified type:
    print(f"Filtering for objects of type '{object_type}'...")
    objects_df = objects_df[objects_df["type"] == object_type].copy()
    if objects_df.empty:
        print(f"No objects of type '{object_type}' found. Exiting.")
        return
    print(f"Found {len(objects_df)} objects of type '{object_type}'.")

    # Optionally download tracking data for each object:
    if download_jsons:
        download_object_tracking_data(objects_df, working_dir, vision_api_interface)

    # Load feature vectors from the dataframe:
    print("Loading feature vectors...")
    # Keep only objects that have a feature vector:
    objects_df = objects_df[
        objects_df["appearance_fv"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ]

    if objects_df.empty:
        print("No objects with feature vectors found. Exiting.")
        return

    object_ids = objects_df["object_id"].tolist()
    feature_vectors = objects_df["appearance_fv"].tolist()

    if not feature_vectors:
        print("No feature vectors found. Exiting.")
        return

    # Create a numpy array of feature vectors:
    X = np.array(feature_vectors)

    # L2-normalize the feature vectors to prepare for cosine similarity:
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Calculate the cosine similarity matrix:
    print("Calculating similarity matrix...")
    similarity_matrix = X_normalized @ X_normalized.T

    # For each object, get the top 10 most similar other objects:
    print("\nTop 10 most similar objects for each object:")
    for i, object_id in enumerate(object_ids):
        # Get similarity scores for the current object, excluding itself
        scores = similarity_matrix[i]
        
        # Get the indices that would sort the array in descending order
        sorted_indices = np.argsort(scores)[::-1]

        print(f"\nObject: {object_id}")
        
        # Start from 1 to skip self-similarity
        other_object_count = 0
        for j in sorted_indices[1:]:
            if other_object_count >= 10:
                break
            
            similar_object_id = object_ids[j]
            similarity_score = scores[j]
            print(
                f"\t{other_object_count+1}. Object {similar_object_id} "
                f"(Similarity: {similarity_score:.4f})"
            )
            other_object_count += 1
        
        if other_object_count == 0:
            print("\tNo other similar objects found.")


if __name__ == "__main__":
    main()
