
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
import glob
import os
import multiprocessing
import collections
import json
from datetime import datetime
import subprocess
import shutil

import numpy as np
import pandas as pd
import cv2

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


def _process_session_for_tkinter(args):
    """
    Processes a single session to create its tkinter resources. This function is
    designed to be called by a multiprocessing pool.
    """
    (
        session_id,
        session_object_ids,
        working_dir,
        all_object_ids,
        cosine_similarity_matrix,
        video_output_size,
        video_path,
    ) = args

    print(f"Processing session {session_id} with {len(session_object_ids)} objects...")

    # Open video
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"  Could not open video {video_path}, skipping session {session_id}.")
        return

    # Get video properties
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_object_ids_list = list(all_object_ids)

    # Prepare data structures for all objects in the session
    object_data = {}
    for object_id in session_object_ids:
        tracking_json_path = os.path.join(
            working_dir, "tracking_results", f"{object_id}_tracking_results.json"
        )
        if not os.path.exists(tracking_json_path):
            print(f"  No tracking data for object {object_id}, skipping.")
            continue

        with open(tracking_json_path, "r") as f:
            tracking_data = json.load(f)

        if "bboxes" not in tracking_data or not tracking_data["bboxes"]:
            print(f"  No bboxes data for object {object_id}, skipping.")
            continue

        bboxes = sorted(
            [bbox for bbox in tracking_data["bboxes"] if len(bbox) == 5],
            key=lambda x: x[0],
        )

        if not bboxes:
            print(
                "  Bboxes list is empty or only contains predictions for object"
                f" {object_id}, skipping."
            )
            continue

        # Pre-calculate which bbox has the median area to save as the thumbnail
        # to avoid storing all thumbnail crops in memory.
        areas = [(bbox[3] - bbox[1]) * (bbox[4] - bbox[2]) for bbox in bboxes]
        median_bbox_idx = (
            np.argmin(np.abs(np.array(areas) - np.median(areas))) if areas else -1
        )

        object_dir = os.path.join(working_dir, "tkinter_resources", object_id)
        os.makedirs(object_dir, exist_ok=True)

        frames_dir = os.path.join(object_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            object_idx_in_matrix = all_object_ids_list.index(object_id)
            cosine_similarity_row = cosine_similarity_matrix[object_idx_in_matrix]
        except ValueError:
            print(f"  Could not find object {object_id} in similarity matrix, skipping.")
            continue

        object_data[object_id] = {
            "bboxes": bboxes,
            "bbox_idx": 0,
            "median_bbox_idx": median_bbox_idx,
            "thumbnail_to_save": None,
            "bbox_data": [],
            "moving_avg_centers": collections.deque(maxlen=10),
            "moving_avg_heights": collections.deque(maxlen=10),
            "object_dir": object_dir,
            "frames_dir": frames_dir,
            "frame_count": 0,
            "cosine_similarity_row": cosine_similarity_row,
        }

    # Iterate through video frames
    while video_cap.isOpened():
        # Check if all objects in this session have been processed
        all_objects_processed = all(
            data["bbox_idx"] >= len(data["bboxes"]) for data in object_data.values()
        )
        if not object_data or all_objects_processed:
            break

        ret, frame = video_cap.read()
        if not ret:
            break

        current_time_ms = video_cap.get(cv2.CAP_PROP_POS_MSEC)

        for object_id, data in object_data.items():
            # Process all bboxes that are meant for frames up to the current time
            while (
                data["bbox_idx"] < len(data["bboxes"])
                and data["bboxes"][data["bbox_idx"]][0] <= current_time_ms
            ):
                _, u1, v1, u2, v2 = data["bboxes"][data["bbox_idx"]]

                # Ensure bbox coordinates are integers and within frame bounds
                u1, v1, u2, v2 = (
                    int(max(0, u1)),
                    int(max(0, v1)),
                    int(min(frame_width, u2)),
                    int(min(frame_height, v2)),
                )

                if u1 >= u2 or v1 >= v2:
                    data["bbox_idx"] += 1
                    continue

                data["bbox_data"].append((u1, v1, u2, v2))

                # Thumbnail crop
                if data["bbox_idx"] == data["median_bbox_idx"]:
                    thumbnail_crop = frame[v1:v2, u1:u2]
                    if thumbnail_crop.size > 0:
                        # Resize and pad to standard size
                        h, w = thumbnail_crop.shape[:2]
                        target_h, target_w = 256, 128
                        scale = min(target_w / w, target_h / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        resized_crop = cv2.resize(thumbnail_crop, (new_w, new_h))

                        # Pad to target size
                        pad_w = (target_w - new_w) // 2
                        pad_h = (target_h - new_h) // 2
                        padded_crop = cv2.copyMakeBorder(
                            resized_crop,
                            pad_h,
                            target_h - new_h - pad_h,
                            pad_w,
                            target_w - new_w - pad_w,
                            cv2.BORDER_CONSTANT,
                            value=[0, 0, 0],
                        )
                        data["thumbnail_to_save"] = padded_crop

                # Video crop with smoothing
                bbox_height = v2 - v1
                bbox_width = u2 - u1
                center_x, center_y = u1 + bbox_width / 2, v1 + bbox_height / 2

                data["moving_avg_centers"].append((center_x, center_y))
                data["moving_avg_heights"].append(bbox_height)

                smooth_center_x = np.mean([c[0] for c in data["moving_avg_centers"]])
                smooth_center_y = np.mean([c[1] for c in data["moving_avg_centers"]])
                smooth_bbox_height = np.mean(data["moving_avg_heights"])

                crop_size = 2 * smooth_bbox_height

                crop_u1 = int(smooth_center_x - crop_size / 2)
                crop_v1 = int(smooth_center_y - crop_size / 2)
                crop_u2 = int(smooth_center_x + crop_size / 2)
                crop_v2 = int(smooth_center_y + crop_size / 2)

                # Handle padding
                pad_left = max(0, -crop_u1)
                pad_right = max(0, crop_u2 - frame_width)
                pad_top = max(0, -crop_v1)
                pad_bottom = max(0, crop_v2 - frame_height)

                final_crop_u1 = max(0, crop_u1)
                final_crop_v1 = max(0, crop_v1)
                final_crop_u2 = min(frame_width, crop_u2)
                final_crop_v2 = min(frame_height, crop_v2)

                square_crop = frame[
                    final_crop_v1:final_crop_v2, final_crop_u1:final_crop_u2
                ]

                if square_crop.size > 0:
                    padded_crop = cv2.copyMakeBorder(
                        square_crop,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0],
                    )
                    resized_video_crop = cv2.resize(
                        padded_crop, (video_output_size, video_output_size)
                    )

                    # Draw bounding box on the video crop
                    scale = video_output_size / crop_size if crop_size > 0 else 1
                    box_u1 = int((u1 - crop_u1) * scale)
                    box_v1 = int((v1 - crop_v1) * scale)
                    box_u2 = int((u2 - crop_u1) * scale)
                    box_v2 = int((v2 - crop_v1) * scale)
                    cv2.rectangle(
                        resized_video_crop,
                        (box_u1, box_v1),
                        (box_u2, box_v2),
                        (0, 255, 0),
                        2,
                    )
                    frame_path = os.path.join(
                        data["frames_dir"], f"frame_{data['frame_count']:05d}.jpg"
                    )
                    cv2.imwrite(frame_path, resized_video_crop)
                    data["frame_count"] += 1
                data["bbox_idx"] += 1

    video_cap.release()

    # Generate resources for each object
    for object_id, data in object_data.items():
        if not data["bbox_data"]:
            print(f"  Could not extract any crops for object {object_id}, skipping.")
            continue

        # Create and save thumbnail (use median area bbox)
        if data["thumbnail_to_save"] is not None:
            thumbnail_path = os.path.join(data["object_dir"], "thumbnail.jpg")
            cv2.imwrite(thumbnail_path, data["thumbnail_to_save"])

        # Create and save video
        if data["frame_count"] > 0:
            video_path_out = os.path.join(data["object_dir"], "video.mp4")
            frames_pattern = os.path.join(data["frames_dir"], "frame_%05d.jpg")

            # Use ffmpeg to create video
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-r",
                "30",
                "-f",
                "image2",
                "-i",
                frames_pattern,
                "-vcodec",
                "libx264",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                video_path_out,
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ffmpeg error for object {object_id}:")
                print(result.stderr)

            # Clean up frames
            shutil.rmtree(data["frames_dir"])

        # Create and save similarities CSV
        sims = []
        for j, other_object_id in enumerate(all_object_ids):
            if object_id == other_object_id:
                continue
            sims.append((other_object_id, data["cosine_similarity_row"][j]))

        sims.sort(key=lambda x: x[1], reverse=True)
        sim_df = pd.DataFrame(sims, columns=["object_id", "cosine_similarity"])
        csv_path = os.path.join(data["object_dir"], "similarities.csv")
        sim_df.to_csv(csv_path, index=False)


def create_tkinter_resources(
    cosine_similarity_matrix,
    object_ids,
    working_dir,
    video_paths,
    session_ids,
    video_output_size=400,
):
    """
    Creates resources for a Tkinter GUI, including object videos, thumbnails, and similarity data.
    """
    # Remove existing resources if they exist
    tkinter_resources_dir = os.path.join(working_dir, "tkinter_resources")
    shutil.rmtree(tkinter_resources_dir, ignore_errors=True)

    # Create a mapping from session_id to video_path
    session_to_video = dict(zip(session_ids, video_paths))
    
    # Group objects by session
    session_objects = collections.defaultdict(list)
    for object_id in object_ids:
        session_id = int(object_id.split("_")[0])
        session_objects[session_id].append(object_id)
        
    tasks = []
    for session_id, session_object_ids in session_objects.items():
        if session_id not in session_to_video:
            print(f"Skipping session {session_id} as no video path provided.")
            continue
        
        video_path = session_to_video[session_id]
        tasks.append(
            (
                session_id,
                session_object_ids,
                working_dir,
                object_ids,
                cosine_similarity_matrix,
                video_output_size,
                video_path,
            )
        )

    with multiprocessing.Pool(4) as pool:
        pool.map(_process_session_for_tkinter, tasks)

    print("\nFinished creating all tkinter resources.")


def filter_out_stationary_objects(
    objects_df: pd.DataFrame, working_dir: str, distance_threshold: float = 2.0
) -> pd.DataFrame:
    """Filter out nearly stationary objects from the DataFrame.

    This function iterates through each object, loads its tracking data,
    and calculates if the object has moved significantly. If not, it is
    marked for removal. The logic is based on maximum displacement
    normalized by the object's mean width.

    Args:
        objects_df: DataFrame containing the objects to filter.
        working_dir: The directory where tracking data is stored.
        distance_threshold: The threshold for movement, normalized by object
                            width. Objects moving less than this are removed.

    Returns:
        A DataFrame with stationary objects filtered out.
    """
    objects_to_remove = []
    tracking_json_dir = os.path.join(working_dir, "tracking_results")

    if not os.path.exists(tracking_json_dir):
        print(
            "Warning: Tracking results directory not found: "
            f"{tracking_json_dir}. Cannot filter stationary objects."
        )
        return objects_df

    print(f"\nFiltering stationary objects with threshold: {distance_threshold}...")

    for object_id in objects_df["object_id"]:
        if object_id == "3716908_442455105-210":
            print(f"\n--- Debugging object {object_id} ---")
        tracking_json_path = os.path.join(
            tracking_json_dir, f"{object_id}_tracking_results.json"
        )

        with open(tracking_json_path, "r") as f:
            tracking_data = json.load(f)

        # Use a DataFrame for easier manipulation of bounding box data
        bboxes_df = pd.DataFrame(
            tracking_data["bboxes"], columns=["time", "u1", "v1", "u2", "v2"]
        )

        if object_id == "3716908_442455105-210":
            print(f"Initial bbox count: {len(bboxes_df)}")

        # Quantize time to handle multiple detections per second and keep unique entries
        bboxes_df["time_quantized"] = bboxes_df["time"] // 2000
        # Instead of dropping duplicates, group by the quantized time and find the median of the bounding box coordinates.
        agg_cols = {"u1": "median", "v1": "median", "u2": "median", "v2": "median"}
        bboxes_df = bboxes_df.groupby("time_quantized").agg(agg_cols).reset_index()

        if object_id == "3716908_442455105-210":
            print(f"Bbox count after quantizing and aggregation: {len(bboxes_df)}")
            print(bboxes_df)

        # Calculate bounding box properties
        bboxes_df["width"] = bboxes_df["u2"] - bboxes_df["u1"]
        bboxes_df["u_snap"] = bboxes_df["u1"] + bboxes_df["width"] / 2
        bboxes_df["v_snap"] = bboxes_df["v1"] + (bboxes_df["v2"] - bboxes_df["v1"]) / 2

        u_snaps = bboxes_df["u_snap"].to_numpy()
        v_snaps = bboxes_df["v_snap"].to_numpy()

        # Calculate pairwise distances between all points
        u_snaps_distances = u_snaps.reshape(-1, 1) - u_snaps.reshape(1, -1)
        v_snaps_distances = v_snaps.reshape(-1, 1) - v_snaps.reshape(1, -1)
        distances = np.sqrt(u_snaps_distances**2 + v_snaps_distances**2)

        mean_width = bboxes_df["width"].mean()

        if object_id == "3716908_442455105-210":
            print(f"Max distance (pixels): {distances.max()}")
            print(f"Mean width (pixels): {mean_width}")

        if mean_width > 0:
            normalized_distances = distances / mean_width
            if object_id == "3716908_442455105-210":
                print(f"Max normalized distance: {normalized_distances.max()}")
                print(f"Distance threshold: {distance_threshold}")
                if normalized_distances.max() < distance_threshold:
                    print("--> Object classified as STATIONARY")
                else:
                    print("--> Object classified as NOT STATIONARY")

            if normalized_distances.max() < distance_threshold:
                objects_to_remove.append(object_id)
        else:
            # If mean_width is 0, the object is likely invalid or stationary
            if object_id == "3716908_442455105-210":
                print("--> Object has mean_width <= 0, classified as STATIONARY")
            objects_to_remove.append(object_id)

    if objects_to_remove:
        print(f"Found and removed {len(objects_to_remove)} stationary objects.")
        objects_df = objects_df[~objects_df["object_id"].isin(objects_to_remove)]

    return objects_df


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
        "--filter_stationary_objects",
        action="store_true",
        help="If set, filter out stationary objects.",
    )
    ap.add_argument(
        "--create_tkinter_resources",
        action="store_true",
        help="If set, download object tracking JSONs and create tkinter resources for each object.",
    )
    ap.add_argument(
        "--video_paths",
        type=str,
        nargs="+",
        help="Paths to video files corresponding to each session ID (required when --create_tkinter_resources is used)",
    )
    ap.add_argument(
        "--video_output_size",
        type=int,
        default=400,
        help="Output size for video crops in pixels (default: 400)",
    )
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    session_ids = args["session_ids"]
    working_dir = args["working_dir"]
    create_tkinter = args["create_tkinter_resources"]
    video_paths = args["video_paths"]
    object_type = args["object_type"]
    video_output_size = args["video_output_size"]
    filter_stationary = args["filter_stationary_objects"]

    # Validate video_paths when creating tkinter resources
    if create_tkinter:
        if not video_paths:
            raise ValueError("--video_paths is required when --create_tkinter_resources is used")
        if len(video_paths) != len(session_ids):
            video_paths = glob.glob(video_paths)
            raise ValueError("Number of video paths must match number of session IDs")

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

    # Download tracking data for each object if creating tkinter resources or filtering stationary objects
    if create_tkinter or filter_stationary:
        download_object_tracking_data(objects_df, working_dir, vision_api_interface)

    # Filter out stationary objects if the flag is set
    if filter_stationary:
        objects_df = filter_out_stationary_objects(objects_df, working_dir)
        print(f"Remaining objects after filtering: {len(objects_df)}")

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

    # Optionally create tkinter resources:
    if create_tkinter:
        create_tkinter_resources(similarity_matrix, object_ids, working_dir, video_paths, session_ids, video_output_size)


if __name__ == "__main__":
    main()
