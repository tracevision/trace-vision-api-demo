#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Claire Roberts-Thomson
#
"""
Retrieve and use results from a Trace Vision session using the GraphQL API.

This script assumes that you have a processed session ready to go. Use
create_trace_vision_session.py to first create a session and upload a video.

Demonstrate:
- Find all existing sessions
- Retrieve session processing status
- Retrieve results from a session
- Use results to overlay tracking data on video
- Use results to create a highlight clip

Since soccer games are supported, we'll be using a session with soccer game
footage.

Usage:
1.  You will need a customer ID and API key to use this script. Contact us to
    get these. We will also share the API URL.
2.  You will need the ID for the session that you created previously, as well
    as the video file you uploaded to that session.
3.  Run the script using the command line:
        python use_trace_vision_session_results.py \
            --customer_id 1234 \
            --api_key "your_api_key" \
            --api_url "api_url" \
            --video_filepath "path/to/video.mp4" \
            --out_dir "path/to/output/directory"
    *   Note that if you omit the output directory, results will be saved in
        the same directory as the video file

You may want to run this script in a debugger to step through the code and
examine API responses.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from vision_api_interface import VisionAPIInterface


def load_tracking_data(filepath, video_start_time_ms):
    """
    Load tracking data from JSON file into a Pandas DataFrame.

    :param filepath: Path to JSON file containing tracking data
    :param video_start_time_ms: Video start time in milliseconds
    :returns tracking_df: DataFrame containing tracking data, with columns
        "time_off": time offset after the video start time, in milliseconds
        "x": horizontal coordinate of the track bounding box center, normalized
            coordinates
        "y": vertical coordinate of the track bounding box center, normalized
            coordinates
        "w": width of the track bounding box, normalized coordinates
        "h": height of the track bounding box, normalized coordinates
        "utc_time_ms": UTC time in milliseconds
        "video_time_ms": Video time in milliseconds
    Note that the normalized coordinates are in the range [0, 1000] relative to
    the original video frame height and width.
    """
    with open(filepath, "r") as jf:
        tracking_data_obj = json.load(jf)
    utc_start_ms = tracking_data_obj["utc_start"]
    tracking_df = pd.DataFrame(
        tracking_data_obj["spotlights"],
        columns=["time_off", "x", "y", "w", "h"],
    )
    # Add UTC time in ms for each data point:
    tracking_df["utc_time_ms"] = utc_start_ms + tracking_df["time_off"]
    # Add video time in ms for each data point:
    tracking_df["video_time_ms"] = (
        tracking_df["utc_time_ms"] - video_start_time_ms
    )
    return tracking_df


def add_bbox_overlay_to_frame(
    frame, video_time_ms, tracking_df, object_id, w, h, overlay_tolerance
):
    """
    Add bounding box overlay to a frame.

    :param frame: Frame to add bounding box overlay to
    :param video_time_ms: Current video time in milliseconds
    :param tracking_df: DataFrame containing tracking data
    :param object_id: ID of the object to add overlay for
    :param w: Frame width
    :param h: Frame height
    :param overlay_tolerance: Tolerance for matching tracking data to video
        time
    :returns frame: Frame with bounding box overlay
    """
    # Find the tracking data point closest to the current video time:
    cur_index = (tracking_df["video_time_ms"] - video_time_ms).abs().idxmin()
    cur_track_time = tracking_df.iloc[cur_index]["video_time_ms"]
    if np.abs(cur_track_time - video_time_ms) <= overlay_tolerance:
        # Convert normalized coordinates to pixel coordinates:
        cur_x = tracking_df.loc[cur_index]["x"] * w / 1000
        cur_y = tracking_df.loc[cur_index]["y"] * h / 1000
        cur_w = tracking_df.loc[cur_index]["w"] * w / 1000
        cur_h = tracking_df.loc[cur_index]["h"] * h / 1000
        # Convert bounding box from x, y, w, h format to u1, v1, u2, v2 format:
        u1 = int(cur_x - cur_w / 2)
        v1 = int(cur_y - cur_h / 2)
        u2 = int(cur_x + cur_w / 2)
        v2 = int(cur_y + cur_h / 2)
        # Draw bounding box on frame:
        cv2.rectangle(frame, (u1, v1), (u2, v2), (0, 255, 0), 2)
        if object_id is not None:
            cv2.putText(
                frame,
                object_id,
                (u1, v1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    return frame


def create_video_with_tracking_overlay(
    video_time_min_ms,
    video_time_max_ms,
    dict_tracking_df,
    text_str,
    video_filepath,
    out_video_filepath,
):
    """
    Create a video with tracking overlay for a specified time range.

    :param video_time_min_ms: Start time of the video range to create, in
        milliseconds
    :param video_time_max_ms: End time of the video range to create, in
        milliseconds
    :param dict_tracking_df: Dictionary containing tracking data for each object
        to overlay, format
        {
            object_id: tracking_df
        }
    :param text_str: Text to add to the video. If None, no text is added
    :param video_filepath: Path to input video file
    :param out_video_filepath: Path to save output video file
    """
    # Find relevant tracking data for the specified time range:
    use_tracking_df = {}
    for object_id in dict_tracking_df:
        mask_df = (
            dict_tracking_df[object_id]["video_time_ms"] >= video_time_min_ms
        ) & (dict_tracking_df[object_id]["video_time_ms"] <= video_time_max_ms)
        if mask_df.sum() == 0:
            print(f"No tracking data found for {object_id} during time range")
            continue
        use_tracking_df[object_id] = (
            dict_tracking_df[object_id].loc[mask_df, :].reset_index(drop=True)
        )
    # Get input video capture and properties:
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Set up output video writer:
    sav = cv2.VideoWriter(
        out_video_filepath,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
        True,
    )
    # For each tracking data point, create a frame with the tracking
    # overlay:
    overlay_tolerance = 1 / fps * 2 * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, video_time_min_ms)
    ret, frm = cap.read()
    cur_video_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    while cur_video_time <= video_time_max_ms:
        # Add bounding box overlay to current frame:
        for object_id, tracking_df in use_tracking_df.items():
            if tracking_df.empty:
                continue
            frm = add_bbox_overlay_to_frame(
                frm,
                cur_video_time,
                tracking_df,
                object_id,
                w,
                h,
                overlay_tolerance,
            )
        if text_str is not None:
            cv2.putText(
                frm,
                text_str,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        # Write frame to output video:
        sav.write(frm)
        # Get the next frame:
        ret, frm = cap.read()
        if not ret:
            break
        cur_video_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    # Close input and output videos:
    sav.release()
    cap.release()


def create_highlight_clips_with_overlay(
    highlights_df,
    video_filepath,
    video_start_time_ms,
    vision_api_interface,
    tracking_json_dir,
    out_dir,
    max_n_highlights=None,
):
    """
    Use OpenCV to create a clip for each highlight with tracking overlay.

    :param highlights_df: DataFrame containing highlight data
    :param video_filepath: Path to video file
    :param video_start_time_ms: Video start time in milliseconds
    :param vision_api_interface: VisionAPIInterface object
    :param tracking_json_dir: Directory containing tracking JSON files
    :param out_dir: Directory to save highlight clips
    :param max_n_highlights: Maximum number of highlights to create. If None,
        creates a clip for each highlight.
    """
    # Ensure output directory exists:
    os.makedirs(out_dir, exist_ok=True)
    # Create a clip for each highlight:
    n_highlights = len(highlights_df)
    if max_n_highlights is not None:
        n_highlights = min(len(highlights_df), max_n_highlights)
    for i, cur_highlight in highlights_df.iterrows():
        if max_n_highlights is not None and i >= max_n_highlights:
            break
        print(f"Creating highlight clip {i+1} of {n_highlights}")
        # Unpack highlight metadata:
        start_offset_ms = cur_highlight["start_offset"]
        video_time_min_ms = start_offset_ms
        duration_ms = cur_highlight["duration"]
        video_time_max_ms = video_time_min_ms + duration_ms
        tags = cur_highlight["tags"]
        text_str = " | ".join(tags)
        objects = cur_highlight["objects"]
        tracking_data = {}
        if len(objects) > 0:
            cur_objects_df = pd.DataFrame(objects)
            players_str = ", ".join(cur_objects_df["object_id"])
            text_str += f" | {players_str}"
            # Download tracking data for each object:
            object_tracking_json_filenames = (
                vision_api_interface.download_all_object_tracking_jsons(
                    cur_objects_df, tracking_json_dir
                )
            )
            for cur_object_id in object_tracking_json_filenames:
                tracking_data[cur_object_id] = load_tracking_data(
                    object_tracking_json_filenames[cur_object_id],
                    video_start_time_ms,
                )
        # Set output filepath:
        out_filepath = os.path.join(out_dir, f"highlight_{i}_overlay.mp4")
        create_video_with_tracking_overlay(
            video_time_min_ms,
            video_time_max_ms,
            tracking_data,
            text_str,
            video_filepath,
            out_filepath,
        )


def create_tracking_sample_videos(
    objects_df,
    object_tracking_json_filenames,
    video_filepath,
    video_start_time_ms,
    out_dir,
    sample_video_duration_s=10,
    max_n_sample_videos=None,
):
    """
    Create sample videos showing tracking overlay.

    Create one sample video for each detected object (person), showing the
    bounding box associated with the track for the specified sample video
    duration. If max_n_sample_videos is specified, only create that many sample
    videos.

    :param objects_df: DataFrame containing object metadata
    :param object_tracking_json_filenames: List of paths to JSON files
        containing tracking data for each object
    :param video_filepath: Path to input video file
    :param video_start_time_ms: Video start time in milliseconds
    :param out_dir: Directory to save sample videos
    :param sample_video_duration_s: Duration of each sample video, in seconds
    :param max_n_sample_videos: Maximum number of sample videos to create. If
        None, creates a sample video for each object.
    """
    # Ensure output directory exists:
    os.makedirs(out_dir, exist_ok=True)
    for i, row in objects_df.iterrows():
        if max_n_sample_videos is not None and i >= max_n_sample_videos:
            break
        # Unpack object metadata:
        cur_id = row["object_id"]
        if object_tracking_json_filenames is None:
            cur_filename = os.path.join(
                os.path.dirname(out_dir),
                "tracking_results",
                f"{cur_id}_tracking_results.json",
            )
        else:
            cur_filename = object_tracking_json_filenames[cur_id]
        print(f"Creating tracking overlay sample for {cur_id}")
        # Load tracking data:
        cur_tracking_df = load_tracking_data(cur_filename, video_start_time_ms)
        # Set up inputs to create_video_with_tracking_overlay:
        video_time_min_ms = cur_tracking_df["video_time_ms"].min()
        video_time_max_ms = video_time_min_ms + sample_video_duration_s * 1000
        dict_tracking_df = {cur_id: cur_tracking_df}
        out_video_filepath = os.path.join(
            out_dir, f"{cur_id}_tracking_sample.mp4"
        )
        text_str = f"{cur_id}"
        create_video_with_tracking_overlay(
            video_time_min_ms,
            video_time_max_ms,
            dict_tracking_df,
            text_str,
            video_filepath,
            out_video_filepath,
        )


def main():
    """
    Main function to retrieve and use results from a Trace Vision session.
    """
    # Define command line arguments:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer_id", required=True, type=int, help="Customer (division) ID"
    )
    ap.add_argument("--api_key", required=True, type=str, help="API key")
    ap.add_argument("--api_url", required=True, type=str, help="API URL")
    ap.add_argument("--session_id", required=True, type=int, help="Session ID")
    ap.add_argument(
        "--video_filepath", required=True, type=str, help="Path to video file"
    )
    ap.add_argument(
        "--out_dir",
        required=False,
        type=str,
        help="Path to directory to save results",
        default=None,
    )
    args = vars(ap.parse_args())

    # Unpack arguments:
    customer_id = args["customer_id"]
    api_key = args["api_key"]
    api_url = args["api_url"]
    session_id = args["session_id"]
    video_filepath = args["video_filepath"]
    out_dir = args["out_dir"]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(video_filepath), "results")

    # Create an interface to help with the API calls:
    vision_api_interface = VisionAPIInterface(customer_id, api_key, api_url)

    # Get all available vision sessions:
    get_sessions_response = (
        vision_api_interface.get_all_available_vision_sessions()
    )
    vision_api_interface.list_sessions(get_sessions_response)

    # Get session status for specified ID:
    session_response = vision_api_interface.get_session(session_id)
    session_status = vision_api_interface.get_session_status(session_response)
    print(f"Session {session_id} status: {session_status}")

    # Get session results:
    session_result_response = vision_api_interface.get_session_result(
        session_id
    )

    # Write session result response data out to JSON file:
    vision_api_interface.write_response_to_json(
        session_result_response, os.path.join(out_dir, "session_result.json")
    )

    # Unpack objects and highlights into Pandas DataFrames:
    (
        objects_df,
        highlights_df,
    ) = vision_api_interface.get_session_objects_highlights(
        session_result_response
    )

    # Write objects and highlights out to CSV files:
    os.makedirs(out_dir, exist_ok=True)
    objects_df.to_csv(os.path.join(out_dir, "objects.csv"), index=False)
    highlights_df.to_csv(os.path.join(out_dir, "highlights.csv"), index=False)

    # Download tracking data for each object:
    tracking_json_dir = os.path.join(out_dir, "tracking_results")
    # object_tracking_json_filenames = None
    object_tracking_json_filenames = (
        vision_api_interface.download_all_object_tracking_jsons(
            objects_df, tracking_json_dir
        )
    )

    # Get the video start time:
    video_start_time_ms = vision_api_interface.get_video_start_time(
        session_response
    )

    # Overlay tracking data on video:
    tracking_samples_dir = os.path.join(out_dir, "tracking_samples")
    create_tracking_sample_videos(
        objects_df,
        object_tracking_json_filenames,
        video_filepath,
        video_start_time_ms,
        tracking_samples_dir,
        sample_video_duration_s=10,
        max_n_sample_videos=10,
    )

    # Create highlight clips with overlay:
    highlights_overlay_dir = os.path.join(out_dir, "highlights_overlay")
    create_highlight_clips_with_overlay(
        highlights_df,
        video_filepath,
        video_start_time_ms,
        vision_api_interface,
        tracking_json_dir,
        highlights_overlay_dir,
        max_n_highlights=10,
    )


if __name__ == "__main__":
    main()
