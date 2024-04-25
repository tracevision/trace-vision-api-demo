#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Claire Roberts-Thomson
#
"""
Python interface for interacting with the Vision API.

Note that _split_video() use the command line utility `split`, which assumes a
Linux environment.
"""
import datetime
import json
import os
import subprocess

import requests
import pandas as pd


class VisionAPIInterface:
    """
    Python interface for interacting with the Vision API.
    """

    def __init__(self, customer_id, api_key, api_url):
        """
        Initialize VisionAPIInterface.

        :param customer_id: Customer (division) ID
        :param api_key: API key
        :param api_url: API URL
        """
        self.customer_id = customer_id
        self.api_key = api_key
        self.api_url = api_url
        # Set up an API session:
        self.api_session = requests.Session()
        # Set the customer token:
        self.customer_token = {
            "customer_id": self.customer_id,
            "token": self.api_key,
        }

    @staticmethod
    def check_response(response):
        """
        Check an API response for errors.

        *   Raise a ValueError if the response status code is not 200 or if the
            response is null.
        *   Print all errors in the response if there are any.

        :param response: Response from the API
        """
        if response.status_code != 200:
            raise ValueError("Received status code other than 200")
        response_text = json.loads(response.text)
        if response_text is None:
            raise ValueError(
                "Received null response from the API. Check the API URL, customer ID, and API key"
            )
        elif "errors" in response_text:
            print("Errors received from the API:")
            for cur_error in response_text["errors"]:
                print(cur_error["message"])

    def get_all_available_vision_sessions(self):
        """
        Get all available vision sessions.

        Query the graphQL API to get all available vision sessions for the
        customer ID. Check the response for any errors.

        :return: Response from the API
        """
        # Get all available vision sessions:
        print(
            f"Querying all available sessions for customer {self.customer_id}"
        )
        get_sessions_query = """
        query sessions($token: CustomerToken!, $limit: Int, $offset: Int) {
            sessions(token: $token, limit: $limit, offset: $offset) {
                session_id
                type
                status
                videos {
                    videos {
                        video_id
                        name
                        start_time
                        duration
                        uploaded_time
                        width
                        height
                        fps
                    }
                }
            }
        }
        """
        variables = {"token": self.customer_token}
        response = self.api_session.post(
            self.api_url,
            json={"query": get_sessions_query, "variables": variables},
        )
        print(f"Done querying all available sessions")
        self.check_response(response)
        return response

    def create_new_session(self, session_input):
        """
        Create a new session.

        :param session_input: Dict containing session metadata for creating a
            session. Expected format is like:
            {
                "type": "soccer_game",
                "game_info": {
                    "home_team": {
                        "name": "Red team",
                        "score": 0,
                        "color": "#ff0000",
                    },
                    "away_team": {
                        "name": "Blue team",
                        "score": 0,
                        "color": "#0000ff"
                    },
                    "start_time": "2024-01-01T12:00:00Z",
                },
                "capabilities": ["tracking", "highlights"],
            }
        :return response: Response from the API
        """
        print(
            f"Sending mutation request to create new session for customer {self.customer_id}"
        )
        create_session_mutation = """
            mutation createSession($token: CustomerToken!, $sessionData: SessionInput!) {
                createSession(token: $token, sessionData: $sessionData) {
                    success
                    error
                    session {
                        session_id
                        type
                        status
                        videos {
                            videos {
                                video_id
                                name
                                start_time
                                duration
                                uploaded_time
                                width
                                height
                                fps
                            }
                        }
                    }
                }
            }
            """
        variables = {
            "token": self.customer_token,
            "sessionData": session_input,
        }
        response = self.api_session.post(
            self.api_url,
            json={
                "query": create_session_mutation,
                "variables": variables,
            },
        )
        print(f"Done sending mutation request to create new session")
        self.check_response(response)
        return response

    @staticmethod
    def get_session_id(create_session_response):
        """
        Get the session ID from the API response to creating a session.

        :param create_session_response: Response from the API, e.g. returned by
            method create_new_session().
        :return session_id: Session ID
        """
        create_session_text = json.loads(create_session_response.text)
        session_id = create_session_text["data"]["createSession"]["session"][
            "session_id"
        ]
        return session_id

    def _put_video_to_s3(self, upload_url, video_filepath):
        """
        Upload a video to an AWS S3 URL using a PUT request.

        :param upload_url: URL to upload video to AWS S3
        :param video_filepath: Local path to video file
        :return response: Response from the API
        """
        print(f"Uploading video to s3 using url {upload_url}")
        # Upload video to upload_url:
        headers = {"Content-Type": "video/mp4"}
        response = self.api_session.put(
            upload_url, data=open(video_filepath, "rb"), headers=headers
        )
        print("Done uploading video to s3")
        return response

    def upload_video(self, session_id, session_input, video_filepath):
        """
        Upload a video to a session.

        :param session_id: Session ID to upload the video to
        :param session_input: Dict containing session metadata for creating a
            session. Expected format is like:
            {
                "type": "soccer_game",
                "game_info": {
                    "home_team": {
                        "name": "Red team",
                        "score": 0,
                        "color": "#ff0000",
                    },
                    "away_team": {
                        "name": "Blue team",
                        "score": 0,
                        "color": "#0000ff"
                    },
                    "start_time": "2024-01-01T12:00:00Z",
                },
                "capabilities": ["tracking", "highlights"],
            }
        :param video_filepath: Local path to video file
        :return response: Response from the Vision API
        :return response_put: Response from the PUT request to upload the video
        """
        print(f"Requesting mutation to upload video to session {session_id}")
        # Upload video:
        upload_video_mutation = """
            mutation uploadVideo($token: CustomerToken!, $session_id: Int!, $video_name: String!, $start_time: DateTime) {
                uploadVideo(token: $token, session_id: $session_id, video_name: $video_name, start_time: $start_time) {
                    success
                    error
                    upload_url
                }
            }
            """
        variables = {
            "token": self.customer_token,
            "session_id": session_id,
            "video_name": os.path.basename(video_filepath),
            "start_time": session_input["game_info"]["start_time"],
        }
        response = self.api_session.post(
            self.api_url,
            json={
                "query": upload_video_mutation,
                "variables": variables,
            },
        )
        print(f"Done requesting mutation to upload video")
        self.check_response(response)
        # Get the URL for uploading the video to s3:
        response_text = json.loads(response.text)
        upload_url = response_text["data"]["uploadVideo"]["upload_url"]
        # Upload video to upload_url:
        response_put = self._put_video_to_s3(upload_url, video_filepath)
        return response, response_put

    @staticmethod
    def _split_video(video_filepath, n_parts):
        """
        Split a video into n parts.

        Note that this method uses the command line utility `split`, which
        assumes a Linux environment.

        :param video_filepath: Path to video file
        :param n_parts: Number of parts to split the video into
        :return video_fileparts: List of paths to video file parts
        """
        print(f"Splitting video file into {n_parts} parts")
        # Split video into n_parts using linux split command:
        cmd = f"split -n '{n_parts}' '{video_filepath}' '{video_filepath}.part'"
        subprocess.run(cmd, shell=True)
        print(f"Done splitting video file")
        # Find the video file parts:
        video_dirname = os.path.dirname(video_filepath)
        video_name = os.path.basename(video_filepath)
        video_fileparts = []
        for cur_file in sorted(os.listdir(video_dirname)):
            if cur_file.startswith(f"{video_name}.part"):
                video_fileparts.append(os.path.join(video_dirname, cur_file))
        print(f"Video filenames:\n{video_fileparts}")
        return video_fileparts

    def upload_video_multipart(
        self, session_id, session_input, video_filepath, n_parts
    ):
        """
        Upload a video to AWS S3 using multipart upload.

        Note that splitting the video into multiple parts uses the command line
        utility `split`, which assumes a Linux environment.

        :param session_id: Session ID to upload the video to
        :param session_input: Dict containing session metadata for creating a
            session. Expected format is like:
            {
                "type": "soccer_game",
                "game_info": {
                    "home_team": {
                        "name": "Red team",
                        "score": 0,
                        "color": "#ff0000",
                    },
                    "away_team": {
                        "name": "Blue team",
                        "score": 0,
                        "color": "#0000ff"
                    },
                    "start_time": "2024-01-01T12:00:00Z",
                },
                "capabilities": ["tracking", "highlights"],
            }
        :param video_filepath: Local path to video file
        :param n_parts: Number of parts to split the video into
        :return response_1: Response from the Vision API
        :return upload_responses: List of responses from the PUT requests to
            upload the video parts
        :return response_2: Response from the Vision API
        """
        # Split video into n_parts:
        video_fileparts = self._split_video(video_filepath, n_parts)
        # Get URLs for uploading video fileparts:
        print(
            f"Requesting mutation to upload multipart video to session {session_id}"
        )
        multipart_upload_video_mutation = """
            mutation multipartUploadVideo($token: CustomerToken!, $session_id: Int!, $video_name: String!, $start_time: DateTime, $total_parts: Int!) {
                multipartUploadVideo(token: $token, session_id: $session_id, video_name: $video_name, start_time: $start_time, total_parts: $total_parts) {
                    success
                    error
                    upload_id
                    upload_parts {
                        upload_url
                        part
                    }
                }
            }
            """
        variables = {
            "token": self.customer_token,
            "session_id": session_id,
            "video_name": os.path.basename(video_filepath),
            "start_time": session_input["game_info"]["start_time"],
            "total_parts": n_parts,
        }
        response_1 = self.api_session.post(
            self.api_url,
            json={
                "query": multipart_upload_video_mutation,
                "variables": variables,
            },
        )
        print(f"Done requesting mutation to upload multipart video")
        self.check_response(response_1)
        response_1_text = json.loads(response_1.text)
        upload_metadata = response_1_text["data"]["multipartUploadVideo"][
            "upload_parts"
        ]
        # Upload video fileparts:
        upload_responses = []
        for i, (video_filepart_path, cur_upload_metadata) in enumerate(
            zip(video_fileparts, upload_metadata)
        ):
            print(f"Uploading video part {i+1} of {n_parts}")
            upload_url = cur_upload_metadata["upload_url"]
            # Upload video to upload_url:
            cur_response = self._put_video_to_s3(
                upload_url, video_filepart_path
            )
            upload_responses.append(cur_response)
            print(f"Done uploading video part {i+1} of {n_parts}")
        # get the etags from the headers:
        etags = [r.headers["ETag"] for r in upload_responses]
        # Complete the multipart upload:
        print(
            f"Requesting mutation to complete multipart video upload to session {session_id}"
        )
        complete_multipart_upload_mutation = """
            mutation multipartUploadVideoComplete($token: CustomerToken!, $session_id: Int!, $upload_id: String!, $parts_info: [UploadVideoPartInput!]!) {
                multipartUploadVideoComplete(token: $token, session_id: $session_id, upload_id: $upload_id, parts_info: $parts_info) {
                    success
                    error
                }
            }
            """
        variables = {
            "token": self.customer_token,
            "session_id": session_id,
            "upload_id": response_1_text["data"]["multipartUploadVideo"][
                "upload_id"
            ],
            "parts_info": [
                {"part": i + 1, "etag": etag} for i, etag in enumerate(etags)
            ],
        }
        response_2 = self.api_session.post(
            self.api_url,
            json={
                "query": complete_multipart_upload_mutation,
                "variables": variables,
            },
        )
        print(f"Done requesting mutation to complete multipart video upload")
        self.check_response(response_2)
        return response_1, upload_responses, response_2

    def get_session(self, session_id):
        """
        Get a specific vision session.

        Query the graphQL API to get a specific vision session for the customer
        ID. Check the response for any errors.

        :param session_id: Session ID
        :return response: Response from the API
        """
        # Get a specific session by ID:
        print(f"Querying session {session_id} for customer {self.customer_id}")
        get_session_query = """
        query session($token: CustomerToken!, $session_id: Int!) {
            session(token: $token, session_id: $session_id) {
                session_id
                type
                status
                videos {
                    videos {
                        video_id
                        name
                        start_time
                        duration
                        uploaded_time
                        width
                        height
                        fps
                    }
                }
            }
        }
        """
        variables = {"token": self.customer_token, "session_id": session_id}
        response = self.api_session.post(
            self.api_url,
            json={"query": get_session_query, "variables": variables},
        )
        print(f"Done querying session {session_id}")
        self.check_response(response)
        return response

    @staticmethod
    def get_session_status(session_response):
        """
        Get the session processing status from the session API response.

        :param session_response: Response from the API, e.g. from method
            get_session
        :return session_status: Session processing status
        """
        session_text = json.loads(session_response.text)
        session_status = session_text["data"]["session"]["status"]
        return session_status

    @staticmethod
    def write_response_to_json(response, filename):
        """
        Write a response to JSON file

        :param response: Response to write to JSON file
        :param filename: Path to JSON file
        """
        # Ensure output directory exists:
        file_dir = os.path.dirname(filename)
        os.makedirs(file_dir, exist_ok=True)
        # Load response into dict:
        response_text = json.loads(response.text)
        # Write response to file:
        with open(filename, "w") as f:
            json.dump(response_text, f, indent=4)

    def get_session_result(self, session_id):
        """
        Get the result of a vision session.

        Query the graphQL API to get the result of a vision session. Check the
        response for any errors.

        :param session_id: Session ID
        :return response: Response from the API
        """
        # Get session results
        print(
            f"Querying session {session_id} result for customer {self.customer_id}"
        )
        get_session_query = """
        query sessionResult($token: CustomerToken!, $session_id: Int!) {
            sessionResult(token: $token, session_id: $session_id) {
                objects {
                    object_id
                    type
                    side
                    tracking_url
                }
                highlights{
                    highlight_id
                    video_id
                    start_offset
                    duration
                    tags
                    video_stream
                    objects {
                        object_id
                        type
                        side
                        tracking_url
                    }
                }
            }
        }
        """
        variables = {"token": self.customer_token, "session_id": session_id}
        response = self.api_session.post(
            self.api_url,
            json={"query": get_session_query, "variables": variables},
        )
        print(f"Done querying session {session_id}")
        self.check_response(response)
        return response

    @staticmethod
    def get_session_objects_highlights(session_result_response):
        """
        Get objects and highlights from the session API response.

        :param session_result_response: Response from the API, e.g. from method
            get_session_result
        :return objects_df: DataFrame containing objects from the session
        :return highlights_df: DataFrame containing highlights from the session
        """
        session_response_text = json.loads(session_result_response.text)
        objects_df = pd.DataFrame(
            session_response_text["data"]["sessionResult"]["objects"]
        )
        highlights_df = pd.DataFrame(
            session_response_text["data"]["sessionResult"]["highlights"]
        )
        return objects_df, highlights_df

    @staticmethod
    def list_sessions(get_sessions_response):
        """
        Print all available sessions to the command line.

        :param get_sessions_response: Response from the API, e.g. from method
            get_all_available_vision_sessions
        """
        # Get list of available sessions:
        get_sessions_response_text = json.loads(get_sessions_response.text)
        session_list = get_sessions_response_text["data"]["sessions"]
        print("Available sessions:")
        for cur_session in session_list:
            print(cur_session)

    @staticmethod
    def download_file_from_url(url, filename):
        """
        Download a file from a URL.

        :param url: URL to download the file from
        :param filename: Local path to save the file to
        """
        # Get the data from the URL
        response = requests.get(url)
        # Check the request was successful
        if response.status_code == 200:
            # Write the response to file
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Saved data to file {filename}")
        else:
            print(
                f"Error downloading data to file. Error: {response.status_code}"
            )

    def download_all_object_tracking_jsons(self, objects_df, out_dir):
        """
        Download all object tracking JSONs from the Vision API.

        :param objects_df: DataFrame containing objects from the session
        :param out_dir: Directory to save the JSONs to
        :return json_filenames: Dict containing paths to downloaded JSONs,
            with format
            {
                object_id: path to json file
            }
        """
        # Ensure output directory exists:
        os.makedirs(out_dir, exist_ok=True)
        json_filenames = {}
        for _, row in objects_df.iterrows():
            # Unpack object metadata:
            cur_id = row["object_id"]
            cur_tracking_url = row["tracking_url"]
            print(f"Downloading tracking data for {cur_id}")
            cur_filename = os.path.join(
                out_dir, f"{cur_id}_tracking_results.json"
            )
            # Download tracking data for current object (player):
            self.download_file_from_url(cur_tracking_url, cur_filename)
            json_filenames[cur_id] = cur_filename
        return json_filenames

    @staticmethod
    def get_video_start_time(session_response):
        """
        Get the start time of the video from the session API response.

        :param session_response: Response from the API, e.g. from method
            get_session
        :return video_start_time_ms: Start time of the video in milliseconds
        """
        session_text = json.loads(session_response.text)
        # Get the video start time from the session metadata:
        video_start_time_str = session_text["data"]["session"]["videos"][
            "videos"
        ][0]["start_time"]
        # Convert the video time string to UTC epoch milliseconds:
        video_start_time_dt = datetime.datetime.strptime(
            video_start_time_str, "%Y-%m-%dT%H:%M:%SZ"
        )
        video_start_time_ms = int(
            (
                video_start_time_dt - datetime.datetime(1970, 1, 1)
            ).total_seconds()
            * 1000
        )
        return video_start_time_ms
