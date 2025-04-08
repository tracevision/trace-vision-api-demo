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
from graphql_query import Argument, Field, Operation, Query, Variable

from vision_api_operations import VisionAPIOperations


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

        self.query_token = Variable(name="token", type="CustomerToken!")
        self.query_limit = Variable(name="limit", type="Int")
        self.query_offset = Variable(name="offset", type="Int")
        self.query_sessionData = Variable(
            name="sessionData", type="SessionCreateInput!"
        )
        self.query_session_id = Variable(name="session_id", type="Int!")
        self.query_video_name = Variable(name="video_name", type="String!")
        self.query_start_time = Variable(name="start_time", type="DateTime")
        self.query_total_parts = Variable(name="total_parts", type="Int!")
        self.query_upload_id = Variable(name="upload_id", type="String!")
        self.query_parts_info = Variable(
            name="parts_info", type="[UploadVideoPartInput!]!"
        )

        self.arg_token = Argument(name="token", value=self.query_token)
        self.arg_limit = Argument(name="limit", value=self.query_limit)
        self.arg_offset = Argument(name="offset", value=self.query_offset)
        self.arg_sessionData = Argument(
            name="sessionData", value=self.query_sessionData
        )
        self.arg_session_id = Argument(name="session_id", value=self.query_session_id)
        self.arg_video_name = Argument(name="video_name", value=self.query_video_name)
        self.arg_start_time = Argument(name="start_time", value=self.query_start_time)
        self.arg_total_parts = Argument(
            name="total_parts", value=self.query_total_parts
        )
        self.arg_upload_id = Argument(name="upload_id", value=self.query_upload_id)
        self.arg_parts_info = Argument(name="parts_info", value=self.query_parts_info)

        self.operations = VisionAPIOperations()

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
        print(f"Querying all available sessions for customer {self.customer_id}")

        # Create Get Sessions Query string
        videos_field = Field(
            name="videos",
            fields=[
                Field(
                    name="videos",
                    fields=[
                        "video_id",
                        "name",
                        "start_time",
                        "duration",
                        "uploaded_time",
                        "width",
                        "height",
                        "fps",
                    ],
                )
            ],
        )

        sessions = Query(
            name="sessions",
            arguments=[self.arg_token, self.arg_limit, self.arg_offset],
            fields=["session_id", "type", "status", videos_field],
        )

        operation = Operation(
            type="query",
            name="sessions",
            variables=[self.query_token, self.query_limit, self.query_offset],
            queries=[sessions],
        )

        # Get Session Query string produced
        get_sessions_query = operation.render()

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

        # Create new session query
        videos_field = Field(
            name="videos",
            fields=[
                Field(
                    name="videos",
                    fields=[
                        "video_id",
                        "name",
                        "start_time",
                        "duration",
                        "uploaded_time",
                        "width",
                        "height",
                        "fps",
                    ],
                )
            ],
        )

        session_field = Field(
            name="session",
            fields=["session_id", "type", "status", videos_field],
        )

        create_session = Query(
            name="createSession",
            arguments=[self.arg_token, self.arg_sessionData],
            fields=["success", "error", session_field],
        )

        operation = Operation(
            type="mutation",
            name="createSession",
            variables=[self.query_token, self.query_sessionData],
            queries=[create_session],
        )

        # Create session string query produced
        create_session_mutation = operation.render()

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
        print("create_session_text:", json.loads(response.text))
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
        print("API Response (create_session_response):", create_session_text)
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

        # Create upload video query
        uploadVideo = Query(
            name="uploadVideo",
            arguments=[
                self.arg_token,
                self.arg_session_id,
                self.arg_video_name,
                self.arg_start_time,
            ],
            fields=[
                "success",
                "error",
                "upload_url",
            ],
        )

        operation = Operation(
            type="mutation",
            name="uploadVideo",
            variables=[
                self.query_token,
                self.query_session_id,
                self.query_video_name,
                self.query_start_time,
            ],
            queries=[uploadVideo],
        )

        # Upload video query produced
        upload_video_mutation = operation.render()

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
        print(f"Requesting mutation to upload multipart video to session {session_id}")

        # Create upload video multipart query
        upload_parts_field = Field(name="upload_parts", fields=["upload_url", "part"])

        multipart_upload_video = Query(
            name="multipartUploadVideo",
            arguments=[
                self.arg_token,
                self.arg_session_id,
                self.arg_video_name,
                self.arg_start_time,
                self.arg_total_parts,
            ],
            fields=["success", "error", "upload_id", upload_parts_field],
        )

        operation = Operation(
            type="mutation",
            name="multipartUploadVideo",
            variables=[
                self.query_token,
                self.query_session_id,
                self.query_video_name,
                self.query_start_time,
                self.query_total_parts,
            ],
            queries=[multipart_upload_video],
        )

        # Upload video multipart string query produced
        multipart_upload_video_mutation = operation.render()

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
            cur_response = self._put_video_to_s3(upload_url, video_filepart_path)
            upload_responses.append(cur_response)
            print(f"Done uploading video part {i+1} of {n_parts}")
        # get the etags from the headers:
        etags = [r.headers["ETag"] for r in upload_responses]
        # Complete the multipart upload:
        print(
            f"Requesting mutation to complete multipart video upload to session {session_id}"
        )

        # Create multipart upload mutation query
        multipart_upload_video_complete = Query(
            name="multipartUploadVideoComplete",
            arguments=[
                self.arg_token,
                self.arg_session_id,
                self.arg_upload_id,
                self.arg_parts_info,
            ],
            fields=["success", "error"],
        )

        operation = Operation(
            type="mutation",
            name="multipartUploadVideoComplete",
            variables=[
                self.query_token,
                self.query_session_id,
                self.query_upload_id,
                self.query_parts_info,
            ],
            queries=[multipart_upload_video_complete],
        )

        # multipart upload mutation query produced
        complete_multipart_upload_mutation = operation.render()

        variables = {
            "token": self.customer_token,
            "session_id": session_id,
            "upload_id": response_1_text["data"]["multipartUploadVideo"]["upload_id"],
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

        # Create get session string query
        videos_field = Field(
            name="videos",
            fields=[
                Field(
                    name="videos",
                    fields=[
                        "video_id",
                        "name",
                        "start_time",
                        "duration",
                        "uploaded_time",
                        "width",
                        "height",
                        "fps",
                    ],
                )
            ],
        )

        session_query = Query(
            name="session",
            arguments=[self.arg_token, self.arg_session_id],
            fields=["session_id", "type", "status", videos_field],
        )

        operation = Operation(
            type="query",
            name="session",
            variables=[self.query_token, self.query_session_id],
            queries=[session_query],
        )

        # Get session string query produced
        get_session_query = operation.render()

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
        print(f"Querying session {session_id} result for customer {self.customer_id}")

        # Create get sessions result string query
        objects_field = Field(
            name="objects",
            fields=["object_id", "type", "side", "tracking_url"],
        )

        highlights_field = Field(
            name="highlights",
            fields=[
                "highlight_id",
                "video_id",
                "start_offset",
                "duration",
                "tags",
                "video_stream",
                objects_field,
            ],
        )

        session_result_query = Query(
            name="sessionResult",
            arguments=[self.arg_token, self.arg_session_id],
            fields=[objects_field, highlights_field],
        )

        operation = Operation(
            type="query",
            name="sessionResult",
            variables=[self.query_token, self.query_session_id],
            queries=[session_result_query],
        )

        # Get session string query produced
        get_session_query = operation.render()

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
            print(f"Error downloading data to file. Error: {response.status_code}")

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
            cur_filename = os.path.join(out_dir, f"{cur_id}_tracking_results.json")
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
        video_start_time_str = session_text["data"]["session"]["videos"]["videos"][0][
            "start_time"
        ]
        # Convert the video time string to UTC epoch milliseconds:
        format_strings = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]
        video_start_time_dt = None
        for cur_format_string in format_strings:
            try:
                video_start_time_dt = datetime.datetime.strptime(
                    video_start_time_str, cur_format_string
                )
                break
            except ValueError:
                continue
        if video_start_time_dt is None:
            raise ValueError(f"Could not parse video start time {video_start_time_str}")
        video_start_time_ms = int(
            (video_start_time_dt - datetime.datetime(1970, 1, 1)).total_seconds() * 1000
        )
        return video_start_time_ms

    def create_new_facility(self, facility_input):
        """
        Create a new facility.

        :param facility_input: Dict containing facility metadata for creating a
            facility. See
            https://api.tracevision.com/graphql/v1/docs/types/FacilityInput
        :return response: Response from the API
        """
        print(
            f"Sending mutation request to create new facility for customer {self.customer_id}"
        )
        variables = {
            "token": self.customer_token,
            "facility": facility_input,
        }
        response = self.api_session.post(
            self.api_url,
            json={
                "query": self.operations.createFacility,
                "variables": variables,
            },
        )
        print(f"Done sending mutation request to create new facility")
        self.check_response(response)
        return response

    def create_new_camera(self, camera_input):
        """
        Create a new camera.

        :param camera_input: Dict containing camera metadata for creating a
            camera. See
            https://api.tracevision.com/graphql/v1/docs/types/CameraInput
        :return response: Response from the API
        """
        print(
            f"Sending mutation request to create new camera for customer {self.customer_id}"
        )
        variables = {
            "token": self.customer_token,
            "camera": camera_input,
        }
        response = self.api_session.post(
            self.api_url,
            json={
                "query": self.operations.createCamera,
                "variables": variables,
            },
        )
        print(f"Done sending mutation request to create new camera")
        self.check_response(response)
        return response

    def create_shape(self, shape_input):
        """
        Create a new shape (line counter or polygon).

        :param shape_input: Dict containing shape metadata for creating a
            shape. See
            https://api.tracevision.com/graphql/v1/docs/types/ShapeInput
        :return response: Response from the API
        """
        print(
            f"Sending mutation request to create new shape for customer {self.customer_id}"
        )
        variables = {
            "token": self.customer_token,
            "shape": shape_input,
        }
        response = self.api_session.post(
            self.api_url,
            json={
                "query": self.operations.createShape,
                "variables": variables,
            },
        )
        print(f"Done sending mutation request to create new shape")
        self.check_response(response)
        return response

    def create_session_from_json(self, session_input_json_path):
        """
        Create a vision session from a JSON file.

        :param session_input_json_path: Path to JSON file containing session
            metadata
        :return session_id: Session ID
        """
        # Get session input data:
        with open(session_input_json_path, "r") as f:
            session_input = json.load(f)
        # Create a session:
        create_session_response = self.create_new_session(session_input)
        # Get the new session ID from the response:
        session_id = self.get_session_id(create_session_response)
        print(f"Created session with ID: {session_id}")
        return session_id

    def create_session_from_json_and_video_file(
        self, session_input_json_path, video_filepath
    ):
        """
        Create a vision session from a JSON file and a video file.

        Optionally set custom runtimes for the session.

        :param session_input_json_path: Path to JSON file containing session
            metadata
        :param video_filepath: Path to video file
        :return session_id: Session ID
        """
        # Create session from input JSON file:
        session_id = self.create_session_from_json(session_input_json_path)
        # Get session input data:
        with open(session_input_json_path, "r") as f:
            session_input = json.load(f)
        # Find the video file size:
        # Note that AWS S3 has a limit of 5 GB for the size of files that can
        # be uploaded with a single PUT request. If the video file is larger
        # than 5 GB, use multipart upload.
        video_filesize_bytes = os.path.getsize(video_filepath)
        print(f"Video file size: {video_filesize_bytes} bytes")
        max_single_upload_bytes = 4.9 * 1024 * 1024 * 1024  # 4.9 (just under 5) GB
        if video_filesize_bytes < max_single_upload_bytes:
            # Upload video in a single PUT request
            (
                upload_video_response,
                put_video_response,
            ) = self.upload_video(session_id, session_input, video_filepath)
        else:
            # Use multi-part upload to upload the video
            print("Using multi-part upload")
            # Calculate the number of parts to split the video into:
            n_parts = int(round(video_filesize_bytes / max_single_upload_bytes + 0.5))
            (
                upload_video_response,
                put_video_responses,
                complete_multipart_upload_response,
            ) = self.upload_video_multipart(
                session_id,
                session_input,
                video_filepath,
                n_parts,
            )
        return session_id
