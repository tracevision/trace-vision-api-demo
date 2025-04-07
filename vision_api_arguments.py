#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Dan Kubala, Claire Roberts-Thomson
#
"""
vision_api_arguments.py
"""
from graphql_query import Argument
from vision_api_variables import VisionAPIVariables


class VisionAPIArguments:
    def __init__(self):
        """
        Vision API arguments

        Class to generate GraphQL arguments for the Trace Vision API.
        """

        variables = VisionAPIVariables()

        self.token = Argument(name="token", value=variables.token)
        self.limit = Argument(name="limit", value=variables.limit)
        self.offset = Argument(name="offset", value=variables.offset)
        self.session_create_input = Argument(
            name="sessionData", value=variables.session_create_input
        )
        self.session_update_input = Argument(
            name="sessionData", value=variables.session_update_input
        )
        self.session_id = Argument(name="session_id", value=variables.session_id)
        self.facility_input = Argument(name="facility", value=variables.facility_input)
        self.facility_update_input = Argument(
            name="facility", value=variables.facility_update_input
        )
        self.facility_id = Argument(name="facility_id", value=variables.facility_id)
        self.camera_input = Argument(name="camera", value=variables.camera_input)
        self.camera_update_input = Argument(
            name="camera", value=variables.camera_update_input
        )
        self.camera_id = Argument(name="camera_id", value=variables.camera_id)
        self.upload_id = Argument(name="upload_id", value=variables.upload_id)
        self.video_name = Argument(name="video_name", value=variables.video_name)
        self.start_time = Argument(name="start_time", value=variables.start_time)
        self.total_parts = Argument(name="total_parts", value=variables.total_parts)
        self.parts_info = Argument(name="parts_info", value=variables.parts_info)
        self.import_video_input = Argument(
            name="video", value=variables.import_video_input
        )
        self.shape_input = Argument(name="shape", value=variables.shape_input)
