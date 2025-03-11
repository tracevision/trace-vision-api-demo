#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Dan Kubala, Claire Roberts-Thomson
#
"""
vision_api_variables.py
"""
from graphql_query import Variable


class VisionAPIVariables:

    def __init__(self):
        """
        Vision API variables

        Class to generate GraphQL variables for the Trace Vision API.
        """

        self.token = Variable(name="token", type="CustomerToken!")
        self.limit = Variable(name="limit", type="Int")
        self.offset = Variable(name="offset", type="Int")
        self.session_create_input = Variable(
            name="sessionData", type="SessionCreateInput!"
        )
        self.session_update_input = Variable(
            name="sessionData", type="SessionUpdateInput!"
        )
        self.session_id = Variable(name="session_id", type="Int!")
        self.facility_id = Variable(name="facility_id", type="Int!")
        self.facility_input = Variable(name="facility", type="FacilityInput!")
        self.facility_update_input = Variable(
            name="facility", type="FacilityUpdateInput!"
        )
        self.camera_id = Variable(name="camera_id", type="Int!")
        self.camera_input = Variable(name="camera", type="CameraInput!")
        self.camera_update_input = Variable(name="camera", type="CameraUpdateInput!")
        self.video_name = Variable(name="video_name", type="String!")
        self.start_time = Variable(name="start_time", type="DateTime")
        self.total_parts = Variable(name="total_parts", type="Int!")
        self.upload_id = Variable(name="upload_id", type="String!")
        self.parts_info = Variable(name="parts_info", type="[UploadVideoPartInput!]!")
        self.import_video_input = Variable(name="video", type="ImportVideoInput!")
