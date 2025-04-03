#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Dan Kubala, Claire Roberts-Thomson
#
"""
vision_api_operations.py
"""
from graphql_query import Operation

from vision_api_variables import VisionAPIVariables
from vision_api_queries import VisionAPIQueries


class VisionAPIOperations:
    def __init__(self):
        """
        Vision API operations

        Class to generate GraphQL operations for the Trace Vision API.
        """

        self.variables = VisionAPIVariables()
        self.queries = VisionAPIQueries()

        # Queries
        self.session = Operation(
            type="query",
            name="session",
            variables=[self.variables.token, self.variables.session_id],
            queries=[self.queries.session],
        ).render()

        self.sessions = Operation(
            type="query",
            name="sessions",
            variables=[
                self.variables.token,
                self.variables.limit,
                self.variables.offset,
            ],
            queries=[self.queries.sessions],
        ).render()

        self.sessions_teams = Operation(
            type="query",
            name="sessions",
            variables=[
                self.variables.token,
            ],
            queries=[self.queries.sessions_teams],
        ).render()

        self.sessionResult = Operation(
            type="query",
            name="sessionResult",
            variables=[self.variables.token, self.variables.session_id],
            queries=[self.queries.sessionResult],
        ).render()

        self.facility = Operation(
            type="query",
            name="facility",
            variables=[self.variables.token, self.variables.facility_id],
            queries=[self.queries.facility],
        ).render()

        self.facilities = Operation(
            type="query",
            name="facilities",
            variables=[
                self.variables.token,
                self.variables.limit,
                self.variables.offset,
            ],
            queries=[self.queries.facilities],
        ).render()

        self.camera = Operation(
            type="query",
            name="camera",
            variables=[self.variables.token, self.variables.camera_id],
            queries=[self.queries.camera],
        ).render()

        self.shapes = Operation(
            type="query",
            name="shapes",
            variables=[
                self.variables.token,
                self.variables.limit,
                self.variables.offset,
            ],
            queries=[self.queries.shapes],
        ).render()

        # Mutations
        self.createSession = Operation(
            type="mutation",
            name="createSession",
            variables=[self.variables.token, self.variables.session_create_input],
            queries=[self.queries.createSession],
        ).render()

        self.updateSession = Operation(
            type="mutation",
            name="updateSession",
            variables=[
                self.variables.token,
                self.variables.session_id,
                self.variables.session_update_input,
            ],
            queries=[self.queries.updateSession],
        ).render()

        self.createFacility = Operation(
            type="mutation",
            name="createFacility",
            variables=[self.variables.token, self.variables.facility_input],
            queries=[self.queries.createFacility],
        ).render()

        self.updateFacility = Operation(
            type="mutation",
            name="updateFacility",
            variables=[
                self.variables.token,
                self.variables.facility_id,
                self.variables.facility_update_input,
            ],
            queries=[self.queries.updateFacility],
        ).render()

        self.createCamera = Operation(
            type="mutation",
            name="createCamera",
            variables=[self.variables.token, self.variables.camera_input],
            queries=[self.queries.createCamera],
        ).render()

        self.updateCamera = Operation(
            type="mutation",
            name="updateCamera",
            variables=[
                self.variables.token,
                self.variables.camera_id,
                self.variables.camera_update_input,
            ],
            queries=[self.queries.updateCamera],
        ).render()

        self.uploadVideo = Operation(
            type="mutation",
            name="uploadVideo",
            variables=[
                self.variables.token,
                self.variables.session_id,
                self.variables.video_name,
                self.variables.start_time,
            ],
            queries=[self.queries.upload_video],
        ).render()

        self.multipartUploadVideo = Operation(
            type="mutation",
            name="multipartUploadVideo",
            variables=[
                self.variables.token,
                self.variables.session_id,
                self.variables.video_name,
                self.variables.start_time,
                self.variables.total_parts,
            ],
            queries=[self.queries.multipartUploadVideo],
        ).render()

        self.multipartUploadVideoComplete = Operation(
            type="mutation",
            name="multipartUploadVideoComplete",
            variables=[
                self.variables.token,
                self.variables.session_id,
                self.variables.upload_id,
                self.variables.parts_info,
            ],
            queries=[self.queries.multipartUploadVideoComplete],
        ).render()

        self.importVideo = Operation(
            type="mutation",
            name="importVideo",
            variables=[
                self.variables.token,
                self.variables.session_id,
                self.variables.import_video_input,
                self.variables.start_time,
            ],
            queries=[self.queries.importVideo],
        ).render()
