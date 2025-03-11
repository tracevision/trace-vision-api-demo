#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Dan Kubala, Claire Roberts-Thomson
#
"""
vision_api_queries.py
"""
from graphql_query import Query
from vision_api_arguments import VisionAPIArguments
from vision_api_fields import VisionAPIFields


class VisionAPIQueries:
    def __init__(self):
        """
        Vision API queries

        Class to generate GraphQL queries for the Trace Vision API.
        """

        self.arguments = VisionAPIArguments()
        self.fields = VisionAPIFields()

        self.session = Query(
            name="session",
            arguments=[self.arguments.token, self.arguments.session_id],
            fields=[
                "session_id",
                "type",
                "status",
                "status_callback_url",
                "metadata",
                self.fields.camera,
                self.fields.TeamGameSessionFragment,
                self.fields.SessionStatusDetails,
            ],
        )

        self.sessions = Query(
            name="sessions",
            arguments=[
                self.arguments.token,
                self.arguments.limit,
                self.arguments.offset,
            ],
            fields=["session_id", "type", "status", self.fields.videos],
        )

        self.sessions_teams = Query(
            name="sessions",
            arguments=[
                self.arguments.token,
            ],
            fields=[
                "session_id",
                "type",
                "status",
                self.fields.TeamGameSessionFragment_team_name_id,
            ],
        )

        self.sessionResult = Query(
            name="sessionResult",
            arguments=[self.arguments.token, self.arguments.session_id],
            fields=[
                self.fields.objects,
                self.fields.highlights,
            ],  # , self.fields.events],
        )

        self.facility = Query(
            name="facility",
            arguments=[self.arguments.token, self.arguments.facility_id],
            fields=[
                "facility_id",
                "name",
                "type",
                "description",
                "address",
                "longitude",
                "latitude",
                "metadata",
                self.fields.cameras,
            ],
        )

        self.facilities = Query(
            name="facilities",
            arguments=[
                self.arguments.token,
                self.arguments.limit,
                self.arguments.offset,
            ],
            fields=[
                "facility_id",
                "name",
                "type",
                "description",
                "address",
                "longitude",
                "latitude",
                "metadata",
                self.fields.cameras,
            ],
        )

        self.camera = Query(
            name="camera",
            arguments=[self.arguments.token, self.arguments.camera_id],
            fields=[
                "camera_id",
                "name",
                "model",
                "indoor",
                self.fields.facility,
                "location",
                "longitude",
                "latitude",
                "fov",
                "direction_north",
                "group_id",
                "metadata",
                "enabled",
            ],
        )

        self.createSession = Query(
            name="createSession",
            arguments=[self.arguments.token, self.arguments.session_create_input],
            fields=["success", "error", self.fields.session],
        )

        self.updateSession = Query(
            name="updateSession",
            arguments=[
                self.arguments.token,
                self.arguments.session_id,
                self.arguments.session_update_input,
            ],
            fields=["success", "error", self.fields.session],
        )

        self.createFacility = Query(
            name="createFacility",
            arguments=[self.arguments.token, self.arguments.facility_input],
            fields=["success", "error", self.fields.facility],
        )

        self.updateFacility = Query(
            name="updateFacility",
            arguments=[
                self.arguments.token,
                self.arguments.facility_id,
                self.arguments.facility_update_input,
            ],
            fields=["success", "error", self.fields.facility],
        )

        self.createCamera = Query(
            name="createCamera",
            arguments=[self.arguments.token, self.arguments.camera_input],
            fields=["success", "error", self.fields.camera],
        )

        self.updateCamera = Query(
            name="updateCamera",
            arguments=[
                self.arguments.token,
                self.arguments.camera_id,
                self.arguments.camera_update_input,
            ],
            fields=["success", "error", self.fields.camera],
        )

        self.upload_video = Query(
            name="uploadVideo",
            arguments=[
                self.arguments.token,
                self.arguments.session_id,
                self.arguments.video_name,
                self.arguments.start_time,
            ],
            fields=[
                "success",
                "error",
                "upload_url",
            ],
        )

        self.multipartUploadVideo = Query(
            name="multipartUploadVideo",
            arguments=[
                self.arguments.token,
                self.arguments.session_id,
                self.arguments.video_name,
                self.arguments.start_time,
                self.arguments.total_parts,
            ],
            fields=["success", "error", "upload_id", self.fields.upload_parts],
        )

        self.multipartUploadVideoComplete = Query(
            name="multipartUploadVideoComplete",
            arguments=[
                self.arguments.token,
                self.arguments.session_id,
                self.arguments.upload_id,
                self.arguments.parts_info,
            ],
            fields=["success", "error"],
        )

        self.importVideo = Query(
            name="importVideo",
            arguments=[
                self.arguments.token,
                self.arguments.session_id,
                self.arguments.import_video_input,
                self.arguments.start_time,
            ],
            fields=[
                "success",
                "error",
            ],
        )
