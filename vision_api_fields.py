#
#  Copyright Alpinereplay Inc., 2024. All rights reserved.
#  Authors: Dan Kubala, Claire Roberts-Thomson
#
"""
vision_api_fields.py
"""
from graphql_query import Field, InlineFragment


class VisionAPIFields:
    def __init__(self):
        """
        Vision API fields

        Class to generate GraphQL fields for the Trace Vision API.
        """

        self.upload_parts = Field(name="upload_parts", fields=["upload_url", "part"])

        self.videos = Field(
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

        self.camera = Field(
            name="camera",
            fields=[
                "camera_id",
                "name",
                "model",
                "indoor",
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

        self.cameras = Field(
            name="cameras",
            fields=[
                "camera_id",
                "name",
                "model",
                "indoor",
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

        self.facility = Field(
            name="facility",
            fields=[
                "facility_id",
                "name",
                "type",
                "description",
                "address",
                "longitude",
                "latitude",
                "metadata",
            ],
        )

        self.objects = Field(
            name="objects",
            fields=[
                "object_id",
                "type",
                "side",
                "appearance_fv",
                "color_fv",
                "tracking_url",
                "role",
            ],
        )

        self.detected_objects = Field(
            name="detected_objects",
            fields=[
                "object_id",
                "type",
                "side",
                "appearance_fv",
                "color_fv",
                "tracking_url",
                "role",
            ],
        )

        self.highlights = Field(
            name="highlights",
            fields=[
                "highlight_id",
                "video_id",
                "start_offset",
                "duration",
                "tags",
                "video_stream",
                self.objects,
            ],
        )

        self.bbox = Field(
            name="bbox",
            fields=["x", "y"],
        )

        self.world_location = Field(
            name="world_location",
            fields=["latitude", "longitude"],
        )

        self.events = Field(
            name="events",
            fields=[
                "event_id",
                "start_time",
                "event_time",
                "end_time",
                "type",
                self.bbox,
                self.world_location,
                self.detected_objects,
            ],
        )

        self.TeamInfo = ["team_id", "name", "score", "color"]
        self.home_team = Field(name="home_team", fields=self.TeamInfo)
        self.away_team = Field(name="away_team", fields=self.TeamInfo)

        self.TeamGameSessionFragment = InlineFragment(
            type="TeamGameSession",
            fields=[
                "session_id",
                "type",
                "status",
                self.videos,
                "status_callback_url",
                "metadata",
                self.home_team,
                self.away_team,
                "start_time",
            ],
        )

        self.TeamGameSessionFragment_team_name_id = InlineFragment(
            type="TeamGameSession",
            fields=[
                "session_id",
                "type",
                "status",
                self.home_team,
                self.away_team,
            ],
        )

        self.session = Field(
            name="session",
            fields=["session_id", "type", "status", self.videos],
        )

        self.SessionStatusDetails = Field(
            name="status_details",
            fields=[
                "error_code",
                "description",
            ],
        )

        self.httpHeaderInput = Field(
            name="httpHeaderInput",
            fields=["key", "value"],
        )

        self.importVideoInput = Field(
            name="importVideoInput",
            fields=["type", "via_url", self.httpHeaderInput],
        )
