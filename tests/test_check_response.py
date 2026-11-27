import io
import json
import unittest
from contextlib import redirect_stdout

from vision_api_interface import VisionAPIInterface


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class TestCheckResponse(unittest.TestCase):
    def test_401_raises_with_status_and_truncated_body(self):
        body = "a" * 500 + "TAIL_SENTINEL_401"
        with self.assertRaises(ValueError) as cm:
            VisionAPIInterface.check_response(FakeResponse(401, body))
        msg = str(cm.exception)
        self.assertIn("401", msg)
        self.assertIn(body[:500], msg)
        self.assertNotIn("TAIL_SENTINEL_401", msg)
        self.assertTrue(msg.endswith(body[:500]))

    def test_403_raises_with_status_and_truncated_body(self):
        body = "b" * 500 + "TAIL_SENTINEL_403"
        with self.assertRaises(ValueError) as cm:
            VisionAPIInterface.check_response(FakeResponse(403, body))
        msg = str(cm.exception)
        self.assertIn("403", msg)
        self.assertIn(body[:500], msg)
        self.assertNotIn("TAIL_SENTINEL_403", msg)
        self.assertTrue(msg.endswith(body[:500]))

    def test_non_200_empty_body_raises_with_status(self):
        with self.assertRaises(ValueError) as cm:
            VisionAPIInterface.check_response(FakeResponse(401, ""))
        self.assertIn("401", str(cm.exception))

    def test_500_raises_with_status_and_truncated_body(self):
        body = "c" * 500 + "TAIL_SENTINEL_500"
        with self.assertRaises(ValueError) as cm:
            VisionAPIInterface.check_response(FakeResponse(500, body))
        msg = str(cm.exception)
        self.assertIn("500", msg)
        self.assertIn(body[:500], msg)
        self.assertNotIn("TAIL_SENTINEL_500", msg)
        self.assertTrue(msg.endswith(body[:500]))

    def test_200_with_errors_prints_header_and_messages_and_returns_none(self):
        payload = json.dumps({"errors": [{"message": "err one"}, {"message": "err two"}]})
        out = io.StringIO()
        with redirect_stdout(out):
            result = VisionAPIInterface.check_response(FakeResponse(200, payload))
        self.assertIsNone(result)
        text = out.getvalue()
        self.assertIn("Errors received from the API:", text)
        self.assertIn("err one", text)
        self.assertIn("err two", text)

    def test_200_null_json_raises_received_null_response(self):
        with self.assertRaises(ValueError) as cm:
            VisionAPIInterface.check_response(FakeResponse(200, "null"))
        self.assertIn("Received null response", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
