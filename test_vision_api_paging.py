"""Offline characterization tests for paging in VisionAPIInterface.

Stdlib unittest only. Batch functions are faked at the
_fetch_sessions_batch / _fetch_facilities_batch boundary: no API key,
no network. requests/pandas/graphql_query stubbed in sys.modules
only if unavailable in the interpreter running these tests.
"""
import contextlib, io, sys, types, unittest
for _n in ("requests", "pandas", "graphql_query"):
    try:
        __import__(_n)
    except ImportError:
        sys.modules[_n] = types.ModuleType(_n)
from vision_api_interface import VisionAPIInterface  # noqa: E402

def make_interface():
    return VisionAPIInterface(customer_id=1, api_key="dummy", api_url="http://dummy")

def make_fake(items, calls):
    def fake(limit=None, offset=None, **kwargs):
        calls.append({"limit": limit, "offset": offset, "kwargs": kwargs})
        if limit is None or limit <= 0:
            return []
        s = offset or 0
        return items[s:s + limit]
    return fake

class PagingBase(unittest.TestCase):
    def setUp(self):
        self.iface = make_interface()
        self.calls = []
    def install_sessions(self, total):
        items = list(range(total))
        self.iface._fetch_sessions_batch = make_fake(items, self.calls)
        return items
    def install_facilities(self, total):
        items = list(range(total))
        self.iface._fetch_facilities_batch = make_fake(items, self.calls)
        return items
    def call_quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()
    def pairs(self):
        return [(c["limit"], c["offset"]) for c in self.calls]

class TestGetSessionsCharacterization(PagingBase):
    def test_limit_none_batches_api_limit_until_short_batch(self):
        items = self.install_sessions(250)
        out, printed = self.call_quiet(self.iface.get_sessions)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(100, 0), (100, 100), (100, 200)])
        self.assertIn("Grabbing all sessions", printed)
        self.assertIn("Finished fetching all sessions", printed)

    def test_limit_none_exact_multiple_stops_on_empty_batch(self):
        items = self.install_sessions(200)
        out, _ = self.call_quiet(self.iface.get_sessions)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(100, 0), (100, 100), (100, 200)])

    def test_limit_le_api_limit_single_call(self):
        items = self.install_sessions(250)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=50)
        self.assertEqual(out, items[:50])
        self.assertEqual(self.pairs(), [(50, 0)])
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_sessions, limit=100)
        self.assertEqual(out, items[:100])
        self.assertEqual(self.pairs(), [(100, 0)])

    def test_limit_le_zero_one_call_passthrough(self):
        self.install_sessions(250)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=0)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(0, 0)])
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_sessions, limit=-5)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(-5, 0)])

    def test_limit_gt_api_limit_batches_and_shrinks_final_call(self):
        items = self.install_sessions(250)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=250)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(100, 0), (100, 100), (50, 200)])
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_sessions, limit=200)
        self.assertEqual(out, items[:200])
        self.assertEqual(self.pairs(), [(100, 0), (100, 100)])
        seq = [[0] * 100, [0] * 30]
        self.iface._fetch_sessions_batch = lambda limit=None, offset=None, **k: seq.pop(0) if seq else []
        out, _ = self.call_quiet(self.iface.get_sessions, limit=250)
        self.assertEqual(len(out), 130)

    def test_empty_batch_first_and_subsequent_ends_loop(self):
        self.install_sessions(0)
        out, _ = self.call_quiet(self.iface.get_sessions)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(100, 0)])
        self.calls.clear()
        self.install_sessions(0)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=250)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(100, 0)])
        seq = [[0] * 100, []]
        self.iface._fetch_sessions_batch = lambda limit=None, offset=None, **k: seq.pop(0)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=250)
        self.assertEqual(len(out), 100)

    def test_nonzero_offset_and_filter_forwarding(self):
        items = self.install_sessions(300)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=150, offset=25,
            camera_id=7, status="done", start_time_min="t0", start_time_max="t1")
        self.assertEqual(out, items[25:175])
        self.assertEqual(self.pairs(), [(100, 25), (50, 125)])
        for c in self.calls:
            self.assertEqual(c["kwargs"], {"camera_id": 7, "status": "done",
                "start_time_min": "t0", "start_time_max": "t1"})

    def test_custom_api_limit_controls_batch_size(self):
        self.iface.api_limit = 50
        items = self.install_sessions(120)
        out, _ = self.call_quiet(self.iface.get_sessions, limit=120)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(50, 0), (50, 50), (20, 100)])
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_sessions)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(50, 0), (50, 50), (50, 100)])

    def test_exception_propagates_unchanged(self):
        def boom(limit=None, offset=None, **k):
            raise RuntimeError("boom")
        self.iface._fetch_sessions_batch = boom
        with self.assertRaises(RuntimeError):
            self.call_quiet(self.iface.get_sessions, limit=50)

class TestGetFacilitiesCharacterization(PagingBase):
    def test_limit_none_batches_and_exact_multiple(self):
        items = self.install_facilities(250)
        out, printed = self.call_quiet(self.iface.get_facilities)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(100, 0), (100, 100), (100, 200)])
        self.assertIn("Grabbing all facilities", printed)
        self.assertIn("Finished fetching all facilities", printed)
        self.calls.clear()
        items = self.install_facilities(200)
        out, _ = self.call_quiet(self.iface.get_facilities)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(100, 0), (100, 100), (100, 200)])

    def test_limit_le_api_limit_and_limit_le_zero(self):
        items = self.install_facilities(250)
        for lim in (50, 100):
            self.calls.clear()
            out, _ = self.call_quiet(self.iface.get_facilities, limit=lim)
            self.assertEqual(out, items[:lim])
            self.assertEqual(self.pairs(), [(lim, 0)])
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_facilities, limit=0)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(0, 0)])

    def test_limit_gt_api_limit_and_exact_multiple(self):
        items = self.install_facilities(250)
        out, _ = self.call_quiet(self.iface.get_facilities, limit=250)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(100, 0), (100, 100), (50, 200)])
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_facilities, limit=200)
        self.assertEqual(out, items[:200])
        self.assertEqual(self.pairs(), [(100, 0), (100, 100)])
        seq = [[0] * 100, [0] * 30]
        self.iface._fetch_facilities_batch = lambda limit=None, offset=None, **k: seq.pop(0) if seq else []
        out, _ = self.call_quiet(self.iface.get_facilities, limit=250)
        self.assertEqual(len(out), 130)

    def test_empty_batch_and_nonzero_offset(self):
        self.install_facilities(0)
        out, _ = self.call_quiet(self.iface.get_facilities, limit=250)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(100, 0)])
        items = self.install_facilities(300)
        self.calls.clear()
        out, _ = self.call_quiet(self.iface.get_facilities, limit=150, offset=25)
        self.assertEqual(out, items[25:175])
        self.assertEqual(self.pairs(), [(100, 25), (50, 125)])

    def test_custom_api_limit_and_exception_propagation(self):
        self.iface.api_limit = 50
        items = self.install_facilities(120)
        out, _ = self.call_quiet(self.iface.get_facilities, limit=120)
        self.assertEqual(out, items)
        self.assertEqual(self.pairs(), [(50, 0), (50, 50), (20, 100)])
        def boom(limit=None, offset=None, **k):
            raise RuntimeError("boom")
        self.iface._fetch_facilities_batch = boom
        with self.assertRaises(RuntimeError):
            self.call_quiet(self.iface.get_facilities, limit=50)

if __name__ == "__main__":
    unittest.main()
