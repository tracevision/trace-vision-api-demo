"""Offline characterization tests for paging in VisionAPIInterface.

Stdlib unittest only. Batch functions are faked at the
_fetch_sessions_batch / _fetch_facilities_batch boundary: no API key,
no network. requests/pandas/graphql_query stubbed in sys.modules
only if unavailable in the interpreter running these tests.
"""
import contextlib, io, sys, types, unittest
def _stub_requests():
    mod = types.ModuleType("requests")
    class _Session:
        def __init__(self, *a, **k):
            pass
        def post(self, *a, **k):
            raise AssertionError("no network in offline tests")
    mod.Session = _Session
    return mod

def _stub_graphql_query():
    mod = types.ModuleType("graphql_query")
    class _Any:
        def __init__(self, *a, **k):
            pass
        def render(self, *a, **k):
            return ""
    mod.Argument = _Any
    mod.Field = _Any
    mod.Operation = _Any
    mod.Query = _Any
    mod.Variable = _Any
    return mod

_STUBS = {"requests": _stub_requests, "pandas": types.ModuleType,
          "graphql_query": _stub_graphql_query}
for _n, _make in _STUBS.items():
    try:
        __import__(_n)
    except ImportError:
        sys.modules[_n] = _make(_n) if _n == "pandas" else _make()
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
        self.assertIn("Grabbing all sessions for customer 1, starting from offset 0", printed)
        self.assertIn("Finished fetching all sessions. Total retrieved: 250", printed)

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
        self.assertIn("Grabbing all facilities for customer 1, starting from offset 0", printed)
        self.assertIn("Finished fetching all facilities. Total retrieved: 250", printed)
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

    def test_negative_limit_none_on_empty_and_subsequent_empty(self):
        self.install_facilities(250)
        out, _ = self.call_quiet(self.iface.get_facilities, limit=-5)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(-5, 0)])
        self.calls.clear()
        self.install_facilities(0)
        out, _ = self.call_quiet(self.iface.get_facilities)
        self.assertEqual(out, [])
        self.assertEqual(self.pairs(), [(100, 0)])
        seq = [[0] * 100, []]
        self.iface._fetch_facilities_batch = lambda limit=None, offset=None, **k: seq.pop(0)
        out, _ = self.call_quiet(self.iface.get_facilities, limit=250)
        self.assertEqual(len(out), 100)

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


class TestFetchPaginatedHelper(PagingBase):
    def test_helper_exists_and_signature_defaults(self):
        import inspect
        self.assertTrue(hasattr(VisionAPIInterface, "_fetch_paginated"))
        sig = inspect.signature(VisionAPIInterface._fetch_paginated)
        params = list(sig.parameters.values())
        self.assertEqual([p.name for p in params], ["self", "fetch_batch", "limit", "offset"])
        self.assertIsNone(params[2].default)
        self.assertIsNone(params[3].default)
        out = self.iface._fetch_paginated(lambda limit=None, offset=None: [1, 2])
        self.assertIsInstance(out, list)
        self.assertEqual(out, [1, 2])

    def test_helper_calls_fetch_batch_with_keyword_args_and_api_limit(self):
        items = list(range(250))
        calls = []
        out = self.iface._fetch_paginated(make_fake(items, calls))
        self.assertEqual(out, items)
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(100, 0), (100, 100), (100, 200)])
        calls.clear()
        out = self.iface._fetch_paginated(make_fake(items, calls), limit=50)
        self.assertEqual(out, items[:50])
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(50, 0)])
        calls.clear()
        out = self.iface._fetch_paginated(make_fake(items, calls), limit=0)
        self.assertEqual(out, [])
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(0, 0)])
        calls.clear()
        out = self.iface._fetch_paginated(make_fake(items, calls), limit=250)
        self.assertEqual(out, items)
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(100, 0), (100, 100), (50, 200)])
        calls.clear()
        out = self.iface._fetch_paginated(make_fake(items, calls), limit=200)
        self.assertEqual(out, items[:200])
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(100, 0), (100, 100)])
        calls.clear()
        out = self.iface._fetch_paginated(make_fake(items, calls), limit=150, offset=25)
        self.assertEqual(out, items[25:175])
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(100, 25), (50, 125)])
        self.iface.api_limit = 50
        calls.clear()
        out = self.iface._fetch_paginated(make_fake(list(range(120)), calls), limit=120)
        self.assertEqual(out, list(range(120)))
        self.assertEqual([(c["limit"], c["offset"]) for c in calls], [(50, 0), (50, 50), (20, 100)])

    def test_helper_propagates_exceptions(self):
        def boom(limit=None, offset=None):
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.iface._fetch_paginated(boom, limit=50)


class TestDelegationAndTodoRemoval(PagingBase):
    def test_get_sessions_delegates_to_fetch_paginated_and_preserves_prints(self):
        import unittest.mock as mock
        items = self.install_sessions(250)
        with mock.patch.object(self.iface, "_fetch_paginated", wraps=self.iface._fetch_paginated) as spy:
            out, printed = self.call_quiet(self.iface.get_sessions, limit=120, offset=10,
                camera_id=7, status="done", start_time_min="t0", start_time_max="t1")
        spy.assert_called_once()
        self.assertEqual(spy.call_args.kwargs.get("limit"), 120)
        self.assertEqual(spy.call_args.kwargs.get("offset"), 10)
        self.assertEqual(out, items[10:130])
        self.assertIn("Grabbing 120 sessions", printed)
        self.assertIn("Finished fetching 120 sessions", printed)
        for c in self.calls:
            self.assertEqual(c["kwargs"], {"camera_id": 7, "status": "done",
                "start_time_min": "t0", "start_time_max": "t1"})

    def test_get_facilities_delegates_to_fetch_paginated_and_preserves_prints(self):
        import unittest.mock as mock
        items = self.install_facilities(250)
        with mock.patch.object(self.iface, "_fetch_paginated", wraps=self.iface._fetch_paginated) as spy:
            out, printed = self.call_quiet(self.iface.get_facilities, limit=120, offset=10)
        spy.assert_called_once()
        self.assertEqual(spy.call_args.kwargs.get("limit"), 120)
        self.assertEqual(spy.call_args.kwargs.get("offset"), 10)
        self.assertEqual(out, items[10:130])
        self.assertIn("Grabbing 120 facilities", printed)
        self.assertIn("Finished fetching 120 facilities", printed)

    def test_fetch_facilities_batch_todo_removed(self):
        import inspect
        doc = inspect.getdoc(VisionAPIInterface._fetch_facilities_batch) or ""
        self.assertNotIn("TODO", doc)

if __name__ == "__main__":
    unittest.main()
