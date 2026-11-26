"""Documentation consistency tests for trace-vision-api-demo.

Stdlib only. Run: python3 -m unittest discover -s tests -v
"""
import ast
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCKER_README = ROOT / "docker-image" / "readme.txt"
REPO_MOUNT = "/tmp/trace-vision-api-demo"
FOOTAGE_MOUNT = "/tmp/local-video-footage"
PY_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.py(?![A-Za-z0-9_])")


def referenced_py(text):
    return set(PY_RE.findall(text or ""))


def check_py_exist(texts, root):
    """Fail naming the first referenced *.py file that does not exist."""
    for text in texts:
        for name in sorted(referenced_py(text)):
            if not (Path(root) / name).exists():
                raise AssertionError("missing script referenced: %s" % name)


def docstrings(root):
    """Module docstrings of every .py in the repo, excluding tests/."""
    out = []
    for path in sorted(Path(root).rglob("*.py")):
        if "tests" in path.relative_to(root).parts:
            continue
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
        if doc is not None:
            out.append(doc)
    return out


def check_docstrings_exist(root):
    check_py_exist(docstrings(root), root)


def mounts(text):
    """All container targets of -v mounts in a docker run command."""
    return re.findall(r"-v\s+\S+:(/\S+)", text or "")


def video_mounts(text):
    """Container targets of -v mounts that are not the repo mount."""
    return [t for t in mounts(text) if t != REPO_MOUNT]


def check_mounts(readme_text, docker_text):
    """Video-footage mount must agree across both docs and prose."""
    for label, text in (("README", readme_text),
                        ("docker-image/readme.txt", docker_text)):
        found = mounts(text)
        if REPO_MOUNT not in found:
            raise AssertionError(
                "%s is missing the repo mount %r (found %s)"
                % (label, REPO_MOUNT, found))
        for target in video_mounts(text):
            if target != FOOTAGE_MOUNT:
                raise AssertionError(
                    "%s video-footage mount %r != %r"
                    % (label, target, FOOTAGE_MOUNT))
    for label, text in (("README", readme_text),
                        ("docker-image/readme.txt", docker_text)):
        prose = set(re.findall(r"/tmp/[A-Za-z0-9_-]+", text or ""))
        prose.discard(REPO_MOUNT)
        bad = sorted(p for p in prose if p != FOOTAGE_MOUNT)
        if bad:
            raise AssertionError(
                "%s prose paths %s != mount %r"
                % (label, bad, FOOTAGE_MOUNT))
    if "/tmp/data" in (readme_text or "") + (docker_text or ""):
        raise AssertionError("stale /tmp/data reference remains")


class TestDocs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = README.read_text(encoding="utf-8")
        cls.docker = DOCKER_README.read_text(encoding="utf-8")

    def test_readme_step3_uses_existing_session_script(self):
        self.assertIn("#### 3.", self.readme)
        section = self.readme.split("#### 3.", 1)[1].split("####", 1)[0]
        self.assertIn("create_session_full_example.py", section)
        self.assertNotIn("create_trace_vision_session.py", self.readme)

    def test_docker_video_footage_mount_consistency(self):
        check_mounts(self.readme, self.docker)

    def test_readme_script_list_includes_all_sample_scripts(self):
        self.assertIn("## Python sample scripts", self.readme)
        section = self.readme.split(
            "## Python sample scripts", 1)[1].split(
            "### Python environment", 1)[0]
        self.assertIn("get_facility_result.py", section)
        self.assertIn("facilityResult", section)
        self.assertIn("coordinate_conversions.py", section)
        self.assertIn("latitude/longitude", section)
        self.assertIn("x/y", section)

    def test_readme_heading_and_no_environnment_typo(self):
        self.assertIn("### Python environment", self.readme)
        for text in [self.readme, self.docker] + docstrings(ROOT):
            self.assertNotIn("environnment", text)

    def test_all_referenced_py_files_exist_in_docs_and_docstrings(self):
        check_py_exist([self.readme, self.docker], ROOT)
        check_docstrings_exist(ROOT)

    def test_four_named_docstrings_reference_create_session_full_example(self):
        for name in ("count_customers.py",
                     "use_trace_vision_session_results.py",
                     "create_session_full_example.py",
                     "find_similar_objects.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            doc = ast.get_docstring(tree) or ""
            self.assertIn("create_session_full_example.py", doc, name)
            self.assertNotIn("create_trace_vision_session.py", doc, name)

    def test_negative_script_existence_validator_and_none_docstring(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError) as ctx:
                check_py_exist(
                    ["run python create_trace_vision_session.py --x"], td)
            self.assertIn("create_trace_vision_session.py",
                          str(ctx.exception))
            Path(td, "nodoc.py").write_text("X = 1\n", encoding="utf-8")
            check_docstrings_exist(td)  # None docstring: no error

    def test_negative_mount_consistency_fixtures(self):
        good_docker = "docker run -v /a:/tmp/trace-vision-api-demo " \
            "-v /v:%s x" % FOOTAGE_MOUNT
        bad_readme_i = "docker run -v /a:/tmp/trace-vision-api-demo " \
            "-v /v:/tmp/data x\nnotes use %s" % FOOTAGE_MOUNT
        with self.assertRaises(AssertionError) as ctx:
            check_mounts(bad_readme_i, good_docker)
        self.assertIn("/tmp/data", str(ctx.exception))
        bad_readme_ii = "docker run -v /a:/tmp/trace-vision-api-demo " \
            "-v /v:%s x\nstep uses /tmp/data" % FOOTAGE_MOUNT
        with self.assertRaises(AssertionError) as ctx:
            check_mounts(bad_readme_ii, good_docker)
        self.assertIn("/tmp/data", str(ctx.exception))
        bad_docker_iii = "docker run -v /a:/tmp/trace-vision-api-demo " \
            "-v /v:/tmp/data x"
        good_readme = "docker run -v /a:/tmp/trace-vision-api-demo " \
            "-v /v:%s x\nnotes use %s" % (FOOTAGE_MOUNT, FOOTAGE_MOUNT)
        with self.assertRaises(AssertionError) as ctx:
            check_mounts(good_readme, bad_docker_iii)
        self.assertIn("/tmp/data", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
