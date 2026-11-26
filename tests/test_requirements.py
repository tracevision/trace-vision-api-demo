"""Every third-party import used by the sample scripts must be pinned in requirements.txt."""
import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMPORT_TO_DIST = {
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "dateutil": "python-dateutil",
}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def collect_third_party_imports():
    local = {p.stem for p in ROOT.glob("*.py")}
    stdlib = sys.stdlib_module_names
    imports = {}
    for path in sorted(ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in stdlib or name in local:
                    continue
                imports.setdefault(name, str(path.name))
    return imports


def pinned_distributions():
    pinned = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        pinned.add(normalize(line.split("==", 1)[0].strip()))
    return pinned


def unpinned(imports, pinned):
    missing = []
    for name, where in sorted(imports.items()):
        dist = IMPORT_TO_DIST.get(name, name)
        if normalize(dist) not in pinned:
            missing.append(f"{name} (from {where}) -> {dist}")
    return missing


class TestRequirements(unittest.TestCase):
    def test_all_script_third_party_imports_are_pinned(self):
        imports = collect_third_party_imports()
        self.assertTrue(imports, "expected third-party imports in scripts")
        missing = unpinned(imports, pinned_distributions())
        self.assertEqual(missing, [], f"unpinned third-party imports: {missing}")

    def test_missing_pin_detected(self):
        imports = collect_third_party_imports()
        pinned = pinned_distributions()
        pinned.discard(normalize("Pillow"))
        missing = unpinned(imports, pinned)
        self.assertTrue(any("PIL" in m for m in missing), f"expected PIL flagged, got {missing}")


if __name__ == "__main__":
    unittest.main()
