"""Guards against version drift between pyproject.toml and retrieval.__version__.

Parses pyproject.toml's version field with a plain regex rather than
tomllib/importlib.metadata: the engine targets py310, and a regex needs no
extra import machinery to check a single string.
"""

import pathlib
import re
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import retrieval  # noqa: E402


class TestVersionMatchesPyproject(unittest.TestCase):
    def test_version_matches_pyproject(self) -> None:
        pyproject_text = (_ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.M)
        self.assertIsNotNone(match, "no version field found in pyproject.toml")
        self.assertEqual(retrieval.__version__, match.group(1))


if __name__ == "__main__":
    unittest.main()
