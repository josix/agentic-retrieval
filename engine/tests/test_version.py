"""Guards against version drift across all four version-declaration sites.

Parses pyproject.toml's version field with a plain regex rather than
tomllib/importlib.metadata: the engine targets py310, and a regex needs no
extra import machinery to check a single string. The plugin manifests
(.claude-plugin/plugin.json and .claude-plugin/marketplace.json) are parsed
with json.load and asserted pairwise against retrieval.__version__, since
the plugin and the engine are versioned in lockstep. Also asserts
docs/changelog.md has a section for the current version.
"""

import json
import pathlib
import re
import sys
import unittest

_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
_REPO_DIR = _ROOT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import retrieval  # noqa: E402


class TestVersionMatchesPyproject(unittest.TestCase):
    def test_version_matches_pyproject(self) -> None:
        pyproject_text = (_ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.M)
        self.assertIsNotNone(match, "no version field found in pyproject.toml")
        self.assertEqual(retrieval.__version__, match.group(1))


@unittest.skipUnless(
    (_REPO_DIR / ".claude-plugin").is_dir(),
    "manifests only present in a monorepo checkout",
)
class TestVersionMatchesPluginManifests(unittest.TestCase):
    # CI always runs against the full monorepo checkout, where
    # .claude-plugin/ is present, so this guard never weakens enforcement —
    # it only avoids failing when the engine is exercised standalone (e.g.
    # a built sdist/wheel, which has no .claude-plugin/ directory at all).

    def test_version_matches_plugin_json(self) -> None:
        plugin_json_path = _REPO_DIR / ".claude-plugin" / "plugin.json"
        data = json.loads(plugin_json_path.read_text(encoding="utf-8"))
        self.assertEqual(
            retrieval.__version__,
            data["version"],
            f"retrieval.__version__ ({retrieval.__version__}) does not match "
            f"{plugin_json_path}'s version ({data['version']})",
        )

    def test_version_matches_marketplace_json(self) -> None:
        marketplace_json_path = _REPO_DIR / ".claude-plugin" / "marketplace.json"
        data = json.loads(marketplace_json_path.read_text(encoding="utf-8"))
        entry = next(
            plugin
            for plugin in data["plugins"]
            if plugin["name"] == "agentic-retrieval"
        )
        self.assertEqual(
            retrieval.__version__,
            entry["version"],
            f"retrieval.__version__ ({retrieval.__version__}) does not match "
            f"the agentic-retrieval entry's version in {marketplace_json_path} "
            f"({entry['version']})",
        )


@unittest.skipUnless(
    (_REPO_DIR / "docs" / "changelog.md").is_file(),
    "changelog only present in a monorepo checkout",
)
class TestChangelogHasCurrentVersionSection(unittest.TestCase):
    def test_changelog_has_section_for_current_version(self) -> None:
        changelog_path = _REPO_DIR / "docs" / "changelog.md"
        text = changelog_path.read_text(encoding="utf-8")
        headings = re.findall(r"^## (.+)$", text, re.M)
        latest_heading = next(
            (
                heading
                for heading in headings
                if heading.strip().lower() != "unreleased"
            ),
            None,
        )
        self.assertIsNotNone(
            latest_heading,
            "docs/changelog.md has no released version section — add a "
            "`## <version>` heading",
        )
        version_match = re.match(r"^(\d+\.\d+\.\d+)", latest_heading.strip())
        self.assertIsNotNone(
            version_match,
            f"could not parse a version out of changelog heading {latest_heading!r}",
        )
        self.assertEqual(
            retrieval.__version__,
            version_match.group(1),
            f"docs/changelog.md's latest dated section ({version_match.group(1)}) "
            f"does not match retrieval.__version__ ({retrieval.__version__}) — "
            "add a `## <version>` section to docs/changelog.md, or fix the "
            "version sites",
        )


if __name__ == "__main__":
    unittest.main()
