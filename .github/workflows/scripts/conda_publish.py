#!/usr/bin/env python3
# ruff: noqa: T201
"""Shared helpers for Conda plugin publication and verification.

Provides functions for:
- Parsing and validating plugin tags
- Querying Anaconda.org for package versions and labels
- Verifying release consistency across GitHub / PyPI / Conda
- Building Conda packages from a specific git tag
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

TAG_RE = re.compile(
    r"^(?P<plugin>spectrochempy-[a-z0-9-]+)-v(?P<version>\d+\.\d+\.\d+)$"
)

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
ANACONDA_FILES_URL = "https://api.anaconda.org/package/{owner}/{package}/files"


def parse_plugin_tag(tag: str) -> tuple[str, str] | None:
    """Parse a plugin tag into (plugin_name, version) or None if invalid."""
    match = TAG_RE.match(tag)
    if not match:
        return None
    return match.group("plugin"), match.group("version")


def validate_tag_format(tag: str, expected_plugin: str | None = None) -> tuple[str, str]:
    """Validate a plugin tag and return (plugin_name, version).

    Raises ValueError with a clear diagnostic if the tag is invalid.
    """
    result = parse_plugin_tag(tag)
    if result is None:
        raise ValueError(
            f"Tag '{tag}' does not match the plugin release pattern "
            f"'<plugin_name>-v<version>' (e.g. spectrochempy-nmr-v0.1.1)"
        )
    plugin, version = result
    if expected_plugin and plugin != expected_plugin:
        raise ValueError(
            f"Tag '{tag}' belongs to plugin '{plugin}', "
            f"but expected plugin is '{expected_plugin}'"
        )
    return plugin, version


def read_plugin_version(plugin_dir: Path) -> str | None:
    """Read the version from a plugin's pyproject.toml."""
    pyproject = plugin_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text().split("[project]", 1)[-1],
        re.MULTILINE,
    )
    return match.group(1) if match else None


def read_plugin_init_version(plugin_dir: Path) -> str | None:
    """Read the version from a plugin's __init__.py."""
    init_file = plugin_dir / "src" / plugin_dir.name.replace("-", "_") / "__init__.py"
    if not init_file.is_file():
        return None
    match = re.search(r'version\s*=\s*"([^"]+)"', init_file.read_text())
    return match.group(1) if match else None


def read_recipe_version(recipe_path: Path) -> str | None:
    """Read context.version from a conda recipe.yaml."""
    if not recipe_path.is_file():
        return None
    for line in recipe_path.read_text().splitlines():
        m = re.match(r'^\s+version:\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    return None


def is_official_plugin(plugin_dir: Path) -> bool:
    """Check if a plugin directory declares official-plugin = true."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = plugin_dir / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text())
        tool_sc = data.get("tool", {}).get("spectrochempy", {})
        return tool_sc.get("official-plugin") is True
    except Exception:
        return False


def discover_official_plugins(plugins_dir: Path) -> list[str]:
    """Return sorted list of official plugin directory names."""
    results: list[str] = []
    if not plugins_dir.is_dir():
        return results
    for pyproject in sorted(plugins_dir.glob("spectrochempy-*/pyproject.toml")):
        if is_official_plugin(pyproject.parent):
            results.append(pyproject.parent.name)
    return results


def fetch_json(url: str, timeout: int = 30) -> dict | None:
    """Fetch JSON from a URL, returning None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def pypi_versions(package: str) -> list[str]:
    """Return all versions of a package from PyPI."""
    data = fetch_json(PYPI_JSON_URL.format(package=package))
    if data is None:
        return []
    return list(data.get("releases", {}).keys())


def pypi_latest_version(package: str) -> str | None:
    """Return the latest version of a package from PyPI."""
    data = fetch_json(PYPI_JSON_URL.format(package=package))
    if data is None:
        return None
    info = data.get("info", {})
    return info.get("version")


def anaconda_versions(
    package: str, owner: str = "spectrocat"
) -> dict[str, list[str]]:
    """Return {version: [labels]} for a package on Anaconda.org.

    Uses the package /files endpoint, which lists every uploaded file with
    its channel labels.  Returns empty dict if the package is not found.
    """
    data = fetch_json(ANACONDA_FILES_URL.format(owner=owner, package=package))
    if not isinstance(data, list):
        return {}
    result: dict[str, list[str]] = {}
    for f in data:
        ver = f.get("version")
        if not ver:
            continue
        labels = []
        for lbl in f.get("labels") or []:
            if isinstance(lbl, str):
                labels.append(lbl)
            elif isinstance(lbl, dict):
                labels.append(lbl.get("name", ""))
        for label in labels:
            if label not in result.setdefault(ver, []):
                result[ver].append(label)
    return result


def anaconda_version_labels(
    package: str, version: str, owner: str = "spectrocat"
) -> list[str]:
    """Return the labels for a specific version of a package on Anaconda.org."""
    return anaconda_versions(package, owner).get(version, [])


def git_tag_exists(tag: str, cwd: str | Path = ".") -> bool:
    """Check if a git tag exists in the repository.

    Checks the remote first; falls back to local tags if no remote is configured.
    """
    # Try remote first (works in CI with origin configured)
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode == 0 and tag in result.stdout:
        return True

    # Fallback: check local tags
    result = subprocess.run(
        ["git", "tag", "--list", tag],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.returncode == 0 and tag in result.stdout


def git_checkout_tag(tag: str, cwd: str | Path = ".") -> bool:
    """Checkout a specific git tag. Returns True on success."""
    result = subprocess.run(
        ["git", "checkout", tag],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.returncode == 0


def run_git(*args: str, cwd: str | Path = ".") -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Verification data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginReleaseCheck:
    """Result of checking a single plugin release across all registries."""

    plugin: str
    version: str
    github_release: bool
    pypi_version: str | None
    conda_main: bool
    conda_dev: bool
    conda_labels: list[str] = field(default_factory=list)
    verdict: str = ""


def check_plugin_release_consistency(
    plugin: str,
    version: str,
    *,
    skip_network: bool = False,
) -> PluginReleaseCheck:
    """Check consistency of a plugin release across GitHub / PyPI / Conda.

    Returns a PluginReleaseCheck with the verdict.
    """
    github_release = git_tag_exists(f"{plugin}-v{version}")

    pypi_versions_list: list[str] = []
    conda_versions_map: dict[str, list[str]] = {}
    conda_labels: list[str] = []

    if not skip_network:
        pypi_versions_list = pypi_versions(plugin)
        conda_versions_map = anaconda_versions(plugin)
        if version in conda_versions_map:
            conda_labels = conda_versions_map[version]

    conda_main = "main" in conda_labels
    conda_dev = "dev" in conda_labels

    pypi_present = version in pypi_versions_list
    pypi_version = version if pypi_present else None

    if not github_release:
        verdict = "missing_github_release"
    elif not skip_network and not pypi_present:
        verdict = "pypi_missing"
    elif conda_main:
        verdict = "aligned"
    elif conda_dev:
        verdict = "conda_dev_only"
    else:
        verdict = "conda_missing"

    return PluginReleaseCheck(
        plugin=plugin,
        version=version,
        github_release=github_release,
        pypi_version=pypi_version,
        conda_main=conda_main,
        conda_dev=conda_dev,
        conda_labels=conda_labels,
        verdict=verdict,
    )


def format_verification_report(checks: list[PluginReleaseCheck]) -> str:
    """Format a list of PluginReleaseCheck into a readable report."""
    lines = [
        "## Plugin release consistency report",
        "",
        "| Plugin | Version | GitHub | PyPI | Conda `main` | Conda `dev` | Verdict |",
        "|--------|---------|--------|------|--------------|-------------|---------|",
    ]
    for c in checks:
        gh = "yes" if c.github_release else "no"
        pypi = c.pypi_version or "not found"
        main = "yes" if c.conda_main else "no"
        dev = "yes" if c.conda_dev else "no"
        verdict_map = {
            "aligned": ":white_check_mark: aligned",
            "conda_missing": ":x: stable Conda missing",
            "conda_dev_only": ":warning: dev-only (no stable)",
            "pypi_missing": ":x: PyPI version missing",
            "missing_github_release": ":x: GitHub release missing",
        }
        verdict = verdict_map.get(c.verdict, c.verdict)
        lines.append(
            f"| {c.plugin} | {c.version} | {gh} | {pypi} | {main} | {dev} | {verdict} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # verify-tag
    p_tag = sub.add_parser("verify-tag", help="Validate a plugin tag format")
    p_tag.add_argument("tag", help="Tag to validate (e.g. spectrochempy-nmr-v0.1.1)")
    p_tag.add_argument("--plugin", help="Expected plugin name")

    # check-release
    p_check = sub.add_parser(
        "check-release", help="Check consistency of a plugin release"
    )
    p_check.add_argument("plugin", help="Plugin name (e.g. spectrochempy-nmr)")
    p_check.add_argument("version", help="Version to check (e.g. 0.1.11)")
    p_check.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network queries (PyPI / Anaconda.org)",
    )
    p_check.add_argument(
        "--json", action="store_true", dest="as_json", help="Output JSON"
    )

    # check-all
    p_all = sub.add_parser(
        "check-all", help="Check consistency for all official plugins"
    )
    p_all.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network queries (PyPI / Anaconda.org)",
    )
    p_all.add_argument(
        "--json", action="store_true", dest="as_json", help="Output JSON"
    )

    # list-official
    sub.add_parser("list-official", help="List official plugin names")

    # is-official
    p_off = sub.add_parser(
        "is-official",
        help=(
            "Exit 0 if a plugin directory is declared official "
            "([tool.spectrochempy] official-plugin = true), exit 1 otherwise, "
            "exit 2 if the marker cannot be evaluated"
        ),
    )
    p_off.add_argument(
        "plugin_dir", help="Path to plugin directory (e.g. plugins/spectrochempy-nmr)"
    )

    return parser.parse_args()


def cmd_verify_tag(args: argparse.Namespace) -> int:
    try:
        plugin, version = validate_tag_format(args.tag, args.plugin)
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f"Valid plugin tag: plugin={plugin}, version={version}")
    return 0


def cmd_check_release(args: argparse.Namespace) -> int:
    check = check_plugin_release_consistency(
        args.plugin, args.version, skip_network=args.skip_network
    )
    if args.as_json:
        print(json.dumps(asdict(check), indent=2))
    else:
        print(format_verification_report([check]))
    return 0 if check.verdict == "aligned" else 1


def cmd_check_all(args: argparse.Namespace) -> int:
    plugins_dir = Path("plugins")
    official = discover_official_plugins(plugins_dir)
    checks: list[PluginReleaseCheck] = []
    for plugin in official:
        version = read_plugin_version(plugins_dir / plugin)
        if not version:
            continue
        checks.append(
            check_plugin_release_consistency(
                plugin, version, skip_network=args.skip_network
            )
        )
    if args.as_json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(format_verification_report(checks))
    return 0 if all(c.verdict == "aligned" for c in checks) else 1


def cmd_list_official(_args: argparse.Namespace) -> int:
    plugins = discover_official_plugins(Path("plugins"))
    for p in plugins:
        print(p)
    return 0


def cmd_is_official(args: argparse.Namespace) -> int:
    try:
        official = is_official_plugin(Path(args.plugin_dir))
    except Exception as exc:
        print(f"::error::Could not evaluate official marker: {exc}", file=sys.stderr)
        return 2
    return 0 if official else 1


def main() -> int:
    args = parse_args()
    if args.command == "verify-tag":
        return cmd_verify_tag(args)
    elif args.command == "check-release":
        return cmd_check_release(args)
    elif args.command == "check-all":
        return cmd_check_all(args)
    elif args.command == "list-official":
        return cmd_list_official(args)
    elif args.command == "is-official":
        return cmd_is_official(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
