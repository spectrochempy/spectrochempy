#!/usr/bin/env python3
# ruff: noqa: T201
"""
Shared helpers for Conda plugin publication and verification.

Provides functions for:
- Parsing and validating plugin tags
- Querying Anaconda.org for package versions and labels
- Verifying release consistency across GitHub / PyPI / Conda
- Building Conda packages from a specific git tag
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

TAG_RE = re.compile(
    r"^(?P<plugin>spectrochempy-[a-z0-9-]+)-v(?P<version>\d+\.\d+\.\d+)$"
)

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
ANACONDA_FILES_URL = "https://api.anaconda.org/package/{owner}/{package}/files"


class ServiceUnavailableError(RuntimeError):
    """
    Raised when a registry (PyPI / Anaconda.org) cannot be queried.

    Distinct from a genuine "not found": an HTTP 404 means the resource does
    not exist, while network errors, timeouts and 5xx mean the check cannot
    conclude.  Callers must not turn the latter into a "version missing".
    """


def parse_plugin_tag(tag: str) -> tuple[str, str] | None:
    """Parse a plugin tag into (plugin_name, version) or None if invalid."""
    match = TAG_RE.match(tag)
    if not match:
        return None
    return match.group("plugin"), match.group("version")


def validate_tag_format(
    tag: str, expected_plugin: str | None = None
) -> tuple[str, str]:
    """
    Validate a plugin tag and return (plugin_name, version).

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
    """
    Read the version from a conda recipe.

    Supports both the ``recipe.yaml`` form (``version: "0.1.1"`` under
    ``context:`` or ``package:``) and the legacy ``meta.yaml`` jinja form
    (``{% set version = "0.1.1" %}``).
    """
    if not recipe_path.is_file():
        return None
    text = recipe_path.read_text()
    match = re.search(r"{%\s*set\s+version\s*=\s*[\"']([^\"']+)[\"']\s*%}", text)
    if match:
        return match.group(1)
    match = re.search(
        r'^\s*version\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s#]+))',
        text,
        re.MULTILINE,
    )
    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return None


def is_official_plugin(plugin_dir: Path) -> bool:
    """
    Check if a plugin directory declares official-plugin = true.

    A *missing* pyproject.toml is treated as "not official".  A pyproject
    that cannot be read or parsed is an evaluation error and raises, so the
    caller can fail loudly instead of silently skipping the plugin.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = plugin_dir / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text())
    except Exception as exc:
        raise ValueError(f"cannot parse {pyproject}: {exc}") from exc
    tool_sc = data.get("tool", {}).get("spectrochempy", {})
    if not isinstance(tool_sc, dict):
        raise ValueError(f"invalid [tool.spectrochempy] table in {pyproject}")
    return tool_sc.get("official-plugin") is True


def discover_official_plugins(plugins_dir: Path) -> list[str]:
    """Return sorted list of official plugin directory names."""
    results: list[str] = []
    if not plugins_dir.is_dir():
        return results
    for pyproject in sorted(plugins_dir.glob("spectrochempy-*/pyproject.toml")):
        if is_official_plugin(pyproject.parent):
            results.append(pyproject.parent.name)
    return results


def fetch_json(url: str, timeout: int = 30) -> dict | list | None:
    """
    Fetch JSON from a URL.

    Returns None when the resource does not exist (HTTP 404) and raises
    :class:`ServiceUnavailableError` on any other failure (network error,
    timeout, HTTP 5xx, unreadable JSON), so that callers can distinguish a
    genuine "missing" package from an unavailable service.
    """
    try:
        # URLs are built from https:// API constants only (never user-supplied
        # or file:// schemes), so the urlopen audit is not applicable here.
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ServiceUnavailableError(f"{url}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise ServiceUnavailableError(f"{url}: {exc}") from exc


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


def anaconda_versions(package: str, owner: str = "spectrocat") -> dict[str, list[str]]:
    """
    Return {version: [labels]} for a package on Anaconda.org.

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


def list_plugin_tags(cwd: str | Path = ".") -> list[tuple[str, str]]:
    """
    Return all (plugin, version) pairs from git release tags.

    Reads the remote tags first and falls back to local tags when no remote
    is configured.  Annotated tag peel lines (``refs/tags/foo^{}``) are
    ignored.
    """
    # git is a trusted VCS binary (fixed args from the repo config, never
    # untrusted input), so the subprocess audit is not applicable.
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin"],  # noqa: S603, S607
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    refs = result.stdout if result.returncode == 0 else ""

    tags: set[str] = set()
    for line in refs.splitlines():
        if "^{}" in line:
            continue
        fields = line.split("\t")
        if len(fields) == 2 and fields[1].startswith("refs/tags/"):
            tags.add(fields[1].removeprefix("refs/tags/"))

    seen: set[tuple[str, str]] = set()
    parsed: list[tuple[str, str]] = []
    for tag in sorted(tags):
        match = parse_plugin_tag(tag)
        if match and match not in seen:
            seen.add(match)
            parsed.append(match)

    # The configured remote may not host every plugin tag (e.g. local work
    # against a fork); fall back to local tags whenever the remote yields no
    # plugin tags at all.
    if not parsed:
        result = subprocess.run(
            ["git", "tag", "--list"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode == 0:
            for tag in result.stdout.splitlines():
                match = parse_plugin_tag(tag.strip())
                if match and match not in seen:
                    seen.add(match)
                    parsed.append(match)

    parsed.sort(key=lambda pair: (pair[0], tuple(int(p) for p in pair[1].split("."))))
    return parsed


def git_tag_exists(tag: str, cwd: str | Path = ".") -> bool:
    """
    Check if a git tag exists in the repository.

    Checks the remote first; falls back to local tags if no remote is configured.
    """
    # git is a trusted VCS binary (fixed args from the repo config, never
    # untrusted input), so the subprocess audit is not applicable.  The tag is
    # validated earlier against the PEP 440 pattern.
    # Try remote first (works in CI with origin configured)
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],  # noqa: S603, S607
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode == 0 and tag in result.stdout:
        return True

    # Fallback: check local tags
    result = subprocess.run(
        ["git", "tag", "--list", tag],  # noqa: S603, S607
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.returncode == 0 and tag in result.stdout


# ---------------------------------------------------------------------------
# Version reading
# ---------------------------------------------------------------------------
# Verification data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginReleaseCheck:
    """Result of checking a single plugin release across all registries."""

    plugin: str
    version: str
    github_release: bool
    pypi_available: bool = True
    conda_available: bool = True
    pypi_version: str | None = None
    conda_main: bool = False
    conda_dev: bool = False
    conda_labels: list[str] = field(default_factory=list)
    verdict: str = ""


def check_plugin_release_consistency(
    plugin: str,
    version: str,
    *,
    skip_network: bool = False,
) -> PluginReleaseCheck:
    """
    Check consistency of a plugin release across GitHub / PyPI / Conda.

    Returns a PluginReleaseCheck with the verdict.  ``pypi_unavailable`` /
    ``conda_unavailable`` verdicts mean the registry could not be queried,
    so nothing can be concluded for that release.
    """
    github_release = git_tag_exists(f"{plugin}-v{version}")

    pypi_versions_list: list[str] = []
    conda_versions_map: dict[str, list[str]] = {}
    conda_labels: list[str] = []
    pypi_available = True
    conda_available = True

    if not skip_network:
        try:
            pypi_versions_list = pypi_versions(plugin)
        except ServiceUnavailableError:
            pypi_available = False
        try:
            conda_versions_map = anaconda_versions(plugin)
            if version in conda_versions_map:
                conda_labels = conda_versions_map[version]
        except ServiceUnavailableError:
            conda_available = False

    conda_main = "main" in conda_labels
    conda_dev = "dev" in conda_labels

    pypi_present = version in pypi_versions_list
    pypi_version = version if pypi_present else None

    if not github_release:
        verdict = "missing_github_release"
    elif not pypi_available:
        verdict = "pypi_unavailable"
    elif not conda_available:
        verdict = "conda_unavailable"
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
        pypi_available=pypi_available,
        conda_available=conda_available,
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
            "pypi_unavailable": ":warning: PyPI unavailable (cannot conclude)",
            "conda_unavailable": ":warning: Conda unavailable (cannot conclude)",
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

    # validate-release
    p_val = sub.add_parser(
        "validate-release",
        help=(
            "Validate that the versions declared by a plugin checkout "
            "(pyproject.toml, __init__.py, recipe) match the release tag version"
        ),
    )
    p_val.add_argument(
        "plugin_dir",
        help="Path to the plugin directory inside the exact tag checkout",
    )
    p_val.add_argument("version", help="Expected version from the release tag")

    # upload-conda
    p_up = sub.add_parser(
        "upload-conda",
        help=(
            "Safely upload a built Conda artifact to Anaconda.org.  Refuses to "
            "overwrite an already-published version unless --allow-override is passed."
        ),
    )
    p_up.add_argument("artifact", help="Path to the .conda / .tar.bz2 artifact")
    p_up.add_argument("--plugin", required=True, help="Expected plugin package name")
    p_up.add_argument("--version", required=True, help="Expected version")
    p_up.add_argument("--owner", default="spectrocat")
    p_up.add_argument("--label", default="main")
    p_up.add_argument("--token", default="", help="Defaults to ANACONDA_API_TOKEN")
    p_up.add_argument(
        "--allow-override",
        action="store_true",
        help="Explicit derogation: allow replacing an already-published version",
    )
    p_up.add_argument("--dry-run", action="store_true")
    p_up.add_argument("--no-verify", dest="verify", action="store_false")
    p_up.add_argument("--retries", type=int, default=8)
    p_up.add_argument("--delay", type=float, default=15.0)

    # verify-conda
    p_vc = sub.add_parser(
        "verify-conda",
        help="Poll the Anaconda /files API until a version carries a label",
    )
    p_vc.add_argument("plugin")
    p_vc.add_argument("version")
    p_vc.add_argument("--owner", default="spectrocat")
    p_vc.add_argument("--label", default="main")

    # conda-state
    p_cs = sub.add_parser(
        "conda-state",
        help=(
            "Report the labels of a version on Anaconda (0=found, 1=missing, "
            "2=service unavailable)"
        ),
    )
    p_cs.add_argument("plugin")
    p_cs.add_argument("version")
    p_cs.add_argument("--owner", default="spectrocat")

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
    """Check consistency for every official plugin release (all git tags)."""
    tags = list_plugin_tags()
    if tags:
        check_keys: list[tuple[str, str]] = tags
    else:
        # Fallback when not inside a git checkout: check the versions declared
        # by the current pyproject.toml files.
        plugins_dir = Path("plugins")
        check_keys = []
        for plugin in discover_official_plugins(plugins_dir):
            version = read_plugin_version(plugins_dir / plugin)
            if version:
                check_keys.append((plugin, version))
    checks: list[PluginReleaseCheck] = []
    for plugin, version in check_keys:
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


def cmd_validate_release(args: argparse.Namespace) -> int:
    """Validate declared versions (pyproject / __init__ / recipe) vs tag version."""
    plugin_dir = Path(args.plugin_dir)
    version = args.version
    try:
        errors: list[str] = []

        declared = read_plugin_version(plugin_dir)
        if declared is None:
            errors.append(f"no version found in {plugin_dir / 'pyproject.toml'}")
        elif declared != version:
            errors.append(
                f"pyproject.toml version ({declared}) != tag version ({version})"
            )

        init_version = read_plugin_init_version(plugin_dir)
        if init_version is not None and init_version != version:
            errors.append(
                f"__init__.py version ({init_version}) != tag version ({version})"
            )

        recipe_file = plugin_dir / "recipe.yaml"
        if not recipe_file.is_file():
            recipe_file = plugin_dir / "meta.yaml"
        if not recipe_file.is_file():
            errors.append(f"no recipe.yaml or meta.yaml found in {plugin_dir}")
        else:
            recipe_version = read_recipe_version(recipe_file)
            if recipe_version is None:
                errors.append(f"could not read the version from {recipe_file}")
            elif recipe_version != version:
                errors.append(
                    f"recipe version ({recipe_version}) != tag version ({version})"
                )
    except Exception as exc:
        print(f"::error::Could not validate release: {exc}", file=sys.stderr)
        return 2

    if errors:
        for err in errors:
            print(f"::error::{err}", file=sys.stderr)
        return 1

    print(f"Plugin {plugin_dir.name} releases consistently at version {version}")
    return 0


def artifact_matches(artifact: str | Path, plugin: str, version: str) -> bool:
    """Return True when the artifact basename starts with '<plugin>-<version>-'."""
    return Path(artifact).name.startswith(f"{plugin}-{version}-")


def verify_conda_upload(
    plugin: str,
    version: str,
    *,
    owner: str = "spectrocat",
    label: str = "main",
    retries: int = 8,
    delay: float = 15.0,
) -> bool:
    """Poll the Anaconda /files API until the version carries the label."""
    for attempt in range(1, retries + 1):
        try:
            labels = anaconda_version_labels(plugin, version, owner)
        except ServiceUnavailableError:
            labels = []
        if label in labels:
            print(
                f"Verified: {plugin}=={version} is on Anaconda '{label}' (labels: {labels})"
            )
            return True
        if attempt < retries:
            print(
                f"  (attempt {attempt}/{retries}: not yet on '{label}', retrying in {delay:.0f}s)"
            )
            time.sleep(delay)
    print(
        f"::error::Timed out waiting for {plugin}=={version} on the '{label}' label",
        file=sys.stderr,
    )
    return False


def upload_conda(
    artifact: str | Path,
    *,
    plugin: str,
    version: str,
    owner: str = "spectrocat",
    label: str = "main",
    token: str = "",
    allow_override: bool = False,
    dry_run: bool = False,
    verify: bool = True,
    retries: int = 8,
    delay: float = 15.0,
) -> int:
    """
    Safely upload a built Conda artifact to Anaconda.org.  Returns an exit code.

    Guards:
    - the artifact basename must match ``<plugin>-<version>-`` exactly;
    - the version must not already exist on the target label unless the
      operator explicitly passes ``allow_override`` (documented derogation);
    - ``anaconda upload`` is called without ``--force`` except in that case;
    - after the upload, the /files API is polled until the label appears.
    """
    artifact_path = Path(artifact)
    if not artifact_path.is_file():
        print(f"::error::Artifact not found: {artifact}", file=sys.stderr)
        return 1
    if not artifact_matches(artifact_path, plugin, version):
        print(
            f"::error::Artifact '{artifact_path.name}' does not match {plugin}=={version}",
            file=sys.stderr,
        )
        return 1

    if not token:
        token = os.environ.get("ANACONDA_API_TOKEN", "")
    if not token and not dry_run:
        print("::error::ANACONDA_API_TOKEN is not set. Cannot upload.", file=sys.stderr)
        return 1

    try:
        existing_labels = anaconda_version_labels(plugin, version, owner)
    except ServiceUnavailableError as exc:
        print(
            f"::error::Could not check Anaconda for an existing package: {exc}",
            file=sys.stderr,
        )
        return 2

    if label in existing_labels and not allow_override:
        print(
            f"::error::{plugin}=={version} already exists on the '{label}' label "
            f"(labels: {existing_labels}). Use --allow-override (explicit "
            "derogation) to replace it.",
            file=sys.stderr,
        )
        return 1

    if dry_run:
        action = "re-upload (override)" if label in existing_labels else "upload"
        print(
            f"[dry-run] would {action}: {artifact_path.name} -> {owner} label '{label}'"
        )
        return 0

    command = ["anaconda", "--token", token, "upload"]
    if allow_override and label in existing_labels:
        command.append("--force")
    command += ["-u", owner, "-l", label, str(artifact_path)]
    print(f"Uploading {artifact_path.name} to {owner} label '{label}'")
    # 'anaconda' is the pinned anaconda-client CLI installed in the job; the
    # token and label come from the workflow, not from untrusted input.
    result = subprocess.run(command)  # noqa: S603, S607
    if result.returncode != 0:
        print(
            f"::error::anaconda upload failed (exit {result.returncode})",
            file=sys.stderr,
        )
        return result.returncode

    if verify and not verify_conda_upload(
        plugin, version, owner=owner, label=label, retries=retries, delay=delay
    ):
        return 1
    if not verify:
        print(f"Uploaded {artifact_path.name} to {owner} label '{label}'")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    return upload_conda(
        args.artifact,
        plugin=args.plugin,
        version=args.version,
        owner=args.owner,
        label=args.label,
        token=args.token,
        allow_override=args.allow_override,
        dry_run=args.dry_run,
        verify=args.verify,
        retries=args.retries,
        delay=args.delay,
    )


def cmd_verify_conda(args: argparse.Namespace) -> int:
    if verify_conda_upload(
        args.plugin, args.version, owner=args.owner, label=args.label
    ):
        return 0
    return 1


def cmd_conda_state(args: argparse.Namespace) -> int:
    try:
        labels = anaconda_version_labels(args.plugin, args.version, args.owner)
    except ServiceUnavailableError as exc:
        print(f"service_unavailable: {exc}", file=sys.stderr)
        return 2
    if not labels:
        print(f"{args.plugin}=={args.version} not found on any label")
        return 1
    print(f"{args.plugin}=={args.version} found on labels: {':'.join(sorted(labels))}")
    return 0


def main() -> int:
    args = parse_args()
    commands = {
        "verify-tag": cmd_verify_tag,
        "check-release": cmd_check_release,
        "check-all": cmd_check_all,
        "list-official": cmd_list_official,
        "is-official": cmd_is_official,
        "validate-release": cmd_validate_release,
        "upload-conda": cmd_upload,
        "verify-conda": cmd_verify_conda,
        "conda-state": cmd_conda_state,
    }
    return commands.get(args.command, lambda _: 1)(args)


if __name__ == "__main__":
    raise SystemExit(main())
