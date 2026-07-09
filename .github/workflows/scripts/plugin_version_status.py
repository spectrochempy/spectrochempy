#!/usr/bin/env python3
# ruff: noqa: S603, T201
"""Report official plugin release status and derive plugin dev versions."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

OFFICIAL_CLASSIFIER = "Framework :: SpectroChemPy :: Official Plugin"


def _discover_official_plugins() -> tuple[str, ...]:
    """Return official plugin directory names by checking classifier in pyproject.toml."""
    plugins_dir = Path("plugins")
    if not plugins_dir.is_dir():
        return ()

    results: list[str] = []
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    for pyproject in sorted(plugins_dir.glob("spectrochempy-*/pyproject.toml")):
        try:
            data = tomllib.loads(pyproject.read_text())
            classifiers = data.get("project", {}).get("classifiers", [])
            if OFFICIAL_CLASSIFIER in classifiers:
                results.append(pyproject.parent.name)
        except Exception as exc:
            print(f"Warning: could not read {pyproject}: {exc}", file=sys.stderr)
            continue
    return tuple(results)


OFFICIAL_PLUGINS = _discover_official_plugins()

TAG_RE = re.compile(
    r"^(?P<plugin>spectrochempy-[a-z0-9-]+)-v" r"(?P<version>\d+\.\d+\.\d+)$"
)


@dataclass(frozen=True)
class PluginVersionStatus:
    plugin: str
    plugin_dir: str
    latest_tag: str
    base_version: str
    commits_since_tag: int
    dev_version: str
    changed_files: int
    has_changes: bool


def release_relevant_paths(plugin_dir: Path) -> list[str]:
    """Return plugin paths that can affect a published package."""
    candidates = [
        plugin_dir / "src",
        plugin_dir / "pyproject.toml",
        plugin_dir / "recipe.yaml",
        plugin_dir / "meta.yaml",
        plugin_dir / "README.md",
        plugin_dir / "LICENSE",
        plugin_dir / "MANIFEST.in",
    ]
    return [path.as_posix() for path in candidates if path.exists()]


def run_git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_tag(tag: str) -> tuple[str, str] | None:
    match = TAG_RE.match(tag)
    if not match:
        return None
    return match.group("plugin"), match.group("version")


def version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def next_patch_dev_version(base_version: str, commits_since_tag: int) -> str:
    major, minor, patch = version_key(base_version)
    return f"{major}.{minor}.{patch + 1}.dev{commits_since_tag}"


def latest_plugin_tag(plugin: str) -> tuple[str, str]:
    tags = []
    for tag in run_git("tag", "--list", f"{plugin}-v*").splitlines():
        parsed = parse_tag(tag)
        if parsed and parsed[0] == plugin:
            tags.append((parsed[1], tag))
    if not tags:
        return "", read_pyproject_version(plugin) or "0.0.0"
    version, tag = sorted(tags, key=lambda item: version_key(item[0]))[-1]
    return tag, version


def read_pyproject_version(plugin: str) -> str:
    pyproject = Path("plugins") / plugin / "pyproject.toml"
    if not pyproject.exists():
        return ""
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text().split("[project]", 1)[-1],
        re.MULTILINE,
    )
    return match.group(1) if match else ""


def count_commits_since_tag(tag: str, paths: list[str]) -> int:
    if not paths:
        return 0
    rev_range = f"{tag}..HEAD" if tag else "HEAD"
    output = run_git("rev-list", "--count", rev_range, "--", *paths)
    return int(output or "0")


def changed_release_files(tag: str, paths: list[str]) -> list[str]:
    if not paths:
        return []
    if not tag:
        output = run_git("ls-files", *paths)
    else:
        output = run_git(
            "diff",
            "--ignore-cr-at-eol",
            "--name-only",
            tag,
            "--",
            *paths,
        )
    return [line for line in output.splitlines() if line]


def plugin_status(plugin: str) -> PluginVersionStatus:
    plugin_dir = Path("plugins") / plugin
    latest_tag, base_version = latest_plugin_tag(plugin)
    paths = release_relevant_paths(plugin_dir)
    changed_files = changed_release_files(latest_tag, paths)
    commits_since_tag = count_commits_since_tag(latest_tag, changed_files)
    return PluginVersionStatus(
        plugin=plugin,
        plugin_dir=plugin_dir.as_posix(),
        latest_tag=latest_tag,
        base_version=base_version,
        commits_since_tag=commits_since_tag,
        dev_version=next_patch_dev_version(base_version, commits_since_tag),
        changed_files=len(changed_files),
        has_changes=bool(changed_files),
    )


def update_file(path: Path, pattern: str, replacement: str) -> bool:
    if not path.exists():
        return False
    with open(path, newline="") as f:
        text = f.read()
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not update version in {path}")
    with open(path, "w", newline="") as f:
        f.write(new_text)
    return True


def strip_official_classifier(text: str) -> tuple[str, bool]:
    """Remove the OFFICIAL_CLASSIFIER line from pyproject.toml content.

    Also cleans up the now-empty classifier list and extra blank lines.
    """
    count = 0
    text, n = re.subn(
        rf'^[ \t]*"{re.escape(OFFICIAL_CLASSIFIER)}"[ \t]*,?[ \t]*\n?',
        "",
        text,
        flags=re.MULTILINE,
    )
    count += n
    text, n = re.subn(r"^classifiers = \[\s*\]\s*\n?", "", text, flags=re.MULTILINE)
    count += n
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, count > 0


def apply_dev_version(status: PluginVersionStatus) -> list[str]:
    plugin_dir = Path(status.plugin_dir)
    package_dir = status.plugin.replace("-", "_")
    version = status.dev_version
    changed = []

    pyproject = plugin_dir / "pyproject.toml"
    if update_file(pyproject, r'^(version\s*=\s*)"[^"]+"', rf'\g<1>"{version}"'):
        changed.append(pyproject.as_posix())

    # Strip the official classifier for dev versions so TestPyPI doesn't
    # reject it (the classifier is only registered on production PyPI).
    text = pyproject.read_text()
    stripped_text, did_strip = strip_official_classifier(text)
    if did_strip:
        pyproject.write_text(stripped_text)
        if pyproject.as_posix() not in changed:
            changed.append(pyproject.as_posix())

    recipe = plugin_dir / "recipe.yaml"
    if update_file(
        recipe,
        r'^(\s+version\s*:\s*)"[^"]+"',
        rf'\g<1>"{version}"',
    ):
        changed.append(recipe.as_posix())

    init_file = plugin_dir / "src" / package_dir / "__init__.py"
    if update_file(
        init_file,
        r'^( {4}version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
    ):
        changed.append(init_file.as_posix())

    return changed


def format_summary(statuses: list[PluginVersionStatus]) -> str:
    lines = [
        "## Plugin release status",
        "",
        "| Plugin | Status | Last tag | Commits | Changed files | Dev version |",
        "|--------|--------|----------|---------|---------------|-------------|",
    ]
    any_changes = False

    for status in statuses:
        tag = f"`{status.latest_tag}`" if status.latest_tag else "--"
        dev_version = f"`{status.dev_version}`"
        if not Path(status.plugin_dir).is_dir():
            lines.append(
                f"| {status.plugin} | :warning: directory missing | -- | -- | -- | -- |"
            )
            any_changes = True
        elif not status.latest_tag:
            lines.append(
                f"| {status.plugin} | :new: no previous plugin tag | -- | "
                f"{status.commits_since_tag} | {status.changed_files} | {dev_version} |"
            )
            any_changes = True
        elif status.has_changes:
            lines.append(
                f"| {status.plugin} | :large_orange_diamond: modified since last tag | "
                f"{tag} | {status.commits_since_tag} | {status.changed_files} | {dev_version} |"
            )
            any_changes = True
        else:
            lines.append(
                f"| {status.plugin} | :white_check_mark: unchanged | {tag} | "
                f"{status.commits_since_tag} | 0 | {dev_version} |"
            )

    lines.append("")
    if any_changes:
        lines.append(
            "_Some plugins have changes since their last release. Consider releasing them._"
        )
    else:
        lines.append(
            "**No official plugin changes detected since their last release tags.**"
        )
    return "\n".join(lines)


def write_github_output(status: PluginVersionStatus, changed_paths: list[str]) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as f:
        f.write(f"plugin={status.plugin}\n")
        f.write(f"latest_tag={status.latest_tag}\n")
        f.write(f"base_version={status.base_version}\n")
        f.write(f"commits_since_tag={status.commits_since_tag}\n")
        f.write(f"dev_version={status.dev_version}\n")
        f.write(f"has_changes={str(status.has_changes).lower()}\n")
        f.write(f"changed_paths={' '.join(changed_paths)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    plugin_group = parser.add_mutually_exclusive_group(required=True)
    plugin_group.add_argument(
        "--plugin", help="Plugin package name, e.g. spectrochempy-nmr"
    )
    plugin_group.add_argument(
        "--all-official",
        action="store_true",
        help="Report all official monorepo plugins",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of Markdown"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write Markdown report to GitHub step summary when available",
    )
    parser.add_argument(
        "--apply-dev-version",
        action="store_true",
        help="Temporarily write the computed dev version to plugin metadata files",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write single-plugin status fields to $GITHUB_OUTPUT",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plugins = list(OFFICIAL_PLUGINS) if args.all_official else [args.plugin]
    statuses = [plugin_status(plugin) for plugin in plugins]

    changed_paths: list[str] = []
    if args.apply_dev_version:
        if len(statuses) != 1:
            print(
                "::error::--apply-dev-version requires exactly one --plugin",
                file=sys.stderr,
            )
            return 2
        changed_paths = apply_dev_version(statuses[0])
        print(
            f"Applied {statuses[0].plugin} development version "
            f"{statuses[0].dev_version} to: {', '.join(changed_paths)}"
        )

    if args.github_output:
        if len(statuses) != 1:
            print(
                "::error::--github-output requires exactly one --plugin",
                file=sys.stderr,
            )
            return 2
        write_github_output(statuses[0], changed_paths)

    if args.json:
        payload = [asdict(status) for status in statuses]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))
    else:
        report = format_summary(statuses)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY") if args.summary else None
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(report + "\n")
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
