#!/usr/bin/env python
"""
Install SpectroChemPy plugins from the local source tree.

Usage::

    python -m spectrochempy.ci.install_plugins --editable all
    python -m spectrochempy.ci.install_plugins nmr iris
    python -m spectrochempy.ci.install_plugins --editable perkinelmer
    python -m spectrochempy.ci.install_plugins --list-names
    python -m spectrochempy.ci.install_plugins --list-names --root /workspace
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_root() -> Path:
    """Discover repository root by walking up from cwd or script location."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in [start, *start.parents]:
            if (candidate / "plugins").is_dir() and list(
                (candidate / "plugins").glob("spectrochempy-*/pyproject.toml")
            ):
                return candidate
    msg = "Cannot find repository root (no plugins/ with spectrochempy-* subdirs)"
    raise SystemExit(msg)


def _discover_plugins(plugins_dir: Path, *, official_only: bool = False) -> list[str]:
    """Return sorted list of plugin directory names, optionally filtered by official marker."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    plugins: list[str] = []
    for p in sorted(plugins_dir.glob("spectrochempy-*/pyproject.toml")):
        name = p.parent.name
        if official_only:
            try:
                data = tomllib.loads(p.read_text())
                tool_sc = data.get("tool", {}).get("spectrochempy", {})
                if tool_sc.get("official-plugin") is not True:
                    continue
            except Exception:
                continue
        plugins.append(name)
    return plugins


def _pip_install(
    plugin: str,
    plugins_dir: Path,
    *,
    editable: bool = False,
    pip_cmd: list[str] | None = None,
    no_deps: bool = False,
    no_build_isolation: bool = False,
) -> int:
    plugin_path = plugins_dir / plugin
    if not plugin_path.is_dir():
        print(f"Error: plugin directory not found: {plugin_path}", file=sys.stderr)
        return 1

    pip = pip_cmd or [sys.executable, "-m", "pip"]
    cmd = [*pip, "install"]
    if editable:
        cmd.append("-e")
    if no_deps:
        cmd.append("--no-deps")
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    cmd.append(str(plugin_path))

    print(f"Installing {plugin} {'(editable)' if editable else ''}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def _resolve_name(name: str) -> str:
    if name.startswith("spectrochempy-"):
        return name
    return f"spectrochempy-{name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install or list SpectroChemPy plugins from the local source tree.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root (auto-detected by default).",
    )
    parser.add_argument(
        "--editable",
        "-e",
        action="store_true",
        help="Install in editable/development mode (pip install -e).",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Pass --no-deps to pip.",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="Pass --no-build-isolation to pip.",
    )
    parser.add_argument(
        "--pip",
        default=None,
        help="Pip command (default: python -m pip). Example: --pip 'uv pip'",
    )
    parser.add_argument(
        "--list-names",
        action="store_true",
        help="Print official plugin directory names (one per line) and exit.",
    )
    parser.add_argument(
        "plugins",
        nargs="*",
        default=["all"],
        metavar="PLUGIN",
        help="Plugin name(s) to install (e.g. nmr, iris). Default: all (official plugins).",
    )
    parsed = parser.parse_args()

    root = Path(parsed.root) if parsed.root else _find_root()
    plugins_dir = root / "plugins"

    if parsed.list_names:
        for name in _discover_plugins(plugins_dir, official_only=True):
            print(name)
        return 0

    pip_cmd = parsed.pip.split() if parsed.pip else None

    if "all" in parsed.plugins:
        targets = _discover_plugins(plugins_dir, official_only=True)
        if not targets:
            print("Error: no official plugins discovered.", file=sys.stderr)
            return 1
    else:
        all_known = _discover_plugins(plugins_dir, official_only=False)
        targets = [_resolve_name(p) for p in parsed.plugins]
        for t in targets:
            if t not in all_known:
                print(f"Warning: unknown plugin {t!r}, skipping", file=sys.stderr)

    errors = 0
    for plugin in targets:
        errors += _pip_install(
            plugin,
            plugins_dir,
            editable=parsed.editable,
            pip_cmd=pip_cmd,
            no_deps=parsed.no_deps,
            no_build_isolation=parsed.no_build_isolation,
        )

    if errors:
        print(f"\n{errors} plugin(s) failed to install.", file=sys.stderr)
        return errors
    print("\nAll plugins installed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
