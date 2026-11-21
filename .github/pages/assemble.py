"""Assemble a persistent GitHub Pages tree from an untrusted HTML artifact."""

import argparse
import html
import json
import posixpath
import re
import shutil
from pathlib import Path

VERSION = re.compile(r"(?:spectrochempy(?:-nmr)?-)?v(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?")


def version_key(value):
    match = VERSION.fullmatch(value)
    return (*map(int, match.group(1, 2, 3)), int(match[4]) if match[4] else 10**9)


def assemble(
    site,
    source,
    target,
    *,
    remove=False,
    archive_only=False,
    core_url=None,
    nmr_url=None,
):
    if (
        target != "latest"
        and not VERSION.fullmatch(target)
        and not re.fullmatch(r"previews/pr-[1-9]\d*", target)
    ):
        raise ValueError("Invalid publication target")
    if site.is_symlink() or any(path.is_symlink() for path in site.rglob("*")):
        raise ValueError("The stored site must not contain symbolic links")
    destination = site / target
    if not remove:
        if not source.is_dir() or source.is_symlink():
            raise ValueError("Missing HTML artifact")
        for path in source.rglob("*"):
            if path.is_symlink() or ".git" in path.relative_to(source).parts:
                raise ValueError("Unsafe artifact path")
            if not path.is_file() and not path.is_dir():
                raise ValueError("Unsupported artifact file")
        for name in ("index.html", "objects.inv", "search.html", "searchindex.js"):
            if not (source / name).is_file():
                raise ValueError(f"Missing site entry point: {name}")
    if archive_only:
        return
    if destination.exists():
        shutil.rmtree(destination)
    if not remove:
        shutil.copytree(source, destination)
        for page in destination.rglob("*.html"):
            relative = posixpath.relpath(
                "pages-navigation.js", page.relative_to(site).parent.as_posix()
            )
            injection = (
                f'<script defer src="{html.escape(relative, quote=True)}"></script>'
            )
            if target.startswith("previews/"):
                injection += '<meta name="robots" content="noindex,nofollow">'
            content = page.read_text(encoding="utf-8")
            for old, new in (
                ("https://www.spectrochempy.fr/projects/nmr/latest/", nmr_url),
                ("https://spectrochempy-nmr.readthedocs.io/latest/", nmr_url),
                ("https://www.spectrochempy.fr/latest/", core_url),
            ):
                if new:
                    content = content.replace(old, html.escape(new.rstrip("/") + "/"))
            page.write_text(
                content.replace("</head>", injection + "</head>", 1), encoding="utf-8"
            )
    versions = sorted(
        (
            path.name
            for path in site.iterdir()
            if path.is_dir() and VERSION.fullmatch(path.name)
        ),
        key=version_key,
        reverse=True,
    )
    stable = next(
        (value for value in versions if VERSION.fullmatch(value)[4] is None), None
    )
    keep = {stable} if stable else set()
    for value in versions:
        if value not in keep:
            shutil.rmtree(site / value)
    stable_path = site / "stable"
    if stable_path.exists():
        shutil.rmtree(stable_path)
    if stable:
        stable_path.mkdir()
        (stable_path / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>Stable documentation</title>"
            f'<meta http-equiv="refresh" content="0;url=../{stable}/">'
            f'<a href="../{stable}/">Stable documentation</a></html>',
            encoding="utf-8",
        )
    entries = [{"name": f"Stable ({stable})", "path": "stable"}] if stable else []
    if (site / "latest/index.html").exists():
        entries.append({"name": "Development", "path": "latest"})
    entries.extend(
        {"name": value, "path": value} for value in versions if value in keep
    )
    (site / "versions.json").write_text(
        json.dumps({"versions": entries, "stable": stable}), encoding="utf-8"
    )
    landing = stable or ("latest" if (site / "latest/index.html").exists() else None)
    if landing:
        (site / "index.html").write_text(
            f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Documentation</title>'
            f'<meta http-equiv="refresh" content="0;url={landing}/">'
            f'<a href="{landing}/">Documentation</a></html>',
            encoding="utf-8",
        )
    (site / ".nojekyll").touch()
    files = [
        path for path in site.rglob("*") if path.is_file() and ".git" not in path.parts
    ]
    if any(path.stat().st_size >= 95_000_000 for path in files):
        raise ValueError("A Pages file exceeds the publication size budget")
    if sum(path.stat().st_size for path in files) >= 950_000_000:
        raise ValueError("The Pages site exceeds the publication size budget")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--core-url")
    parser.add_argument("--nmr-url")
    args = parser.parse_args()
    assemble(
        args.site,
        args.source,
        args.target,
        remove=args.remove,
        archive_only=args.archive_only,
        core_url=args.core_url,
        nmr_url=args.nmr_url,
    )
