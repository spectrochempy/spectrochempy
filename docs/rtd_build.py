# ruff: noqa: S603, S310, T201
"""Build on Read the Docs, optionally consuming GitHub Actions artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen
from zipfile import ZipFile

from docs import ci_artifacts
from docs import preexecute_notebooks as notebooks

API = "https://api.github.com/repos/spectrochempy/spectrochempy/actions"
WORKFLOW = "docs-artifact-prototype.yml"


def _request(path, token):
    request = Request(
        f"{API}/{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    # urllib copies normal headers to redirected requests. Keep the token
    # on the GitHub API request only, away from the signed storage URL.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urlopen(request, timeout=60)


def _json(path, token):
    with _request(path, token) as response:
        return json.load(response)


def _find_artifacts(commit, token):
    query = urlencode({"head_sha": commit, "status": "success", "per_page": 100})
    runs = _json(f"workflows/{WORKFLOW}/runs?{query}", token)["workflow_runs"]
    names = {f"docs-{kind}-{commit}": kind for kind in ci_artifacts.KINDS}
    for run in runs:
        if run["head_sha"] != commit or run["event"] not in (
            "push",
            "workflow_dispatch",
        ):
            continue
        artifacts = _json(f"runs/{run['id']}/artifacts?per_page=100", token)[
            "artifacts"
        ]
        matches = {
            names[item["name"]]: item
            for item in artifacts
            if not item["expired"] and item["name"] in names
        }
        if matches.keys() == set(ci_artifacts.KINDS):
            return matches
    return None


def _download(artifact, token, destination):
    archive = destination.with_suffix(".zip")
    with (
        _request(f"artifacts/{int(artifact['id'])}/zip", token) as response,
        archive.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if artifact.get("digest") != f"sha256:{digest}":
        raise RuntimeError("GitHub Actions artifact ZIP checksum mismatch")
    with ZipFile(archive) as bundle:
        bundle.extractall(destination)


def _render_from_actions(token):
    commit = ci_artifacts._commit()
    deadline = time.monotonic() + 900
    while (artifacts := _find_artifacts(commit, token)) is None:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"No complete CI artifact pair found for {commit}")
        print(f"Waiting for documentation artifacts for {commit}", flush=True)
        time.sleep(30)
    with tempfile.TemporaryDirectory(prefix="scp-rtd-artifacts-") as directory:
        root = Path(directory)
        for kind, artifact in artifacts.items():
            _download(artifact, token, root / kind)
        ci_artifacts.render(root)


def _render_on_rtd():
    notebooks.prepare()
    manifest = json.loads(notebooks.MANIFEST.read_text(encoding="utf-8"))
    for name in manifest["notebooks"]:
        subprocess.run(
            [sys.executable, "-u", "-m", "docs.preexecute_notebooks", "execute", name],
            check=True,
        )
    notebooks.verify()
    ci_artifacts.render_html(gallery_from_artifact=False)


def main():
    enabled = os.environ.get("SCPY_RTD_DOCS_ARTIFACTS") == "1"
    external = os.environ.get("READTHEDOCS_VERSION_TYPE") == "external"
    token = os.environ.pop("SCPY_RTD_GITHUB_TOKEN", None)
    if enabled and not external:
        if not token:
            raise RuntimeError("SCPY_RTD_GITHUB_TOKEN is required in artifact mode")
        _render_from_actions(token)
    else:
        _render_on_rtd()


if __name__ == "__main__":
    main()
