# ruff: noqa: S603, S310, T201
"""Build on Read the Docs, optionally consuming GitHub Actions artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen
from zipfile import ZipFile

from docs.build_diagnostics import _diagnostics
from docs.build_diagnostics import _phase

if __name__ == "__main__":
    with _diagnostics(), _phase("import artifact consumer dependencies"):
        from docs import ci_artifacts
else:
    from docs import ci_artifacts

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


def _find_artifacts(commit, token, unavailable):
    query = urlencode({"head_sha": commit, "per_page": 100})
    runs = _json(f"workflows/{WORKFLOW}/runs?{query}", token)["workflow_runs"]
    runs = [
        run
        for run in runs
        if run["head_sha"] == commit and run["event"] in ("push", "workflow_dispatch")
    ]
    names = {f"docs-{kind}-{commit}": kind for kind in ci_artifacts.KINDS}
    for run in runs:
        if run["conclusion"] != "success":
            continue
        artifacts = _json(f"runs/{run['id']}/artifacts?per_page=100", token)[
            "artifacts"
        ]
        matches = {
            names[item["name"]]: item
            for item in artifacts
            if not item["expired"]
            and item["name"] in names
            and item["id"] not in unavailable
        }
        if matches.keys() == set(ci_artifacts.KINDS):
            return run["id"], matches
    if runs and all(run["status"] == "completed" for run in runs):
        latest = max(runs, key=lambda run: run["id"])
        if latest["conclusion"] != "success":
            raise RuntimeError(
                f"Documentation artifact production for {commit} ended with "
                f"{latest['conclusion']}; no usable artifact pair is available. "
                f"See https://github.com/spectrochempy/spectrochempy/actions/runs/{latest['id']}"
            )
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
    unavailable = set()
    while True:
        with _phase(f"find artifact pair for {commit}"):
            selection = _find_artifacts(commit, token, unavailable)
        if selection is not None:
            run_id, artifacts = selection
            with tempfile.TemporaryDirectory(prefix="scp-rtd-artifacts-") as directory:
                root = Path(directory)
                for kind, artifact in artifacts.items():
                    try:
                        with _phase(
                            f"download and extract {kind} artifact {artifact['id']} from run {run_id}"
                        ):
                            _download(artifact, token, root / kind)
                    except HTTPError as error:
                        if error.code != 410:
                            raise
                        unavailable.add(artifact["id"])
                        break
                else:
                    print(
                        f"Checking CI artifact pair for {commit} from run {run_id}: "
                        f"tutorials={artifacts['tutorials']['id']}, "
                        f"gallery={artifacts['gallery']['id']}",
                        flush=True,
                    )
                    try:
                        ci_artifacts.render(root)
                    except ci_artifacts.ArtifactVersionMismatchError as error:
                        print(f"Skipping run {run_id}: {error}", flush=True)
                        unavailable.update(
                            artifact["id"] for artifact in artifacts.values()
                        )
                    else:
                        return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"No complete CI artifact pair found for {commit}")
        print(f"Waiting for documentation artifacts for {commit}", flush=True)
        if selection is None:
            time.sleep(30)


def main():
    token = os.environ.pop("SCPY_RTD_GITHUB_TOKEN", None)
    if not token:
        raise RuntimeError("SCPY_RTD_GITHUB_TOKEN is required in artifact mode")
    with _diagnostics(), _phase("RTD artifact build"):
        _render_from_actions(token)


if __name__ == "__main__":
    main()
