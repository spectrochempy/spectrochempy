# ruff: noqa: S310, S603, S607, T201
"""Select verified public GitHub results without exposing an Actions token to RTD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.request import urlopen

REPOSITORY = "spectrochempy/spectrochempy"
API = f"https://api.github.com/repos/{REPOSITORY}"
WORKFLOW = ".github/workflows/docs-artifact-prototype.yml"
GITHUB_ACTIONS_APP = 15368
KINDS = ("tutorials", "gallery")


def public_mode():
    return os.environ.get("SCPY_RTD_GITHUB_RESULTS") == "1" and (
        os.environ.get("READTHEDOCS_VERSION") == "latest"
        or os.environ.get("READTHEDOCS_VERSION_TYPE") == "external"
    )


def selected():
    return public_mode() or (
        os.environ.get("SCPY_RTD_DOCS_ARTIFACTS") == "1"
        and os.environ.get("READTHEDOCS_VERSION_TYPE") != "external"
    )


def check_name(project):
    return f"RTD prepared inputs: {project}"


def _json(path):
    request = Request(
        f"{API}/{path}",
        headers={"Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def storage_url(url):
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not hostname.endswith(
            (".blob.core.windows.net", ".actions.githubusercontent.com")
        )
    ):
        raise ValueError("Unexpected GitHub artifact storage URL")
    return url


def envelope(commit):
    project = os.environ.get("READTHEDOCS_PROJECT", "spectrochempy")
    query = urlencode({"check_name": check_name(project), "per_page": 100})
    checks = _json(f"commits/{commit}/check-runs?{query}")["check_runs"]
    for check in sorted(checks, key=lambda item: item["id"], reverse=True):
        if (
            check.get("name") != check_name(project)
            or check.get("head_sha") != commit
            or check.get("app", {}).get("id") != GITHUB_ACTIONS_APP
            or check.get("conclusion") != "success"
        ):
            continue
        value = json.loads(check["output"]["text"])
        if value.get("commit") != commit or value.get("project") != project:
            raise RuntimeError("The prepared inputs do not match this RTD checkout")
        if value.get("expires_at", 0) <= time.time() + 5:
            continue
        if set(value.get("artifacts", {})) != set(KINDS):
            raise RuntimeError("Incomplete prepared input pair")
        return value
    return None


def verified_inputs(commit):
    value = envelope(commit)
    if value is None:
        return None
    run_id = int(value["run_id"])
    run = _json(f"actions/runs/{run_id}")
    if (
        run["head_sha"] != commit
        or run.get("repository", {}).get("full_name") != REPOSITORY
        or run.get("path") != WORKFLOW
        or run.get("conclusion") != "success"
        or run.get("event") not in {"push", "pull_request", "workflow_dispatch"}
    ):
        raise RuntimeError("The prepared inputs come from an unexpected workflow run")
    artifacts = {
        item["id"]: item
        for item in _json(f"actions/runs/{run_id}/artifacts?per_page=100")["artifacts"]
        if not item["expired"]
    }
    result = {}
    for kind, entry in value["artifacts"].items():
        artifact = artifacts.get(entry["id"])
        if (
            artifact is None
            or artifact["name"] != f"docs-{kind}-{commit}"
            or artifact.get("digest") != entry["digest"]
        ):
            raise RuntimeError(
                "Prepared input identity or checksum differs from GitHub"
            )
        result[kind] = {**artifact, "transfer_url": storage_url(entry["url"])}
    return run_id, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("selected", "ready"))
    args = parser.parse_args()
    if args.operation == "selected":
        return 0 if selected() else 1
    if not public_mode():
        return 0
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True
    ).strip()
    if envelope(commit) is None:
        print(
            "GitHub is preparing the documentation; RTD will be triggered when ready."
        )
        return 183
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
