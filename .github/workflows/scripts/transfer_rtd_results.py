# ruff: noqa: S310, T201
"""Hand public artifact download links to RTD from a trusted workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler
from urllib.request import Request
from urllib.request import build_opener

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from docs.rtd_transfer import API  # noqa: E402
from docs.rtd_transfer import KINDS  # noqa: E402
from docs.rtd_transfer import REPOSITORY  # noqa: E402
from docs.rtd_transfer import WORKFLOW  # noqa: E402
from docs.rtd_transfer import check_name  # noqa: E402
from docs.rtd_transfer import storage_url  # noqa: E402

RTD_API = "https://app.readthedocs.org/api/v3"
PROJECTS = {"spectrochempy", "spectrochempy-artifacts-test"}


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, token, method="GET", data=None, *, rtd=False):
    payload = json.dumps(data).encode() if data is not None else None
    req = Request(url, data=payload, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("Content-Type", "application/json")
    req.add_unredirected_header(
        "Authorization", f"{'Token' if rtd else 'Bearer'} {token}"
    )
    return build_opener(NoRedirects()).open(req, timeout=30)


def github(path, token, method="GET", data=None):
    with request(f"{API}/{path}", token, method, data) as response:
        return json.load(response)


def rtd(path, token, method="GET", data=None):
    with request(f"{RTD_API}/{path}", token, method, data, rtd=True) as response:
        return json.load(response)


def version_for(run, token):
    if (
        run.get("repository", {}).get("full_name") != REPOSITORY
        or run.get("path") != WORKFLOW
        or run.get("conclusion") != "success"
    ):
        return None
    commit = run["head_sha"]
    if run["event"] == "pull_request":
        pulls = github(f"commits/{commit}/pulls?per_page=100", token)
        matches = [
            pr
            for pr in pulls
            if pr["state"] == "open"
            and pr["head"]["sha"] == commit
            and pr["base"]["ref"] == "master"
            and pr["base"]["repo"]["full_name"] == REPOSITORY
        ]
        return str(matches[0]["number"]) if len(matches) == 1 else None
    if run["event"] in {"push", "workflow_dispatch"} and run["head_branch"] == "master":
        branch = github("branches/master", token)
        if branch["commit"]["sha"] == commit:
            return "latest"
    return None


def artifact_pair(run, token):
    items = github(f"actions/runs/{int(run['id'])}/artifacts?per_page=100", token)[
        "artifacts"
    ]
    result = {}
    for kind in KINDS:
        matches = [
            item
            for item in items
            if item["name"] == f"docs-{kind}-{run['head_sha']}" and not item["expired"]
        ]
        if len(matches) != 1 or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", matches[0].get("digest", "")
        ):
            raise RuntimeError("A verified documentation input artifact is missing")
        result[kind] = matches[0]
    return result


def signed_url(artifact_id, token):
    try:
        response = request(f"{API}/actions/artifacts/{int(artifact_id)}/zip", token)
    except HTTPError as error:
        if error.code != 302:
            raise
        return storage_url(error.headers["Location"])
    else:
        response.close()
        raise RuntimeError("GitHub did not return a temporary artifact URL")


def publish_links(run, artifacts, project, token, check_id=None):
    envelope = {
        "commit": run["head_sha"],
        "run_id": run["id"],
        "project": project,
        "expires_at": time.time() + 45,
        "artifacts": {
            kind: {
                "id": item["id"],
                "digest": item["digest"],
                "url": signed_url(item["id"], token),
            }
            for kind, item in artifacts.items()
        },
    }
    payload = {
        "name": check_name(project),
        "head_sha": run["head_sha"],
        "external_id": str(run["id"]),
        "status": "completed",
        "conclusion": "success",
        "details_url": run["html_url"],
        "output": {
            "title": "Prepared documentation inputs",
            "summary": "Temporary links to public documentation results; no GitHub credential is shared with RTD.",
            "text": json.dumps(envelope),
        },
    }
    if check_id is None:
        return github("check-runs", token, "POST", payload)["id"]
    payload.pop("head_sha")
    github(f"check-runs/{int(check_id)}", token, "PATCH", payload)
    return check_id


def transfer(run_id, project, github_token, rtd_token):
    if project not in PROJECTS:
        raise ValueError("Unexpected RTD project")
    run = github(f"actions/runs/{int(run_id)}", github_token)
    version = version_for(run, github_token)
    if version is None:
        print("Skipping an obsolete or unsupported documentation revision")
        return
    artifacts = artifact_pair(run, github_token)
    check_id = publish_links(run, artifacts, project, github_token)
    created = rtd(
        f"projects/{project}/versions/{version}/builds/", rtd_token, "POST", {}
    )
    build = created.get("build", created)
    build_id = int(build["id"])
    print(
        f"Triggered RTD build {build_id} for {version} at {run['head_sha']}", flush=True
    )
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        check_id = publish_links(run, artifacts, project, github_token, check_id)
        status = rtd(f"projects/{project}/builds/{build_id}/", rtd_token)
        state = status["state"]
        if isinstance(state, dict):
            state = state["code"]
        print(f"RTD build {build_id}: {state}", flush=True)
        if state in {"finished", "cancelled"}:
            if (
                state != "finished"
                or not status["success"]
                or status.get("commit") != run["head_sha"]
            ):
                raise RuntimeError(
                    f"RTD build {build_id} did not publish the expected revision"
                )
            return
        time.sleep(20)
    raise RuntimeError(
        f"RTD build {build_id} did not finish within the transfer job budget"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument(
        "--project", choices=sorted(PROJECTS), default="spectrochempy-artifacts-test"
    )
    args = parser.parse_args()
    transfer(
        args.run_id,
        args.project,
        os.environ["GITHUB_TOKEN"],
        os.environ["SCPY_RTD_API_TOKEN"],
    )


if __name__ == "__main__":
    main()
