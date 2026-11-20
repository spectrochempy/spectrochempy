"""Keep the ten latest Core releases visible on Read the Docs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler
from urllib.request import Request
from urllib.request import build_opener

from packaging.version import Version
from release_version import stable_core_version_from_tag

API = "https://app.readthedocs.org/api/v3/projects/spectrochempy/"
ALIASES = {"latest", "stable"}


def release_number(name):
    """Recognize canonical Core tags and numeric release archive names."""
    tagged = stable_core_version_from_tag(name)
    if tagged is not None:
        return Version(tagged)
    if isinstance(name, str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", name):
        parsed = Version(name)
        if str(parsed) == name:
            return parsed
    return None


def retention_plan(versions, tags, project, keep=10):
    """Plan changes without renaming versions or changing other projects."""
    if keep < 1:
        raise ValueError("The retained release count must be positive")
    releases = {number for tag in tags if (number := release_number(tag)) is not None}
    if len(releases) < keep:
        raise ValueError("Incomplete release tag history; refusing to prune versions")
    wanted = sorted(releases, reverse=True)[:keep]
    by_number = defaultdict(list)
    by_slug = {}
    for item in versions:
        slug = item["slug"]
        if slug in by_slug:
            raise ValueError(f"Duplicate RTD slug: {slug}")
        by_slug[slug] = item
        if slug in ALIASES or item["type"] not in {"branch", "tag"}:
            continue
        number = release_number(item["verbose_name"])
        if number is None:
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug):
            raise ValueError("Unexpected release URL identifier from RTD")
        if number > max(releases):
            raise ValueError("RTD has a newer release than the local tag inventory")
        if number in releases:
            by_number[number].append(item)
    missing = [str(number) for number in wanted if not by_number[number]]
    if missing:
        raise ValueError(
            f"Release versions not yet registered on RTD: {', '.join(missing)}"
        )
    protected = ALIASES | {project["default_version"]}
    stable = by_slug.get("stable")
    if stable and stable.get("ref"):
        protected.add(stable["ref"])
    chosen = []
    for number in wanted:
        candidates = by_number[number]
        pinned = [item for item in candidates if item["slug"] in protected]
        if len(pinned) > 1:
            raise ValueError(
                "Multiple protected URLs for one release; review RTD aliases"
            )
        candidate = min(
            pinned or candidates,
            key=lambda item: (
                not item["active"],
                not item["built"],
                item["type"] != "branch",
                item["slug"] != str(number),
                item["slug"],
            ),
        )
        chosen.append(candidate)
    selected = {item["slug"] for item in chosen}
    deactivate = sorted(
        item["slug"]
        for candidates in by_number.values()
        for item in candidates
        if item["active"] and item["slug"] not in selected
    )
    if protected.intersection(deactivate):
        raise ValueError("The default/stable target is outside the retention window")
    return {
        "keep": [item["slug"] for item in chosen],
        "activate": [
            item["slug"] for item in chosen if not item["active"] or item["hidden"]
        ],
        "deactivate": deactivate,
    }


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Unexpected RTD API redirect")


class Client:
    def __init__(self, token):
        self.token = token
        self.opener = build_opener(NoRedirects())

    def request(self, method, path, data=None):
        url = urljoin(API, path)
        if not url.startswith(API):
            raise ValueError("RTD API URL leaves the Core project")
        request = Request(  # noqa: S310
            url,
            method=method,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        request.add_unredirected_header("Authorization", f"Token {self.token}")
        try:
            with self.opener.open(request, timeout=30) as response:
                body = response.read()
        except HTTPError as error:
            raise RuntimeError(
                f"RTD API {method} failed with HTTP {error.code}"
            ) from None
        return json.loads(body) if body else None

    def versions(self):
        path = "versions/?limit=100"
        seen = set()
        result = []
        count = None
        while path:
            if path in seen:
                raise RuntimeError("Repeated RTD pagination URL")
            seen.add(path)
            page = self.request("GET", path)
            if count is not None and count != page["count"]:
                raise RuntimeError("RTD version inventory changed during pagination")
            count = page["count"]
            result.extend(page["results"])
            path = page["next"]
            if path:
                target = urlsplit(urljoin(API, path))
                if target._replace(query="", fragment="").geturl() != API + "versions/":
                    raise ValueError("RTD pagination leaves the version listing")
        if len(result) != count:
            raise RuntimeError("Incomplete RTD version inventory")
        return result


def synchronize(client, tags, *, apply=False, keep=10):
    if apply:
        client.request("POST", "sync-versions/")
    for attempt in range(7 if apply else 1):
        project = client.request("GET", "")
        versions = client.versions()
        try:
            plan = retention_plan(versions, tags, project, keep)
        except ValueError as error:
            if (
                not str(error).startswith("Release versions not yet registered")
                or attempt == 6
                or not apply
            ):
                raise
            time.sleep(10)
        else:
            break
    sys.stdout.write(json.dumps(plan, indent=2) + "\n")
    if not apply:
        return plan
    for slug in plan["activate"]:
        client.request(
            "PATCH",
            f"versions/{quote(slug, safe='')}/",
            {"active": True, "hidden": False},
        )
    current = client.versions()
    verified = retention_plan(current, tags, client.request("GET", ""), keep)
    if verified["keep"] != plan["keep"] or verified["activate"]:
        raise RuntimeError(
            "Retained versions changed or were not activated; no pruning performed"
        )
    for slug in verified["deactivate"]:
        client.request("PATCH", f"versions/{quote(slug, safe='')}/", {"active": False})
    final = retention_plan(client.versions(), tags, client.request("GET", ""), keep)
    if final["activate"] or final["deactivate"] or final["keep"] != plan["keep"]:
        raise RuntimeError("RTD retention verification failed")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="apply activation and deactivation"
    )
    args = parser.parse_args()
    token = os.environ.get("SCPY_RTD_API_TOKEN")
    if not token:
        parser.error("SCPY_RTD_API_TOKEN is required (Read the Docs API credential)")
    tags = subprocess.check_output(["git", "tag", "--list"], text=True).splitlines()  # noqa: S603, S607
    synchronize(Client(token), tags, apply=args.apply)


if __name__ == "__main__":
    main()
