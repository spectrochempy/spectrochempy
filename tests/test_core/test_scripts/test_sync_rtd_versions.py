"""Regression checks for Core RTD release retention."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[3] / ".github" / "workflows" / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "sync_rtd_versions", SCRIPTS / "sync_rtd_versions.py"
    )
    rtd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtd)
finally:
    sys.path.remove(str(SCRIPTS))


def version(
    name, *, slug=None, kind="branch", active=True, built=True, hidden=False, ref=None
):
    return {
        "slug": slug or name,
        "verbose_name": name,
        "type": kind,
        "active": active,
        "built": built,
        "hidden": hidden,
        "ref": ref,
    }


TAGS = ["spectrochempy-v1.8.0", "spectrochempy-v1.9.0", "spectrochempy-v1.10.0"]
PROJECT = {"default_version": "latest"}


def test_semantic_order_and_core_release_filter():
    items = [version(n) for n in ("1.8.0", "1.9.0", "1.10.0")]
    items += [
        version(n, kind="tag")
        for n in ("spectrochempy-v2.0.0rc1", "spectrochempy-iris-v8.0.0")
    ]
    items += [version("latest"), version("stable"), version("999", kind="external")]
    plan = rtd.retention_plan(
        items, TAGS + [i["verbose_name"] for i in items[-5:]], PROJECT, 2
    )
    assert plan == {
        "keep": ["1.10.0", "1.9.0"],
        "activate": [],
        "deactivate": ["1.8.0"],
    }


def test_draft_to_published_transition_requires_the_new_tag():
    items = [version(n) for n in ("1.8.0", "1.9.0", "1.10.0", "1.11.0")]
    with pytest.raises(ValueError, match="newer release"):
        rtd.retention_plan(items, TAGS, PROJECT, 2)
    plan = rtd.retention_plan(items, [*TAGS, "spectrochempy-v1.11.0"], PROJECT, 2)
    assert plan["keep"] == ["1.11.0", "1.10.0"]
    assert plan["deactivate"] == ["1.8.0", "1.9.0"]


def test_existing_active_tag_url_is_kept_instead_of_inactive_archive():
    items = [
        version("1.8.0"),
        version("1.9.0", active=False),
        version("spectrochempy-v1.9.0", kind="tag"),
        version("1.10.0"),
    ]
    assert rtd.retention_plan(items, TAGS, PROJECT, 2)["keep"] == [
        "1.10.0",
        "spectrochempy-v1.9.0",
    ]


def test_duplicate_tag_and_hidden_archive_are_reconciled():
    items = [
        version("1.8.0"),
        version("1.9.0"),
        version("spectrochempy-v1.9.0", kind="tag"),
        version("1.10.0", hidden=True),
    ]
    plan = rtd.retention_plan(items, TAGS, PROJECT, 2)
    assert plan["activate"] == ["1.10.0"]
    assert plan["deactivate"] == ["1.8.0", "spectrochempy-v1.9.0"]


@pytest.mark.parametrize("default", ["1.8.0", "latest"])
def test_default_and_stable_target_are_protected(default):
    items = [version(n) for n in ("1.8.0", "1.9.0", "1.10.0")]
    items.append(version("stable", ref="1.8.0" if default == "latest" else None))
    with pytest.raises(ValueError, match="outside the retention window"):
        rtd.retention_plan(items, TAGS, {"default_version": default}, 2)


@pytest.mark.parametrize(
    "tags,items,match",
    [
        (TAGS[:1], [version("1.8.0")], "Incomplete"),
        (TAGS, [version("1.8.0")], "not yet registered"),
        (
            TAGS,
            [version(n) for n in ("1.8.0", "1.9.0", "1.10.0", "2.0.0")],
            "newer release",
        ),
    ],
)
def test_incomplete_or_stale_inventory_never_prunes(tags, items, match):
    with pytest.raises(ValueError, match=match):
        rtd.retention_plan(items, tags, PROJECT, 2)


class FakeClient:
    def __init__(self, items, fail_activation=False):
        self.items = deepcopy(items)
        self.calls = []
        self.fail_activation = fail_activation

    def versions(self):
        return deepcopy(self.items)

    def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if method == "GET":
            return PROJECT
        if method == "PATCH":
            if self.fail_activation and data.get("active"):
                raise RuntimeError("activation failed")
            slug = path.split("/")[1]
            next(item for item in self.items if item["slug"] == slug).update(data)
        return None


def test_dry_run_is_read_only_and_apply_is_idempotent():
    client = FakeClient(
        [version("1.8.0"), version("1.9.0"), version("1.10.0", active=False)]
    )
    rtd.synchronize(client, TAGS, keep=2)
    assert all(method == "GET" for method, _, _ in client.calls)
    client.calls.clear()
    rtd.synchronize(client, TAGS, apply=True, keep=2)
    patches = [(path, data) for method, path, data in client.calls if method == "PATCH"]
    assert patches == [
        ("versions/1.10.0/", {"active": True, "hidden": False}),
        ("versions/1.8.0/", {"active": False}),
    ]
    client.calls.clear()
    rtd.synchronize(client, TAGS, apply=True, keep=2)
    assert not any(method == "PATCH" for method, _, _ in client.calls)


def test_failed_activation_prevents_all_deactivations():
    client = FakeClient(
        [version("1.8.0"), version("1.9.0"), version("1.10.0", active=False)],
        fail_activation=True,
    )
    with pytest.raises(RuntimeError, match="activation failed"):
        rtd.synchronize(client, TAGS, apply=True, keep=2)
    assert not any(data == {"active": False} for _, _, data in client.calls)


def test_missing_version_is_resynchronized_before_any_changes(monkeypatch):
    client = FakeClient([version("1.8.0"), version("1.9.0")])
    monkeypatch.setattr(
        rtd.time,
        "sleep",
        lambda _: client.items.append(version("1.10.0", active=False)),
    )
    rtd.synchronize(client, TAGS, apply=True, keep=2)
    assert client.calls[0] == ("POST", "sync-versions/", None)
    assert next(i for i in client.items if i["slug"] == "1.10.0")["active"]


def test_pagination_collects_all_versions_and_rejects_external_urls(monkeypatch):
    client = rtd.Client("test-token")
    responses = iter(
        [
            {
                "count": 2,
                "results": [1],
                "next": "/api/v3/projects/spectrochempy/versions/?offset=100",
            },
            {"count": 2, "results": [2], "next": None},
        ]
    )
    monkeypatch.setattr(client, "request", lambda *_: next(responses))
    assert client.versions() == [1, 2]
    monkeypatch.setattr(
        client,
        "request",
        lambda *_: {"count": 0, "results": [], "next": "https://example.com/versions/"},
    )
    with pytest.raises(ValueError, match="pagination leaves"):
        client.versions()


def test_api_rejects_cross_project_requests_before_sending_credentials():
    client = rtd.Client("test-token")
    with pytest.raises(ValueError, match="leaves the Core project"):
        client.request("GET", "/api/v3/projects/other/versions/")


def test_truncated_inventory_is_rejected(monkeypatch):
    client = rtd.Client("test-token")
    monkeypatch.setattr(
        client, "request", lambda *_: {"count": 2, "results": [1], "next": None}
    )
    with pytest.raises(RuntimeError, match="Incomplete RTD"):
        client.versions()


def test_api_redirects_are_not_followed():
    with pytest.raises(RuntimeError, match="Unexpected RTD API redirect"):
        rtd.NoRedirects().redirect_request(
            None, None, 302, "redirect", {}, "https://example.com"
        )
