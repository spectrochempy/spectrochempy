---
name: spectrochempy-release
description: Prepare a SpectroChemPy release without publishing. Use to consolidate changelog, verify version consistency, and produce a release-ready branch. Does not push or publish.
metadata:
  audience: maintainer
  workflow: release
---

## Trigger

Use this skill when preparing a release. This skill prepares the branch only
— it does not push, open PRs, create releases, or publish packages unless
explicitly delegated.

---

## Inputs

* Target version (e.g. `1.2.0`) or instruction to determine it from the
  changelog.
* Confirmation of release type (major / minor / patch) if not obvious.

---

## Procedure

### 1. Verify changelog completeness

Read `docs/sources/whatsnew/changelog.rst`:

* All entries since the last release are present.
* Entries explain what changed and why it matters.
* No implementation-journal style entries (those belong in audit notes).
* No duplicate or near-duplicate entries for related work.

If `latest.rst` is stale, regenerate it by running the project's changelog
generation workflow (pre-commit hook or doc build tooling).

### 2. Check version consistency

The project uses `setuptools-scm` with `dynamic = ["version"]` in
`pyproject.toml`. The version is derived from git tags, not hardcoded.

* Check the latest tag: `git describe --tags`
* Check the version in `src/spectrochempy/__init__.py` matches.
* Check the version in `docs/sources/whatsnew/` changelog headers matches.

### 3. Verify CI is green on the current branch

```bash
gh pr checks <PR_NUMBER> --json name,state
```

Check the CI status once for the exact commit. Do not poll CI in a loop.

### 4. Check open blockers

* No open issues or PRs marked as blockers for this release.
* No failing tests on the target branch.
* No uncommitted changes.

### 5. Consolidate if needed

If the changelog has many small entries for a single feature, propose a
consolidated entry. Never edit `latest.rst` directly.

### 6. Produce the release branch plan

Prepare (but do not create unless delegated):

```
release/<version>
```

The branch name encodes the version. The `release/` prefix triggers the
publication workflow, so only create this branch when the maintainer is ready
to publish.

---

## Deliverable

Produce a release readiness report:

1. **Version** — proposed version and justification (major/minor/patch).
2. **Changelog** — summary of entries, consolidated form if needed.
3. **Blockers** — any open issues, failing tests, or inconsistencies.
4. **Next steps** — commands to run / decisions needed before publication.

---

## Validation

The preparation is complete when:
* changelog is consistent and complete;
* version is consistent across all locations;
* blockers are identified and reported;
* no push or publication has occurred without explicit delegation.
