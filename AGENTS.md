# SpectroChemPy — Agent Guide

## Project

SpectroChemPy (SCPy) is a framework for processing, analyzing and modeling spectroscopic data.

Key characteristics:

- Python >= 3.11
- src/ layout
- setuptools + setuptools-scm
- versioning from git tags
- monorepo containing core package and official plugins

---

## General Principles

- Prefer minimal and targeted changes.
- Preserve backward compatibility whenever possible.
- Avoid unrelated refactoring.
- Keep pull requests focused on a single objective.
- Inspect existing code before proposing architectural changes.
- Prefer maintainability over short-term fixes.

---

## Branching Policy

Do not work directly on `master`.

Use this workflow for all new work:

1. keep local `master` aligned with `upstream/master`;
2. create a dedicated development branch for the chantier;
3. commit and push work on that branch only;
4. open the pull request from the development branch to `upstream/master`;
5. after merge, resync local `master` and only then start the next branch.

Practical rules:

- treat local `master` as a clean mirror of `upstream/master`;
- for direct commits to `master`, push to `upstream/master`; `origin/master`
  may be used only as a backup mirror;
- do not accumulate chantier commits on `master`;
- prefer branch names derived from the chantier, for example:
  - `coordset-storage-pr5`
  - `analysis-pca-cleanup`
  - `dev/<topic>` if a generic prefix is useful;
- if a PR needs follow-up fixes, continue on the same PR branch rather than on
  `master`.
- temporary PR branches may be pushed to `upstream` and optionally to `origin`
  as backup, but delete them promptly after the PR is merged:
  - resync local `master` with `upstream/master`;
  - delete the local temporary branch;
  - delete the matching temporary branch on `upstream`;
  - delete the matching temporary branch on `origin` if it exists.
- keep only long-lived branches that have an explicit ongoing purpose.

This reduces divergence between:

- local `master`;
- `origin/master`;
- `upstream/master`.

It also makes post-merge cleanup and the start of the next chantier simpler.

---

## Resource Usage and Sobriety

Be conservative with:

- token usage;
- execution time;
- CI consumption;
- test execution.

Prefer:

- static analysis;
- source inspection;
- targeted validation;
- small reproducible checks.

Avoid:

- full test suite execution;
- coverage runs;
- unnecessary CI runs;
- unnecessary pushes;
- repeated validation cycles.

When a potentially expensive validation is required:

- run only the smallest validation necessary;
- avoid unrelated tests;
- avoid triggering CI unnecessarily.

---

## Development Environment

Use the smallest suitable environment for the task.

For core development, prefer a dedicated core-only environment.

Example:

```bash
conda create -n scpy-core python=3.12
conda activate scpy-core
pip install -e ".[dev]"
```

The environment name `scpy-core` is only a convention. Contributors may use a different name.

For plugin-specific work, install only the plugin required by the task.

Example:

```bash
pip install -e plugins/spectrochempy-nmr
```

Avoid installing all plugins unless explicitly required.

---

## Testing Policy

Run only the smallest validation necessary for the current task.

Examples:

```bash
pytest tests/test_analysis/test_decomposition/test_pca.py
```

```bash
pytest tests/test_analysis/test_decomposition/test_pca.py::TestPCA::test_fit
```

Avoid:

```bash
pytest tests
```

unless explicitly requested.

Useful markers:

- slow
- network
- serial
- docs
- plugin
- data

External test data are optional and tests should continue to skip gracefully when unavailable.

Prefer understanding the code before executing tests.

Avoid repeated cycles of:

```text
test -> fail -> small fix -> retest
```

Inspect first, then validate.

---

## Linting and Formatting

Use Ruff as configured by the repository, but do not run Ruff routinely during
intermediate work.

Prefer source inspection and targeted tests while developing. Run Ruff only:

- when explicitly requested;
- when investigating a formatting/lint failure that blocks the current task;
- near PR completion, normally through the final pre-commit pass.

Avoid targeted Ruff commands on changed files during normal iteration unless
there is a clear, immediate need. They can be unexpectedly expensive in this
repository and are already covered by pre-commit before finalization.

Examples:

```bash
ruff check src/spectrochempy/
```

```bash
ruff check --fix src/spectrochempy/
ruff format src/spectrochempy/
```

Follow project configuration from `pyproject.toml`.

---

## Pre-commit Policy

Do not run:

```bash
pre-commit run --all-files
```

during intermediate work.

Run pre-commit only:

- before the final commit;
- before opening a pull request;
- when explicitly requested.

In normal workflows this should happen only once near PR completion.

If pre-commit modifies files, include those modifications in the final commit.
Generated files updated by pre-commit hooks (e.g. ``latest.rst``) belong in
the final PR state — do not discard them after pre-commit runs.

---

## Changelog Policy

Entry file: `docs/sources/whatsnew/changelog.rst`. Never edit `latest.rst`.

User-facing changes (New Features, Bug Fixes, Dependency Updates, Breaking Changes, Deprecations) go without prefix.

Developer section entries require a prefix:

| Prefix    | Usage                                |
|-----------|--------------------------------------|
| `FEATURE` | New dev-facing capability            |
| `FIX`     | Test, CI, or internal bug fix        |
| `MAINT`   | Refactoring, cleanup                 |
| `CI`      | CI/CD workflow changes               |
| `DEV`     | Developer tooling (bump scripts, …)  |

See `docs/sources/devguide/contributing_codebase.rst` for the full reference.

---

## Audit Policy

For substantial work:

- create or update a file in `audit/`;
- prefix filenames with `~`;
- audit files must remain untracked;
- audit files must never be included in a PR.

Example:

```text
audit/~analysis phase5 - pca modernization.md
```

Audit files should contain:

- objective;
- inspected files;
- modified files;
- rationale;
- validation commands;
- results;
- remaining concerns;
- recommendations for the next phase.

Whenever producing a substantial report, analysis or audit, write it to an audit file rather than keeping it only in the chat.

---

## Commit Policy

Commit only when:

- work is complete;
- validation succeeds;
- changes are coherent.

Avoid unnecessary intermediate commits.

Prefer a single coherent commit per completed phase.

---

## Pull Request Policy

Unless explicitly requested:

- do not merge;
- do not alter release workflows;
- do not alter publication workflows.

PR titles should use the same prefix style as commit messages:

| Prefix     | Usage                              |
|------------|------------------------------------|
| `MAINT:`   | Refactoring, cleanup               |
| `FIX:`     | Bug fix (including test/CI fixes)  |
| `DOC:`     | Documentation changes              |
| `TEST:`    | Test addition or change            |
| `CI:`      | CI/CD workflow changes             |
| `DEV:`     | Developer tooling                   |
| `FEATURE:` | New feature                        |

At the end of a chantier:

- prepare a clean final commit;
- prepare a concise PR description using the same prefix as the PR title;
- update changelog when appropriate;
- run pre-commit before finalizing the PR.

---

## Plugin Architecture

Plugins are independent packages located under:

```text
plugins/
```

Examples:

- spectrochempy-nmr
- spectrochempy-iris
- spectrochempy-hypercomplex
- spectrochempy-carroucell

Keep core/plugin separation intact.

Do not move plugin functionality into the core package without explicit justification.

---

## Test Ownership

Plugin-specific tests should generally live in plugin repositories.

Core repository tests should focus on:

- plugin discovery;
- plugin loading;
- compatibility;
- missing-plugin user experience.

Avoid detailed plugin algorithm testing in the core repository.

---

## Generated Files

Do not edit manually:

```text
requirements/*.txt
src/spectrochempy/__init__.pyi
docs/sources/reference/generated/*
docs/sources/whatsnew/latest.rst
```

These files are generated.

---

## Documentation

Use NumPyDoc conventions.

Respect existing documentation structure and validation rules.

---

## Repository Structure

Main locations:

```text
src/spectrochempy/    Core package
tests/                Test suite
plugins/              Official plugins
docs/                 Documentation
audit/                Local audit files (untracked)
```
