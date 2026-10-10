# AGENTS.md

## Scope

This document defines permanent rules and authorization limits for AI-assisted
development in SpectroChemPy.

Procedural details live in the shared OpenCode, Codex, and Claude Code
skills in `.agents/skills/` and in `CONTRIBUTING.md`. Load the relevant skill
before starting work.

**Read `CONTRIBUTING.md` at the start of each session.**

When rules overlap, follow the stricter requirement.

---

## Core Principles

Priorities:

1. correctness;
2. behavior preservation;
3. maintainability;
4. reviewability;
5. resource efficiency.

Prefer small, reversible, reviewable changes. Avoid broad rewrites unless
explicitly requested.

---

## Public Behavior Preservation

Unless explicitly requested otherwise:

* preserve public APIs;
* preserve public behavior;
* preserve backward compatibility;
* preserve serialization compatibility;
* preserve warning behavior;
* preserve documented semantics.

Internal refactoring must not introduce user-visible behavior changes.

When adding a new public top-level API symbol (`spectrochempy.<name>` /
`scp.<name>`), also update `docs/sources/reference/index.rst` in the
appropriate section.

---

## Authorization Limits

The maintainer controls commits, branches, pushes, pull requests, releases,
and package publication.

Unless explicitly delegated:

* do not create branches;
* do not commit;
* do not push;
* do not open or merge pull requests;
* do not create releases or publish packages;
* do not run pre-commit during development (required only before pushing a PR
  branch — see `spectrochempy-dev` skill).
* do not run a broad or repeated rewriting pass (`ruff check --fix`,
  `ruff format`) while developing; read-only checks (`ruff check`,
  `ruff format --check`) on targeted paths are fine. The pre-push hooks remain
  mandatory: they own lint and formatting, they may rewrite code, so inspect
  the diff after any run that modifies files.

When a push is explicitly delegated, push to `origin` unless the maintainer
explicitly directs a push to `upstream`.

When a task is authorized, proceed to a validated solution and a first review
of the diff without requesting confirmation for ordinary technical choices.

## PR Base

Always base PRs on `upstream/master` unless explicitly instructed otherwise.
Before creating a new branch, fetch `upstream/master` and branch from it
directly (see `spectrochempy-dev` skill).

---

## Temporary Files

Scratch space is disposable by definition. It belongs in the temporary
directory of the current environment; its location, backing store and lifetime
are the platform's business, not this repository's. `tempfile` resolves the
location itself (`TMPDIR`, `TEMP`, `TMP`, platform defaults), so no path is
hardcoded.

* Repository code uses context-managed temporary directories
  (`tempfile.TemporaryDirectory()`) or the pytest `tmp_path` fixture.
* Remove scratch space in the task that created it, and report what was left
  behind.
* Keeping a result and committing it are different acts: a report or a note
  worth keeping belongs in a repository, while a build tree, an environment or
  a dataset may stay in a persistent non-versioned location.

---

## Skills

Load the relevant skill before starting work:

| Skill | When to load |
|---|---|
| `spectrochempy-dev` | Any development, refactoring, or bug fix |
| `spectrochempy-review` | Separate review in a new session (see skill) |
| `spectrochempy-release` | Preparing a release (without publishing) |

The skills are shared by OpenCode, Codex, and Claude Code. They live only in
`.agents/skills/`; Claude Code reaches them through relative symlinks in
`.claude/skills/`.

<!-- On Windows checkouts, these relative symlinks need Developer Mode (or
admin rights) and `git config core.symlinks true`; otherwise Git materializes
them as plain text files and Claude Code cannot discover the skills. -->

---

## Repository Map

* `.` — Main library (this repository)
* `../spectrochempy_maintainer/` — Private maintainer governance, RFCs, audits,
  roadmap (separate clone, see its own `AGENTS.md`)
* `../spectrochempy_assistant/` — Companion assistant application
* `../spectrochempy_data/` — Reference datasets

---

## References

* `CONTRIBUTING.md` — Commit/PR prefixes, full PR workflow, developer guide
* `docs/sources/devguide/` — Full developer documentation
* `maintainers/` — Release and emergency recovery procedures only
