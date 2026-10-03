# AGENTS.md

## Scope

This document defines permanent rules and authorization limits for AI-assisted
development in SpectroChemPy.

Procedural details live in the project's OpenCode skills and in
`CONTRIBUTING.md`. Load the relevant skill before starting work.

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
* do not run `ruff` or `ruff format` directly while developing, not even on the
  files being edited: the hooks own lint and formatting, since a standalone run
  may rewrite code that was never reviewed. If the hooks cannot be installed,
  report that instead of substituting a direct run.

When a task is authorized, proceed to a validated solution and a first review
of the diff without requesting confirmation for ordinary technical choices.

## PR Base

Always base PRs on `upstream/master` unless explicitly instructed otherwise.
Before creating a new branch, fetch `upstream/master` and branch from it
directly (see `spectrochempy-dev` skill).

---

## Temporary Files

`/tmp` is a shared, size-limited resource (a `tmpfs`, i.e. RAM). What is left
there is invisible to the repositories and does not survive a reboot, so it can
never hold the only copy of a result.

* Repository code uses context-managed temporary directories
  (`tempfile.TemporaryDirectory()`) or the pytest `tmp_path` fixture, never a
  hardcoded `/tmp/...` path.
* Scratch space created while working — build trees, documentation output,
  throwaway virtual environments, validation copies — goes under
  `/tmp/opencode/<job>.XXXXXX`, where the `mktemp` suffix marks it disposable.
  Create `/tmp/opencode` first when it is missing: `mktemp` does not create
  intermediate directories.
* Remove that scratch space in the same task that created it
  (`trap 'rm -rf "$work"' EXIT`), and report the directories left behind.
* Anything that must outlive the task belongs in the repository, under an
  explicit name — never in `/tmp`. `maintainers/` stays restricted to release
  and emergency recovery procedures.

---

## Skills

Load the relevant skill before starting work:

| Skill | When to load |
|---|---|
| `spectrochempy-dev` | Any development, refactoring, or bug fix |
| `spectrochempy-review` | Separate review in a new session (see skill) |
| `spectrochempy-release` | Preparing a release (without publishing) |

**For Codex:** skills are in `.opencode/skills/` — read the relevant `SKILL.md` directly.

---

## Repository Map

* `spectrochempy/` — Main library (this repository)
* `spectrochempy_maintainer/` — Private maintainer governance, RFCs, audits,
  roadmap (separate clone, see its own `AGENTS.md`)
* `spectrochempy_assistant/` — Companion assistant application
* `spectrochempy_data/` — Reference datasets

---

## References

* `CONTRIBUTING.md` — Commit/PR prefixes, full PR workflow, developer guide
* `docs/sources/devguide/` — Full developer documentation
* `maintainers/` — Release and emergency recovery procedures only
