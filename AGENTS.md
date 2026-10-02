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

When a task is authorized, proceed to a validated solution and a first review
of the diff without requesting confirmation for ordinary technical choices.

## PR Base

Always base PRs on `upstream/master` unless explicitly instructed otherwise.
Before creating a new branch, sync with `upstream/master` first.

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
