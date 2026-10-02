# AGENTS.md

## Scope

This document defines permanent rules and authorization limits for AI-assisted
development in SpectroChemPy.

Procedural details live in the project's OpenCode skills and in
`CONTRIBUTING.md`. Load the relevant skill before starting work.

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

## VSCode Source Control Handoff

By default, the agent stages modified files (`git add`) so the full diff is
visible and reviewable in VSCode Source Control — but does **not** commit or
push. The maintainer examines the diff in VSCode, then decides whether to
commit, amend, or request changes.

Pre-commit is **not** needed during development. It is mandatory only once,
before the final commit and push to a PR branch. Ruff and other linters are
executed by that final pre-commit run.

Override explicitly if you want the agent to commit directly (e.g., "commit
and push on my behalf").

---

## Task Execution

Unless explicitly delegated to finalize:

* produce code changes, test updates, documentation updates;
* update audit notes in `spectrochempy_maintainer/notes/audits/`;
* propose a prefixed commit title and PR title (see `CONTRIBUTING.md`);
* propose targeted validation commands;
* list remaining risks and recommended follow-up.

For multi-PR projects, update or create the relevant audit note before
considering the task complete.

---

## Self-Review vs. Separate Review

### Self-Review (Implementation)

The implementer performs a short self-review of the final diff before
completing. This is **not** an independent review — it is the implementer's
final check.

Check: conformance to need, omissions, consistency of tests and
documentation, obvious errors. Fix any problem found before finishing.

### Separate Review

A separate review in a **new OpenCode session** is required for changes
affecting:

* behavior (user-visible changes, bug fixes with semantic impact);
* API (new, modified, or removed public symbols);
* scientific computation (numerical methods, transforms, fitting);
* readers / I/O (file formats, serialization);
* CI / publication (workflows, release process).

A separate review is **not** required for pure documentation edits,
mechanical refactoring with no behavior change, or test-only additions that
verify existing behavior. Justify briefly when skipping.

### Handoff to Separate Review

When a separate review is required, do **not** launch a second session or
another model automatically. Produce a short prompt for a new session
containing: the need, the branch or PR, the base and commit to examine, and
the sensitive points. The reviewer starts from this prompt, the applicable
instructions, the diff, and the relevant sources.

---

## Skills

Load the relevant skill before starting work:

| Skill | When to load |
|---|---|
| `spectrochempy-dev` | Any development, refactoring, or bug fix |
| `spectrochempy-review` | Separate review in a new session (see above) |
| `spectrochempy-release` | Preparing a release (without publishing) |

**For Codex:** skills are in `.opencode/skills/` — read the relevant `SKILL.md` directly.

---

## Example Prompts

### 1. Fix a bug

```
Corrige le bug de serialization NDDataset en gardant les tests à jour.
Valide avec le marker approprié et mets à jour la note d'audit.
```

### 2. Review a PR

```
Revois la PR #1842 en focalisant sur la préservation de l'API publique.
Produis un verdict et les commandes de validation avant merge.
```

### 3. Prepare a release

```
Prépare la release 1.2.0 — vérifie la cohérence de version et la complétude
du changelog. Ne pousse pas et n'ouvre pas de PR.
```

### 4. Separate review in a new session

```
Revue séparée — branche fix/coordset-negative-axis, commit abc1234
Base : a1b2c3d.
Besoin : permettre les axes décroissants dans CoordSet sans casser la
sérialisation.
Points sensibles : (1) suppose que axis.step conserve son signe après
reshape, (2) le test test_coordset_reverse ne vérifie pas la round-trip
JSON, (3) aucun test pour les axes non-monotoniques.
```

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
