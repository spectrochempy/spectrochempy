---
name: spectrochempy-dev
description: Develop, refactor, or fix bugs in SpectroChemPy. Use for any code change: conventions, testing, pre-commit, audit notes, and behavior preservation.
metadata:
  audience: maintainer
  workflow: development
---

## Trigger

Use this skill when modifying source code, tests, or documentation in the
SpectroChemPy ecosystem.

---

## Git Operations

Push rule: push to `origin` only, never to `upstream`. PRs are opened from
`origin/<branch>` → `upstream/master`.

Do NOT create branches, commit, push, or open PRs unless explicitly delegated.

### Branch Names

Do **not** create branches starting with `release/`. That prefix is reserved
for the official publication process. Use descriptive prefixes like `fix/`,
`feat/`, `docs/`, `chore/`, `refactor/`.

### Commit & PR Titles

Every commit and PR title **must** use a prefix from `CONTRIBUTING.md`:

| Prefix | Usage |
|---|---|
| `ENH:` | User-facing enhancement or feature |
| `FIX:` | Bug fix |
| `DOC:` | Documentation changes |
| `TEST:` | Test addition or modification |
| `CI:` | CI/CD and workflow changes |
| `DEV:` | Developer tooling |
| `PERF:` | Performance improvement |
| `MAINT:` | Refactoring, cleanup, maintenance |
| `REL:` | Release-related work |

Never invent new prefixes. Non-standard prefixes require the `non-standard-prefix`
PR label.

---

## Python Environment

Use the `scpy-core` Conda/micromamba environment for all Python and pytest
commands:

```bash
micromamba run -n scpy-core python ...
micromamba run -n scpy-core python -m pytest ...
```

If unavailable, fall back to `.venv` or system Python with `PYTHONPATH=src/`.

---

## Testing Policy

Run only the smallest validation necessary:

* a single test;
* a focused test file;
* a targeted marker selection.

```bash
# Single test
micromamba run -n scpy-core python -m pytest tests/test_x.py::test_y -v

# By marker
micromamba run -n scpy-core python -m pytest -m "not slow" tests/

# Focused directory
micromamba run -n scpy-core python -m pytest tests/core/ -v
```

Do NOT run the full suite unless explicitly requested or preparing final
validation. Propose validation commands rather than executing them when
possible.

---

## Pre-commit Policy

**Pre-commit is NOT needed during development.** It is a waste of time to run
it repeatedly. Ruff and other linters are executed by the final pre-commit run.

**Pre-commit is MANDATORY only once, before the final commit and push to a PR
branch:**

```bash
git add -A
pre-commit run --all-files
```

Repeat until clean (0 failures, no file modifications). This is mandatory.

**Important:** `pre-commit run --all-files` only inspects `git ls-files` paths.
Untracked files are silently skipped. Always `git add -A` first, especially
when adding new files.

### VSCode Source Control Handoff (Default)

By default, after implementing and self-reviewing:

1. Stage all modified files: `git add -A`
2. Show the staged diff briefly
3. **Stop — do not commit or push**

This lets the maintainer examine the full diff in VSCode Source Control
before deciding whether to commit, amend, or request changes.

When the maintainer asks to commit and push to a PR:

```bash
git add -A
pre-commit run --all-files   # repeat if it modifies files
git commit -m "PREFIX: message"
git push origin <branch>
```

Override explicitly if you want the agent to commit or push directly.

---

## Changelog

Edit `docs/sources/whatsnew/changelog.rst` only. Never edit
`docs/sources/whatsnew/latest.rst` manually (it is generated).

When modifying `changelog.rst`, run the generation workflow so `latest.rst`
is regenerated. Include its diff in the commit.

**If no changelog entry is provided with the task, do NOT add one.** Apply
the `no-changelog` label to the PR. The maintainer will decide whether a
changelog entry is needed. Use the `no-changelog` label on PRs for:
* internal refactoring with no user-visible change;
* test-only changes;
* documentation-only changes;
* trivial fixes;
* multi-PR campaign internal changes with consolidated changelog.

---

## Code Conventions

* Do NOT add comments unless asked.
* Follow existing patterns in the codebase.
* Prefer existing helpers over new ones.
* Never add new dependencies without explicit approval.
* Never create new public API without asking.

---

## Behavior Preservation

* Preserve public APIs and behavior.
* Preserve backward compatibility.
* Preserve serialization compatibility.
* Preserve warning behavior and documented semantics.

For large refactoring or migration projects, prefer:

```
behavioring tests
  -> responsibility extraction
  -> adapter layer
  -> internal migration
  -> implementation switch
  -> serialization update
```

Avoid combining multiple migration phases in a single PR.

---

## Audit Notes

After each work session, create or update an audit note in
`spectrochempy_maintainer/notes/audits/` documenting:

1. What was done
2. Key decisions
3. Test results
4. Risks
5. Next steps

Hygiene rules:
* Prefer editing an existing document over creating a new one.
* Before creating a new note, check for existing audit, roadmap, RFC, or
  architecture note on the same topic.
* Keep `roadmap/current-roadmap.md` short.
* Archive notes in `spectrochempy_maintainer/archive/audits/` once they become
  primarily historical.

---

## Self-Review (Implementation)

Before reporting completion, perform a short self-review of the final diff.
This is **not** an independent review — it is the implementer's final check.

### Self-Review Checklist

1. **Conformance** — Does the change satisfy the stated need? Are there
   omissions or scope creep?
2. **Consistency** — Are tests, documentation, and code coherent? Do test
   names and docstrings reflect the actual behavior?
3. **Obvious errors** — Typos, wrong variable names, off-by-one, incorrect
   imports, dead code, leftover debug statements.
4. **Behavior preservation** — Does the diff preserve existing public behavior?
   Are there unintended side effects?

Fix any problem found before considering the task complete.

---

## When a Separate Review Is Required

A separate review in a new OpenCode session is required for changes that
affect:

* **Behavior** — any user-visible behavior change, bug fix with semantic
  impact, or algorithm modification.
* **API** — new, modified, or removed public API symbols.
* **Scientific computation** — numerical methods, transforms, fitting,
  preprocessing, or any calculation that affects results.
* **Readers / I/O** — file format handling, serialization, or data import.
* **CI / Publication** — workflows, release process, packaging.

A separate review is **not** required for:

* Pure documentation edits (typos, formatting, clarifications).
* Mechanical refactoring with no behavior change (renames, reformatting).
* Test-only additions that verify existing behavior.

Justify briefly when skipping the separate review based on actual risk.

---

## Handoff to Separate Review

When a separate review is required, do **not** launch a second session or
another model automatically. Instead, produce a short prompt directly usable
in a new OpenCode session. Include:

1. **Need** — the original requirement or problem statement.
2. **Branch / PR** — the branch name or PR number to review.
3. **Base and commit** — the exact base commit and the commit to examine
   (e.g., `git diff <base>..<commit>`).
4. **Sensitive points** — specific areas the implementer flags for attention
   (e.g., "edge case in negative wavenumbers", "assumes monotonic axis").

The reviewer works from this prompt, the applicable instructions, the diff,
and the relevant sources. The implementer's report is information to verify,
not authority.

Example handoff prompt:

```
Revue séparée — branch fix/coordset-negative-axis, commit abc1234
Besoin : permettre les axes décroissants dans CoordSet sans casser la
sérialisation. Base : a1b2c3d.
Points sensibles : (1) suppose que axis.step conserve son signe après
reshape, (2) le test test_coordset_reverse ne vérifie pas la round-trip
JSON, (3) aucun test pour les axes non-monotoniques.
```

---

## Validation

Before reporting completion:

1. Tests directly affected by the change pass.
2. No broken imports or circular dependencies.
3. Audit note updated in maintainer repository.
4. Self-review performed and issues fixed.
5. If required, handoff prompt for separate review produced.
6. Files staged (`git add -A`) for VSCode Source Control handoff. Do not
   commit or push unless explicitly delegated.

When committing and pushing to a PR (explicitly delegated):

1. `git add -A`
2. `pre-commit run --all-files` (repeat until clean)
3. `git commit -m "PREFIX: message"`
4. `git push origin <branch>`
