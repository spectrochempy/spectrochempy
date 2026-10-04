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

Read `CONTRIBUTING.md` at the start of each session.

---

## Git Operations

Push to `origin` by default. Push to `upstream` only when the maintainer
explicitly directs it. PRs are opened from `origin/<branch>` →
`upstream/master`.

**Before creating a new branch, inspect the local state first
(`git status`, current branch), then branch directly from the updated
`upstream/master` — without modifying the local `master`:**

```bash
git fetch upstream
git checkout -b <new-branch> upstream/master
```

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

## Temporary Files

Use the temporary directory of the current environment. `tempfile` resolves its
location from `TMPDIR`, `TEMP`, `TMP` and the platform defaults. Whether that
location is backed by RAM or disk, and how long it survives, remain the
platform's business.

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="opencode-validation-") as tmp:
    work = Path(tmp)
    build(work)  # build tree, generated documentation, validation copy
```

* The context manager removes the directory when the task ends, on every
  platform — and it fails loudly on Windows if a handle is still open, rather
  than leaving a directory behind silently.
* For a large build, do not assume the default location has room: check the
  free space, and pass `dir=` to place the work on a local disk.
* Never hardcode a temporary path in repository code: use
  `tempfile.TemporaryDirectory()` or the pytest `tmp_path` fixture.
* Keeping a result and committing it are different acts. A report or a note
  worth keeping belongs in a repository; a build tree, an environment or a
  dataset may stay in a persistent non-versioned location. Say which is which
  when reporting, and where you left anything.

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
validation. Execute the targeted tests and report their results.

---

## Pre-commit Policy

**Pre-commit is NOT needed during development.** It is a waste of time to run
it repeatedly: the final pre-push run owns lint and formatting.

A read-only check on a targeted path is fine — `ruff check <path>`,
`ruff format --check <path>`. A rewriting pass (`ruff check --fix`,
`ruff format`) is not a development step, and neither is re-running one to
"fix" a file you are editing: the hooks rewrite code as well, so review the
diff after any run that modifies files. If the hooks cannot be installed,
report that instead of substituting a direct rewriting run.

**Pre-commit is MANDATORY only once, before the final commit and push to a PR
branch:**

```bash
git add <files>
pre-commit run --all-files
```

Repeat until clean (0 failures, no file modifications). This is mandatory.

**Important:** `pre-commit run --all-files` only inspects `git ls-files` paths.
Untracked files are silently skipped. Stage the task-relevant files (including
new files) explicitly before running it.

### VSCode Source Control Handoff (Default)

By default, after implementing and self-reviewing:

1. Inspect the initial state (`git status`) to identify only the files
   relevant to the task
2. Stage those files explicitly (`git add <files>`)
3. Show the staged diff briefly
4. **Stop — do not commit or push**

This lets the maintainer examine the full diff in VSCode Source Control
before deciding whether to commit, amend, or request changes.

When the maintainer asks to commit and push to a PR:

```bash
git add <files>
pre-commit run --all-files   # repeat if it modifies files
git commit -m "PREFIX: message"
git push origin <branch>
```

Override explicitly if you want the agent to commit or push directly.

---

## Changelog

Edit `docs/sources/whatsnew/changelog.rst` only. Never edit
`docs/sources/whatsnew/latest.rst` or `docs/sources/whatsnew/index.rst`
manually (they are generated by the release workflow script).

When modifying `changelog.rst`, run the generation workflow so `latest.rst`
is regenerated. Include its diff in the commit.

Do not add a changelog entry unless the maintainer provides one. If no entry is
provided, apply the `no-changelog` label to the PR where appropriate.

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
behavioral tests
  -> responsibility extraction
  -> adapter layer
  -> internal migration
  -> implementation switch
  -> serialization update
```

Avoid combining multiple migration phases in a single PR.

---

## Audit Notes

Create or update an audit note in `../spectrochempy_maintainer/notes/audits/`
when the task involves multi-PR coordination, architectural decisions, or
durable knowledge worth preserving. For simple bug fixes or small changes,
skip the audit note.

For multi-PR work, update or create the relevant audit note before considering
the task complete.

If `../spectrochempy_maintainer` is not cloned, skip audit notes and report it.

Hygiene rules:
* Prefer editing an existing document over creating a new one.
* Before creating a new note, check for existing audit, roadmap, RFC, or
  architecture note on the same topic.
* Keep `roadmap/current-roadmap.md` short.
* Archive notes in `../spectrochempy_maintainer/archive/audits/` once they become
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

1. Execute targeted tests and report their results.
2. No broken imports or circular dependencies.
3. Audit note updated if applicable (see Audit Notes section).
4. Self-review performed and issues fixed.
5. If required, handoff prompt for separate review produced.
6. Files staged (`git add <files>`) for VSCode Source Control handoff. Do not
   commit or push unless explicitly delegated.

When committing and pushing to a PR (explicitly delegated):

1. `git add <files>`
2. `pre-commit run --all-files` (repeat until clean)
3. `git commit -m "PREFIX: message"`
4. `git push origin <branch>`
