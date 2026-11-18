---
name: spectrochempy-dev
description: >-
  Develop, refactor, or fix bugs in SpectroChemPy. Use for any code change:
  conventions, testing, pre-commit, shared decisions, and behavior preservation.
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

## PR Publication

Keep PR preparation local. Push a branch only when the maintainer explicitly
asks to push it or to open a PR that needs it. Open a GitHub PR, including a
draft PR, only when the maintainer explicitly asks to open it.
When opening an authorized PR, write its description as part of opening it.

## Other GitHub Content

Before creating or editing a GitHub issue, comment, review, or reply, show
the exact proposed text to the maintainer and obtain explicit validation.
Treat a request to write that content as a request for a local draft unless
the maintainer separately authorizes its publication after review.

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
* Keep execution records in the task's working record. Shared decisions and
  reusable audit evidence belong in maintained documents; retaining a result
  does not by itself require a repository file or a commit. Say where retained
  results, build trees, environments or datasets are available.

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

Do not treat targeted validation as a substitute for this pre-push gate:
`ruff check`, `ruff format --check`, focused pytest commands, or a green local
test are useful during development, but they do not satisfy the mandatory
`pre-commit run --all-files` requirement before committing and pushing to a PR
branch.

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

Every PR must satisfy the changelog compliance gate before it is opened:

* user-visible changes require an entry in `docs/sources/whatsnew/changelog.rst`;
* all other changes require the GitHub `no-changelog` label.

Never leave both absent: the `pr-compliance` workflow fails when neither the
changelog nor the label is present.

Link each new changelog entry to the PR that introduced the change. In Core,
use the existing Sphinx role, for example ``(:pr:`1874`)``. For standalone
plugins, use the changelog's local PR role if it targets that repository.
Otherwise use an explicit reStructuredText link to the plugin PR.
If the number is assigned when the PR opens, add the reference in a follow-up
commit and regenerate the release notes before handing off the PR. Verify the
PR number and target repository.

Edit `docs/sources/whatsnew/changelog.rst` only. Never edit
`docs/sources/whatsnew/latest.rst` or `docs/sources/whatsnew/index.rst`
manually (they are generated by the release workflow script).

When modifying `changelog.rst`, run the generation workflow so `latest.rst`
is regenerated. Include its diff in the commit.

**Decide based on user impact.** If the change is visible to users (bug fix,
new feature, behavior change), write a changelog entry. If it is purely
internal (refactoring, tests, docs), apply the `no-changelog` label to the PR.

---

## Code Conventions

* Do NOT add comments unless asked.
* Follow existing patterns in the codebase.
* Prefer existing helpers over new ones.
* Never add new dependencies without explicit approval.
* Never create new public API without asking.
* In gallery examples, notebooks, and user guides, keep calculations on
  `NDDataset` and `Coord` when their operators and methods express the
  calculation clearly. Use `scp.Coord.linspace()` for coordinates,
  `scp.linspace()` for sampled values, and `scp.normal()` for normal random
  samples. For line shapes, use the public helper for the intended model:
  `scp.gaussian()`, `scp.lorentzian()`, `scp.voigt()`, or
  `scp.pseudovoigt()`. Match the model's width and normalization to the
  example before replacing a formula. Keep NumPy for interoperability,
  low-level comparisons, and operations without a suitable public equivalent.
* Likewise, prefer `dataset.plot()`, `scp.multiplot()`, `scp.plot_multiple()`,
  and their `output=` lifecycle over direct Matplotlib calls. Keep Matplotlib
  when the example demonstrates integration or needs a feature the public
  plotting API does not expose.
  Check this on edited examples even when the imports were already present.
* In notebook-based tutorials and gallery examples, use `_ = expression`
  when an unused return value would appear as cell output. For example,
  `_ = dataset.plot()` hides the returned axes representation while keeping
  the figure visible. Keep useful diagnostics and intentionally displayed
  objects.

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

## Task Records and Shared Maintainer Documents

Keep investigation, commands, validation results and limits, review findings,
and delivery status in the task's working record under the workspace workflow.
For a standalone checkout, use the local handoff or PR discussion. A routine
PR, review round, push, or merge does not require a maintainer Markdown file.

Consult existing maintainer decisions before substantial work when the private
repository is available. Update its shared corpus only for information another
maintainer needs to understand, decide, or resume a topic: a durable API or
architecture decision, an important scientific qualification, or a reusable
campaign synthesis. Prefer the existing RFC, architecture reference, or topic
audit over a new report. Multi-PR coordination alone does not require a note.

Shared conclusions must include the evidence and limitations needed to assess
them without access to local task tracking. Keep product changes and their
validation understandable from the public code, tests, documentation and PR.
If a needed shared update cannot be made, record that pending work explicitly.

### Delivery Reconciliation

After an authorized push, verify the remote branch and PR head, CI-relevant
labels, and local worktree status; update the task record. After a merge,
verify it on GitHub, fetch the official product `master`, and record the
delivered outcome and any explicit successors before closing the task.

Update maintainer documents and their indexes only when shared content changes.
Do not create a reconciliation report or duplicate task statuses in an index.
Report any required shared update still awaiting authorized publication.
Inspect task-owned worktrees before removing clean merged ones. Preserve
unrelated work and existing audit history. Commits and publication still
require the applicable authorization.

---

## Self-Review (Implementation)

Before reporting completion, perform a short self-review of the final diff.
This is **not** an independent review — it is the implementer's final check.

### Self-Review Checklist

1. **Conformance** — Does the change satisfy the stated need? Are there
   omissions or scope creep?
2. **Consistency** — Are tests, documentation, and code coherent? Do test
   names and docstrings reflect the actual behavior?
3. **Maintainer coherence** — Does the task confirm, supersede, or introduce
   durable information tracked in `../spectrochempy_maintainer`? Update the
   relevant existing note when necessary; do not create a duplicate note merely
   to record a PR.
4. **Obvious errors** — Typos, wrong variable names, off-by-one, incorrect
   imports, dead code, leftover debug statements.
5. **Behavior preservation** — Does the diff preserve existing public behavior?
   Are there unintended side effects?

Fix any problem found before considering the task complete.

---

## When a Separate Review Is Required

Every PR requires independent review before merge, including documentation,
test-only and mechanical changes. Use a fresh temporary subagent session
without the implementation history. For work that is not going to merge,
separate review is still appropriate for changes to behavior, public APIs,
scientific calculations, I/O, CI or packaging.

## Handoff to Separate Review

When the maintainer requests a review or delegates a merge, start an
independent review in a fresh temporary subagent session. Give the reviewer
the prompt, applicable repository instructions, and repository access.
If an isolated reviewer is unavailable, report the limitation and provide
the review prompt. Leave any pending PR open until approved.

Include:

1. **Need** — the original requirement or problem statement.
2. **Branch / PR** — the branch name or PR number to review.
3. **Base and commit** — the exact base commit and the commit to examine
   (e.g., `git diff <base>..<commit>`).
4. **Sensitive points** — specific areas the implementer flags for attention
   (e.g., "edge case in negative wavenumbers", "assumes monotonic axis").

The reviewer works from this prompt, the applicable instructions, the diff,
and the relevant sources. The implementer's report is information to verify,
not authority.

For uncommitted work, provide the exact base commit and a snapshot of the
diff under review. Record which snapshot the verdict covers. The review
request itself authorizes the review loop locally.

After each review, verify the findings, fix actionable issues, and run
targeted validation. Start a fresh independent subagent session for each
updated diff without asking again. Give each reviewer the exact revision
and keep the implementation conversation and earlier reviews out of their
context. Repeat until the current revision receives an explicit `APPROVE`.
If a finding needs a maintainer decision, surface it and resume the review
loop once it is resolved.

Record the verdict and reviewed commit. Immediately before merging,
verify that GitHub still reports that head and that the required checks
passed. Make the merge request conditional on the approved head commit.
If GitHub reports a different head or rejects the merge for a head mismatch,
review the new head. If a guarded merge or independent review is unavailable,
or the verdict requires changes or discussion, keep the PR open.

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
3. Task record updated; shared documents updated when warranted (see Task
   Records and Shared Maintainer Documents).
4. Self-review performed and issues fixed.
5. If review was requested or merge delegated, complete the independent review
   loop and record the verdict for the exact revision. If blocked, report the
   blocker and provide the handoff prompt when a reviewer is unavailable.
6. Files staged (`git add <files>`) for VSCode Source Control handoff. Do not
   commit or push unless explicitly delegated.

When committing and pushing to a PR (explicitly delegated):

1. `git add <files>`
2. `pre-commit run --all-files` (repeat until clean)
3. `git commit -m "PREFIX: message"`
4. `git push origin <branch>`
