---
name: spectrochempy-dev
description: >-
  Develop, refactor, or fix bugs in SpectroChemPy. Use for any code change:
  conventions, testing, pre-commit, audit notes, and behavior preservation.
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

## Audit Notes

Create or update an audit note in `../spectrochempy_maintainer/notes/audits/`
when the task involves multi-PR coordination, architectural decisions, or
durable knowledge worth preserving.

A small diff or narrowly scoped PR is **not** by itself a reason to skip the
maintainer note. Durable decisions include, for example:

* retaining, deprecating, or deliberately not changing a public API;
* reader relationships, format routing, or serialization policy;
* architectural or compatibility boundaries;
* known limitations deliberately deferred out of scope;
* rationale that a future maintainer would otherwise have to reconstruct from
  closed issues or PR discussions.

Skip the audit note only when the change is both locally scoped **and**
introduces no durable project decision worth recovering later.

For multi-PR work, or whenever such a durable decision is made, update or
create the relevant audit note before considering the task complete.

If `../spectrochempy_maintainer` is not cloned, skip audit notes and report it.

Hygiene rules:
* Prefer editing an existing document over creating a new one.
* Before creating a new note, check for existing audit, roadmap, RFC, or
  architecture note on the same topic.
* Keep `roadmap/current-roadmap.md` short.
* Archive notes in `../spectrochempy_maintainer/archive/audits/` once they become
  primarily historical.

### Maintainer Reconciliation After Push or Merge

For maintainer-directed work, reconciliation is mandatory after every push that
updates a PR and again after its merge. Do not treat a product PR as complete
while its maintainer records are only local, staged, or on an unrelated branch.

After a push, verify the remote branch, PR head, CI-relevant labels, and local
worktree status. Update the associated Bead and run
`bd -C ../spectrochempy_maintainer backup sync` when Beads data changed.

After a merge, verify the merge on GitHub, fetch the official product `master`,
then reconcile `spectrochempy_maintainer` against the merged result:

1. update and publish the relevant audit, plan, or decision note on maintainer
   `master` (unless the maintainer explicitly defers it);
2. review the applicable README/index, task status, roadmap, and archive
   placement so active, completed, and deferred work agree;
3. update/close the associated Bead and synchronize its backup;
4. inspect every task-owned worktree and remove clean merged worktrees;
5. leave no task-owned notes staged locally or only pushed to a side branch.

If a reconciliation item is intentionally deferred, record the reason in the
maintainer note or Bead before reporting the merge as complete.

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
3. Audit note updated if applicable (see Audit Notes section).
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
