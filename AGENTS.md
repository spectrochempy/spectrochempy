# AGENTS.md

## Scope

This document supplements:

* CONTRIBUTING.md
* docs/sources/devguide/
* maintainers/ (release and recovery procedures, plus shared stable maintainer RFCs, architecture references, and selected governance references)

Agents must follow all project contribution rules defined there.

This file defines additional requirements specific to AI-assisted development.

When rules overlap, follow the stricter requirement.

---

# Core Principles

Priorities:

1. correctness;
2. behavior preservation;
3. maintainability;
4. reviewability;
5. resource efficiency.

Prefer small, reversible, reviewable changes.

Avoid broad rewrites unless explicitly requested.

---

# Public Behavior Preservation

Unless explicitly requested otherwise:

* preserve public APIs;
* preserve public behavior;
* preserve backward compatibility;
* preserve serialization compatibility;
* preserve warning behavior;
* preserve documented semantics.

Internal refactoring should not introduce user-visible behavior changes.

When adding or exposing a new public top-level API symbol
(``spectrochempy.<name>`` / ``scp.<name>``), also update
`docs/sources/reference/index.rst` in the appropriate section of the public API
reference.

---

# Architecture Strategy

For large refactoring or migration projects, prefer:

```text id="px8jk3"
behavioral tests
    ->
responsibility extraction
    ->
adapter layer
    ->
internal migration
    ->
implementation switch
    ->
serialization update
```

Avoid combining multiple migration phases in a single PR.

Prefer incremental migration.

---

# Design and Documentation Workflow

Active, private, exploratory, or not-yet-stable working notes belong in the
**SpectroChemPy Maintainer Repository** (`spectrochempy_maintainer`).

Use `spectrochempy_maintainer` for:

* `notes/audits/` — active investigations, campaign logs, risk analysis, test
  results, and implementation notes;
* `notes/`, `proposals/`, `roadmap/`, `governance/` — private incubation,
  maintainer planning, and non-public working material;
* `archive/` — historical context that remains useful but is no longer an
  active reference.

Stable shared maintainer documentation belongs in the public repository under
`spectrochempy/maintainers/`:

* `maintainers/rfcs/` — accepted or still-relevant shared RFCs and contracts;
* `maintainers/architecture/` — maintained architecture references;
* `maintainers/governance/` — stable governance and documentation-process
  references;
* `maintainers/` root — operational maintainer procedures such as release and
  recovery.

Local scratch notes may remain unversioned. They must not be introduced into
the public `spectrochempy/` repository.

The old `audit/` directory in the public repository has been removed. No new
files should be created there.

## Promotion and Review

When a private audit or proposal leads to a durable design decision, extract
the stable result into the appropriate document under `spectrochempy/maintainers/`.

Promotion never consists of simply moving or copying the audit file. Rewrite
the durable content as a maintained public document and keep the original audit
in `spectrochempy_maintainer` when its historical context is still useful.

Discussion of RFCs, architecture notes, and other durable shared maintainer
documentation should normally happen through the corresponding Pull Request.
Separate Issues or Discussions may still be used when they add value, but they
are not required.

Long-lived engineering knowledge should not remain exclusively in private notes
or Pull Request discussions once a design has stabilized.

## Working-note hygiene

For multi-PR projects, maintain a dedicated audit note in
`spectrochempy_maintainer/notes/audits/` and update it as the work evolves.

When writing into `spectrochempy_maintainer`, prefer updating an existing
active document over creating a new one.

Before creating a new note, verify whether the same topic already has:

* an active audit in `spectrochempy_maintainer/notes/audits/`;
* a maintained contract in `spectrochempy/maintainers/rfcs/`;
* a maintained architecture note in `spectrochempy/maintainers/architecture/`.

For any active topic, keep a single obvious entry point:

* audit for evidence, implementation notes, and unresolved questions;
* RFC for the normative contract or decision;
* architecture note for the maintained current model.

Move an audit note to `spectrochempy_maintainer/archive/` once it becomes
primarily a validation log, implementation history record, or historical
context already absorbed by code and maintained documentation.

At the end of the session, explicitly decide for each maintainer note touched
whether it remains active, should be promoted, or should be archived.

## Lifecycle

Large design or architectural work should normally follow this progression:

```text
Exploration
    ->
RFC (optional)
    ->
Implementation
    ->
Pull Request review and discussion
    ->
Merge
    ->
Architecture / governance / RFC update when needed
```

Agents must update the relevant private audit note after each work session when
the task involves substantial implementation, design, or investigation work.

---

# Changelog Policy

The changelog is a release document.

It should explain:

* what changed;
* why it matters.

Do not use the changelog as a PR-by-PR implementation journal.

For multi-PR projects:

* keep detailed history in research notes in the maintainer repository;
* consolidate related work into meaningful release-level entries;
* prefer updating an existing entry over creating many near-duplicate entries.

Never edit:

```text id="c90b8w"
docs/sources/whatsnew/latest.rst
```

manually.

Edit:

```text id="z9ggiw"
docs/sources/whatsnew/changelog.rst
```

only.

Generated files derived from changelog entries (for example `latest.rst`)
should not be edited manually.

Agents should update `changelog.rst` only and leave generation of derived files
to the normal project workflow.

When `docs/sources/whatsnew/changelog.rst` is modified, agents must run the
normal generation workflow (for example the relevant pre-commit hook or the
documentation build tooling) so `docs/sources/whatsnew/latest.rst` is
regenerated. The generated `latest.rst` diff must be included in the final
commit; omitting it causes CI failures. The prohibition is against hand-editing
generated files, not against committing tool-generated updates.

## Changelog CI Workflow Bypass

A CI workflow verifies that every PR has a corresponding changelog entry in
``docs/sources/whatsnew/changelog.rst`` after applying the ``no-changelog``
label.

Use the ``no-changelog`` label when the PR does not need a changelog entry:

* internal refactoring with no user-visible behavior change;
* test-only changes (unless they test a new user-facing feature);
* documentation-only changes (example gallery, docstrings);
* trivial fixes (typos, comment corrections);
* multi-PR campaign internal changes where the changelog entry is consolidated
  in a later PR.

To apply the label on an open PR:

```bash
gh pr edit <PR_NUMBER> --add-label no-changelog
```

The label must be applied before the workflow runs, or CI will fail.

## Safe-docs CI bypass

For PRs that modify only non-executable maintainer or policy documentation,
maintainers may apply the label ``safe-docs-no-ci`` to bypass the heavyweight
test and docs workflows.

This label is intentionally narrow.  It is valid only when the changed files
are limited to safe documentation/policy paths such as:

* `AGENTS.md`;
* root-level `*.md`;
* `maintainers/**/*.md`;
* `CONTRIBUTING.md`;
* the PR template and similar non-executable repository-policy documents.

Do **not** use ``safe-docs-no-ci`` for:

* `docs/` changes, even when the files are non-Python;
* `examples/` changes;
* gallery/example/tutorial updates;
* plugin Markdown such as `plugins/**/README.md`;
* code changes hidden inside a documentation PR;
* any change that can affect executable documentation, packaging, plugins, or
  runtime behavior.

The CI bypass is enforced conservatively: the label alone is not sufficient if
the changed files are outside the approved safe-documentation scope.

---

# Cost-Aware Development

Assume agent actions consume limited resources.

Prefer:

* focused context;
* focused searches;
* targeted validation;
* incremental work.

Avoid unnecessary expensive operations.

Prefer analysis over execution whenever possible.

When multiple valid approaches exist, prefer the one requiring:

* fewer agent actions;
* fewer test executions;
* fewer CI runs;
* fewer GitHub operations.

---

# Python Environment

Prefer using an existing Conda environment over creating ad-hoc venvs or
relying on system paths.  If a micromanba, mamba or conda environment named ``scpy-core`` exists
(the project's test environment), use it for all Python and pytest commands:

```bash
micromamba run -n scpy-core python ...
micromamba run -n scpy-core python -m pytest ...
```

or

```bash
conda run -n scpy-core python ...
conda run -n scpy-core python -m pytest ...
```

If ``scpy-core`` is not available, fall back to the project's ``.venv`` or a
system Python with ``PYTHONPATH`` pointing to ``src/``.

---

# Testing Policy

Run only the smallest validation necessary.

Prefer:

* a single test;
* a focused test file;
* a targeted marker selection;

over broad test execution.

Do not run large validation suites unless:

* explicitly requested;
* preparing final validation;
* investigating a specific failure.

When possible:

* propose validation commands;
* let the maintainer execute them.

---

# Pre-commit Policy

Pre-commit validation must be run **once** before creating a pull request or
pushing to ``upstream/master``.  It must not be run during normal development.

Command:

```bash id="4qukx8"
pre-commit run --all-files
```

Rationale:

- Running pre-commit repeatedly during development wastes CI quota and agent
  time.
- A single final run before push is sufficient because pre-commit hooks are
  deterministic.
- Pre-commit handles all linting and formatting, including ``ruff`` checks,
  so there is no need to run linting tools manually during development.

When not delegated:

* provide the command;
* explain why it should be run.

---

# Local and Remote Action Policy

The maintainer controls:

* commits;
* branches;
* pushes;
* pull requests;
* releases;
* package publication.

Agents assist development.

Agents do not operate the repository by default.

---

# Allowed By Default

Agents may:

* inspect files;
* modify source code;
* modify tests;
* modify documentation;
* update audit notes;
* analyze architecture;
* review code;
* suggest validation commands.

---

# Not Allowed By Default

Unless explicitly requested, do not:

* create branches;
* create commits;
* push branches;
* open pull requests;
* merge pull requests;
* create releases;
* publish packages;
* run broad test suites;
* run pre-commit during normal development (see Pre-commit Policy).

---

# Task Execution Defaults

Unless explicitly requested otherwise:

* do not create branches;
* do not commit changes;
* do not push changes;
* do not open pull requests;
* do not run pre-commit during normal development (see Pre-commit Policy);
* do not run broad test suites.

Prefer producing:

* code changes;
* audit updates;
* suggested commit title;
* suggested PR title;
* concise PR description;
* targeted validation commands.

The maintainer is expected to perform final validation, commits, pushes, and
PR creation unless explicitly delegated otherwise.

---

# Commit and PR Titles

Follow the prefix conventions defined in CONTRIBUTING.md.

Always propose:

* a prefixed commit title;
* a prefixed PR title.

Never propose unprefixed titles.

---

# Default Deliverable

Unless explicitly delegated to finalize work, provide:

* source changes;
* test updates if needed;
* documentation updates if needed;
* audit updates;
* suggested commit title;
* suggested PR title;
* concise PR description;
* targeted validation commands;
* remaining risks;
* recommended follow-up work.

For multi-PR projects, update or create the relevant audit note before
considering the task complete.

Do not perform git operations automatically.

---

# Code Review Expectations

Evaluate:

* correctness;
* behavior preservation;
* regression risk;
* architectural consistency;
* testing adequacy;
* roadmap alignment.

Do not approve changes solely because tests pass.
