# AGENTS.md

## Scope

This document supplements:

* CONTRIBUTING.md
* docs/sources/devguide/
* maintainers/

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

# Audit Policy

## Local working notes

All implementation reports, investigations, reviews and working notes must be
written under `audit/`.

Files in `audit/` are intentionally untracked and must never be committed.

Use audit notes for:

* migration details;
* architectural decisions;
* implementation notes;
* roadmap planning;
* risk analysis;
* test results.

For multi-PR projects, maintain dedicated audit files.

Name new audit files with a leading tilde so they match the repository ignore
rule, for example `audit/~project-architecture-audit.md`.

Local audit files (`~*.md`) are working notes and may not be shared with other
maintainers.

## Promotion destinations

If an audit leads to a durable architectural decision, promote that knowledge
into the appropriate tracked document.

**Promotion never consists of simply moving or copying the local audit file.**
The maintainer document must be a **rewritten, maintained document** that
extracts the durable knowledge from the audit.  The original local audit may
remain as an untracked working note, be deleted, or—when it has long-term
historical value—be rewritten and committed separately under
`maintainers/audits/`.

| Destination | Use for |
|---|---|
| `maintainers/rfcs/` | Normative behavior contracts and accepted decisions |
| `maintainers/architecture/` | Durable architecture notes and current reference |
| `maintainers/roadmap/` | Migration roadmaps and campaign ordering |
| `maintainers/conventions.md` | Lightweight conventions and quick-reference |

The original audit remains a transient working document and is not the
authoritative reference.

## Promoted historical audits

Some audits retain long-term value as historical context even though they are
no longer primary authority.  These may be promoted to `maintainers/audits/`.

`maintainers/audits/` contains **tracked, curated, and rewritten** historical
audits that preserve decision-space analysis, migration baselines, and risk
maps for future maintainers.  They are **not committed copies** of the local
working notes.  They are distinct from:

- `audit/` — local untracked working notes;
- `maintainers/architecture/` — current durable architecture reference;
- `maintainers/rfcs/` — normative behavior contracts.

Only promote an audit to `maintainers/audits/` when it preserves knowledge
that future maintainers will need for context, not for authority.

## Campaign closure

Before closing a campaign, verify whether the audit contains any
architectural, maintenance, compatibility, or roadmap knowledge that future
maintainers will need.

If so, summarize that information in the appropriate `maintainers/` destination
before considering the campaign complete.

## Examples

```text id="llqkmn"
# Local working note (untracked)
audit/~project-architecture-audit.md

# Promoted historical audit (tracked)
maintainers/audits/coordinate-arithmetic-audit.md

# Accepted RFC (tracked, normative)
maintainers/rfcs/namespace-api-convention.md

# Durable architecture note (tracked, current reference)
maintainers/architecture/reader-normalization-architecture.md

# Roadmap (tracked)
maintainers/roadmap/vendor-io-migration.md
```

Detailed implementation history belongs in audits, not changelog entries.

Agents must produce or update an audit note after each work session,
documenting what was done, key decisions, test results, risks, and next
steps.  For multi-PR projects, maintain dedicated audit files and update
them before considering a task complete.

---

# Architecture Documentation Lifecycle

Architecture work should normally progress through the following lifecycle:

```text
Audit
  ↓
RFC (optional)
  ↓
Implementation
  ↓
Architecture Note / Maintainer Reference
```

The goal is to ensure that durable architectural knowledge does not remain
exclusively in local audit notes after a campaign is completed.

## Audit

Use audits for:

* exploration;
* characterization;
* investigation;
* design discussion.

Audits are working documents.

Audits are not authoritative by default for current maintained contracts.

## RFC

Use RFCs to:

* define a proposed contract;
* record decisions;
* guide implementation.

RFCs may be:

* proposed;
* accepted;
* implemented;
* superseded.

## Architecture Notes

Use architecture notes to:

* describe current architecture;
* capture stable contracts;
* document important design decisions;
* serve as maintainer references.

Architecture notes become the authoritative source once a design stabilizes.

## Promotion Requirement

When a campaign results in:

* an accepted RFC;
* significant architectural change;
* multiple implementation PRs;
* a long-term contract;

the maintainer should evaluate whether part of the audit material must be
promoted into the appropriate `maintainers/` destination:

* `maintainers/rfcs/` — for normative contracts and decisions;
* `maintainers/architecture/` — for durable current architecture reference;
* `maintainers/roadmap/` — for migration ordering and campaign planning;
* `maintainers/audits/` — for historical context that future maintainers
  will need (non-authoritative).

before the campaign is considered complete.

Architectural reasoning should not remain exclusively in audit documents.

## Audit Deliverables

Major architecture audits should end with a final section named:

```text
Promotion Candidates
```

That section should identify:

* content suitable for RFCs;
* content suitable for architecture notes;
* content that should remain historical only.

Where possible, suggest target filenames.

This requirement applies to major architecture audits, not to minor bug
investigations or narrow implementation notes.

## Campaign Closure Checklist

Before closing a major architecture campaign, verify:

* RFC status updated;
* roadmap updated;
* relevant architecture notes updated or created;
* promotion candidates reviewed;
* authoritative documentation synchronized.

This checklist is meant to keep durable knowledge discoverable, not to add
heavy process.

---

# Changelog Policy

The changelog is a release document.

It should explain:

* what changed;
* why it matters.

Do not use the changelog as a PR-by-PR implementation journal.

For multi-PR projects:

* keep detailed history in audit notes;
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
relying on system paths.  If a Conda environment named ``scpy-core`` exists
(the project's test environment), use it for all Python and pytest commands:

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
- Between commits, ensure code passes ``ruff`` and ``ruff-format`` manually
  (e.g. via editor integration or a targeted command) so that the final
  pre-commit run succeeds quickly.

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
