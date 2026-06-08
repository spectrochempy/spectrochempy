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

Audit files are the authoritative implementation history.

Use audit notes for:

* migration details;
* architectural decisions;
* implementation notes;
* roadmap planning;
* risk analysis;
* test results.

For multi-PR projects, maintain dedicated audit files.

Examples:

```text id="llqkmn"
audit/project-architecture-audit.md
audit/project-pr12-notes.md
audit/project-pr13-notes.md
```

Detailed implementation history belongs in audits, not changelog entries.

Agents should update relevant audit notes before considering a multi-PR task complete.

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

Do not run:

```bash id="4qukx8"
pre-commit run --all-files
```

unless explicitly requested.

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
* run pre-commit.

---

# Task Execution Defaults

Unless explicitly requested otherwise:

* do not create branches;
* do not commit changes;
* do not push changes;
* do not open pull requests;
* do not run pre-commit;
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
