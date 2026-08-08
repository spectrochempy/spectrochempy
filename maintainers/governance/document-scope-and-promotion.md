[Maintainer Docs](../README.md) · [Governance References](README.md)

# Public Maintainer Document Scope And Promotion

## Status

Implemented governance reference.

This document defines the maintained boundary between the shared public
maintainer corpus in `spectrochempy/maintainers/` and the private working
material kept in `spectrochempy_maintainer/`.

## Purpose

This note answers one practical maintainer question:

```text
What belongs in the public maintainer corpus,
what stays private,
and when should private material be promoted?
```

## Public Shared Corpus

The public maintainer corpus in `spectrochempy/maintainers/` is intentionally
small.

It should contain only material that is stable enough to be shared, reviewed,
and maintained in the main repository:

- stable RFCs and accepted contracts;
- maintained architecture references;
- shared governance references with project-wide value;
- selected historical documents that still help explain current architecture or
  policy.

It should not contain:

- active audit notes;
- private implementation logs;
- immature design exploration;
- assistant work in progress;
- private roadmap planning;
- temporary session notes or decision-space dumps.

## Private Maintainer Repository

The private `spectrochempy_maintainer/` repository remains the normal location
for:

- active audits and campaign notes;
- proposals and incubating design work;
- private governance discussion;
- archive and historical context;
- non-public planning or exploratory comparison work.

That repository is allowed to contain alternatives, partial drafts, competing
ideas, and evidence that would be too noisy for the shared public corpus.

## Promotion Rule

Promote a private document when all of the following are true:

1. the core conclusion is stable enough to be shared publicly;
2. the document no longer depends on private coordination context;
3. future maintainers would reasonably expect to find the answer in the main
   repository;
4. the material can be rewritten as a maintained reference rather than kept as
   a raw working note.

Promotion means rewriting, not copying.

## Promotion Destinations

Use the public destination that matches the document's role:

| Destination | Use for |
|---|---|
| `maintainers/rfcs/` | Normative maintainer contracts, accepted positions, implemented decision records |
| `maintainers/architecture/` | Maintained current architecture references and selected historical architecture notes |
| `maintainers/governance/` | Shared process boundaries, documentation standards, and stable maintainer-governance references |

## Promotion Checklist

Before promoting material from the private repository:

1. remove private coordination details, open-ended comparisons, and local
   campaign mechanics;
2. rewrite the document for the broader shared maintainer audience;
3. make the public document self-contained;
4. update or archive the private source note with a clear pointer to the
   public destination;
5. keep only the durable conclusion in the public repository.

## Anti-Patterns

- copying a proposal or audit verbatim into the public repository;
- using the public corpus as a second working notebook;
- keeping a stable shared answer only in private notes;
- promoting documents whose main value is still local discussion history.
