# Maintainer Documentation

This directory contains **public authoritative maintainer references** for
SpectroChemPy: accepted and proposed RFCs, durable architecture notes,
public roadmaps, and operational release and recovery procedures.

For governance discussion, proposal incubation, exploratory research, and
long-form strategic conversation, see the separate **SpectroChemPy Maintainer
Repository**. Accepted decisions from that repository are promoted here when
stable.

This is **not user-facing documentation**. User documentation is at
[spectrochempy.fr](https://www.spectrochempy.fr) and in [`docs/`](../docs/).

## Structure

| Path | Role | Purpose |
|---|---|---|
| [`rfcs/INDEX.md`](rfcs/INDEX.md) | What — contracts | RFC inventory with status and authority |
| [`architecture/INDEX.md`](architecture/INDEX.md) | How — organisation | Curated architecture references |
| [`roadmap/current-roadmap.md`](roadmap/current-roadmap.md) | When — ordering | Current priorities and active candidates |
| [`audits/INDEX.md`](audits/INDEX.md) | Why — context | Retired audits with pointers to governance location |
| [`release-process.md`](release-process.md) | How — release | Release procedure |
| [`emergency-recovery.md`](emergency-recovery.md) | How — recovery | Incident recovery |
| [`api-conventions.md`](api-conventions.md) | What — API | Namespace API quick reference |

## Reading Path

For maintainer orientation, start with the roadmap, then the indexes:

1. [`roadmap/current-roadmap.md`](roadmap/current-roadmap.md)
2. [`rfcs/INDEX.md`](rfcs/INDEX.md)
3. [`architecture/INDEX.md`](architecture/INDEX.md)

For contributors and plugin authors, see [`api-conventions.md`](api-conventions.md).

## Local Audit Notes

The repository-level `audit/` directory (untracked) contains local working
notes, implementation logs, and campaign investigations. These are not
authoritative. See the SpectroChemPy Maintainer Repository for shared
maintainer notes and proposal incubation.
