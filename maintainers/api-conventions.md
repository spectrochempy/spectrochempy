# API Conventions

## Purpose

This document summarizes the **namespace-based public API conventions** for
SpectroChemPy.  It is intended as a quick reference for maintainers, contributors,
and plugin authors.

For the full specification, see the tracked RFC
[`maintainers/rfcs/namespace-api-convention.md`](../maintainers/rfcs/namespace-api-convention.md).

## Core Principles

### 1. Namespaces are functional domains

A **namespace** represents a functional domain, not necessarily a file format.

```python
scp.jcamp      # an open standard
scp.csv        # a generic text format
scp.perkinelmer   # a vendor
scp.nmr        # a scientific domain
```

Despite their different natures, they all expose the same uniform surface.

### 2. Short method names

The namespace provides context; methods should never repeat the domain name.

```python
# Correct
scp.jcamp.read(...)
scp.jcamp.write(...)
scp.jcamp.info(...)

# Incorrect — redundant prefix
scp.jcamp.read_jcamp(...)
scp.jcamp.write_jcamp(...)
```

### 3. Minimal and consistent API

Each namespace exposes only the operations it implements.  Common operations
include:

- `read(...)` — import data
- `write(...)` — export data
- `info(...)` — inspect file metadata
- `validate(...)` — check file integrity

A namespace may add domain-specific operations (e.g., `convert`) when relevant.

### 4. Stability across implementation changes

The public namespace API remains stable even when the underlying implementation
changes.  Moving a reader from core to plugin, or refactoring a parser, must
not alter the user-visible API.

### 5. Generic dispatchers preserved

The top-level generic dispatchers remain the primary recommended APIs:

```python
scp.read("file.sp")          # auto-detect format
scp.write("file.jdx", ds)    # auto-detect format
```

They are never deprecated.

### 6. Legacy aliases retained

Top-level aliases such as `scp.read_jcamp(...)` and `scp.write_jcamp(...)`
remain as compatibility shims.  They may stay indefinitely; deprecation is
optional and unlikely.

### 7. Reserved namespaces

A set of public namespaces is reserved by the project.  See the RFC for the
authoritative list:
[`maintainers/rfcs/namespace-api-convention.md#reserved-public-namespaces`](../rfcs/namespace-api-convention.md#reserved-public-namespaces).

## Summary

```python
# Recommended — automatic format detection
ds = scp.read("file.sp")
scp.write("file.jdx", ds)

# Explicit — when you know the domain
ds = scp.jcamp.read("file.jdx")
scp.jcamp.write("file.jdx", ds)

# Legacy — still works, documented for compatibility
ds = scp.read_jcamp("file.jdx")
scp.write_jcamp("file.jdx", ds)
```

The namespace API is the **canonical documentation target**.  All new
documentation, tutorials, and examples should use the namespace form.
