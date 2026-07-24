[Maintainer Docs](../README.md) · [Governance References](README.md)

# Official Plugin Documentation Standard

## Status

Implemented governance reference.

This document defines the shared minimum documentation standard for official
SpectroChemPy plugins.

## Purpose

This note answers one recurring maintainer question:

```text
What documentation baseline should every official plugin provide?
```

## Scope

This standard applies to official SpectroChemPy plugins that expose
user-facing APIs, readers, writers, analysis tools, or domain-specific
workflows.

It defines a practical minimum standard. It does not prescribe the whole
global documentation architecture.

## Core Rule

Every official plugin should provide a small, predictable, user-facing
documentation surface in the main SpectroChemPy documentation.

Users should not need to infer implementation ownership before they can find:

- what the plugin does;
- how to install it;
- what public API they should call;
- what limitations or compatibility paths exist.

## Minimum Required Sections

Every official plugin should provide, at minimum:

1. a dedicated plugin page;
2. a short introduction explaining purpose and scope;
3. installation instructions;
4. the recommended public API form for new code;
5. compatibility alias documentation when aliases exist;
6. an API reference block or clear link to plugin-owned API pages when useful;
7. scope notes or current limitations;
8. links to examples or relevant workflows.

## Recommended Page Shape

The page structure should stay lightweight and predictable:

1. title and short purpose statement;
2. installation;
3. recommended API;
4. compatibility aliases;
5. API reference;
6. examples and workflows;
7. limitations.

Not every section must be long, but the structure should be recognizable across
official plugins.

## Ownership Rule

Authoritative user documentation ownership should follow implementation
ownership.

In practice:

- plugin-owned APIs should be documented from plugin-owned pages;
- namespaced entry points should be shown before compatibility aliases;
- compatibility aliases may remain documented, but as secondary paths;
- global discovery pages may point to plugin pages, but should not replace
  them as the owning surface.

## Reader And Writer Plugins

For reader or writer plugins, the plugin page should answer:

- what formats are covered;
- which API new code should use;
- whether generic dispatchers such as `scp.read(...)` or `scp.write(...)`
  can also reach the format;
- what limitations or support boundaries apply;
- what compatibility paths remain visible.

## Non-Goals

This standard does not:

- redesign the full user documentation tree;
- require identical documentation volume for every plugin;
- require immediate migration of all existing pages;
- freeze future documentation architecture work.

It sets a stable baseline so plugin growth does not produce a fragmented user
experience.
