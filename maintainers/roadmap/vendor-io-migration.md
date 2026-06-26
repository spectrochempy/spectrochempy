[Maintainers](../README.md) · [Roadmap](INDEX.md)

# Vendor I/O Migration Roadmap

## Goal

Progressively move vendor-specific file I/O operations from the SpectroChemPy
core package into optional official plugins, keeping the core lightweight and
domain-neutral.

## Status

| Vendor | Format | Status | Plugin | Notes |
|--------|--------|--------|--------|-------|
| PerkinElmer | `.sp` | ✅ Migrated | `spectrochempy-perkinelmer` | Reference implementation for future migrations |
| Bruker | OPUS (`.0`, `.1`, ...) | ☐ In core | — | Candidate for `spectrochempy-bruker` or `spectrochempy-opus` |
| Thermo Nicolet | OMNIC (`.spa`, `.spg`, `.srs`) | ☐ In core | — | Candidate for `spectrochempy-omnic` |
| Thermo Galactic | `.spc` | ☐ In core | — | Candidate for `spectrochempy-galactic` |
| Renishaw | WiRE (`.wdf`) | ☐ In core | — | Candidate for `spectrochempy-renishaw` |
| Surface Optics Corp | `.ddr`, `.hdr`, `.sdr` | ☐ In core | — | Candidate for `spectrochempy-soc` |
| Pfeiffer Vacuum | QUADERA (`.asc`, `.QD`) | ☐ In core | — | Candidate for `spectrochempy-quadera` |
| LABSPEC | `.txt` | ☐ In core | — | Low priority (simple text format) |
| JCAMP-DX | `.jdx`, `.dx` | ☐ In core | — | Domain-neutral; may stay in core |
| Agilent | Various | ☐ Not implemented | — | Requires sample files |
| UV/VIS | Various | ☐ Not implemented | — | Requires format specification |

## Reference model

The `spectrochempy-perkinelmer` plugin should serve as the reference
implementation for all future vendor I/O migrations:

- Standalone package with its own `pyproject.toml`
- Entry-point registration via `spectrochempy.plugins`
- `SpectroChemPyPlugin` subclass with `register_readers()`
- Reader integrated with `Importer()` and `@_importer_method`
- Public test data shipped in `spectrochempy_data`
- Tests using `PluginTestHarness`
- Core `features.py` entries for missing-plugin stubs (following existing
  convention until dynamic discovery is implemented)
- Public API follows the **Namespace API Convention**
  ([`maintainers/rfcs/namespace-api-convention.md`](../rfcs/namespace-api-convention.md))

## Blockers / prerequisites

1. **Dynamic plugin discovery for stubs**
   Migrate `KNOWN_PLUGIN_READERS` and `KNOWN_PLUGIN_NAMESPACES` from static
   lists to entry-point-driven discovery.  This would eliminate the need for
   any core edits when adding new official plugins.

2. **Test data licensing**
   Each vendor migration requires public, BSD-compatible sample files.
   Some vendors may not provide distributable test data.

3. **Maintainer bandwidth**
   Each migration is a focused PR but requires review and documentation updates.

## NMR reader strategy

The official NMR plugin (`spectrochempy-nmr`) currently provides only a
TopSpin/Bruker reader.  The audit
([`audit/~nmr-plugin-architecture-audit.md`](../audit/~nmr-plugin-architecture-audit.md))
identifies Varian/Agilent, JEOL, and NMRPipe as the priority candidates for
future readers.

The preferred implementation strategy for these formats follows the general
vendor-reader approach documented in
[`maintainers/architecture/reader-normalization-architecture.md`](../architecture/reader-normalization-architecture.md#vendor-reader-implementation-strategy):

- **nmrglue** remains an excellent technical reference. Its BSD license allows
  selective reuse of mature parser code with attribution.
- Future NMR readers should be implemented through **selective porting** of
  format-specific parser code into the official NMR plugin, rather than
  introducing a mandatory runtime dependency on the upstream nmrglue project.
- SpectroChemPy will maintain the integrated code according to its own release
  cadence and compatibility policy.
- The existing vendored nmrglue subset (`nmrglue.py` in the plugin tree) may
  serve as a reference for the porting pattern, including the license notice
  convention (`NMRGLUE_LICENSE.rst`).

This approach avoids coupling the NMR plugin's release cycle to nmrglue's
while still benefiting from its format knowledge.

## Open questions

- Should JCAMP-DX (domain-neutral, text-based) stay in core or move to a plugin?
- Should Bruker OPUS and Bruker TopSpin (NMR) be in separate plugins or a
  single `spectrochempy-bruker` plugin?
- Should the Carroucell reader (already a plugin) be renamed or consolidated?

## Related

- Issue #897 (PerkinElmer `.sp` plugin)
- `maintainers/architecture/tensor-plugin-migration.md` (plugin architecture)
