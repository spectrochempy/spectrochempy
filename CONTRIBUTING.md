# Contributing to SpectroChemPy

Full developer guide: [`docs/sources/devguide/`](docs/sources/devguide/)

---

## Quick start

### Using uv (Recommended)

```bash
git clone https://github.com/your-user-name/spectrochempy.git
cd spectrochempy
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
git clone https://github.com/your-user-name/spectrochempy.git
cd spectrochempy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Commit, Pull Request, and Developer Changelog Prefixes

Use the same prefix vocabulary for:

* commit messages;
* pull request titles;
* developer changelog entries.

Preferred prefixes:

| Prefix   | Usage                              |
| -------- | ---------------------------------- |
| `ENH:`   | User-facing enhancement or feature |
| `FIX:`   | Bug fix                            |
| `DOC:`   | Documentation changes              |
| `TEST:`  | Test addition or modification      |
| `CI:`    | CI/CD and workflow changes         |
| `DEV:`   | Developer tooling                  |
| `PERF:`  | Performance improvement            |
| `MAINT:` | Refactoring, cleanup, maintenance  |
| `REL:`   | Release-related work               |

Examples:

```text
MAINT: migrate CoordSet reshape lifecycle handling

FIX: preserve reference coordinates during native deserialization

TEST: add CoordSet serialization regression tests

CI: simplify core-only validation workflow
```

Avoid introducing alternative prefixes unless there is a clear project-wide
need.

### Automated checks

A GitHub Actions workflow (`pr-compliance.yml`) reads the prefix table above
from this file and enforces it on every pull-request title.  If a PR
intentionally needs a non-standard prefix, add the label
**`non-standard-prefix`** to bypass the check.

The same workflow verifies that `docs/sources/whatsnew/changelog.rst` has
been modified.  If the PR genuinely does not need a changelog entry (e.g.
pure CI or internal tooling change), add the label **`no-changelog`**.

For PRs that change only safe maintainer-policy or governance Markdown
documents (`AGENTS.md`, root `*.md`, `maintainers/**/*.md`, PR templates, and
similar non-executable repository-policy files), maintainers may add the label
**`safe-docs-no-ci`** to bypass the heavyweight test and docs workflows.

This label must **not** be used for:

* `docs/` changes, even when the files are non-Python;
* `examples/` changes;
* plugin Markdown such as `plugins/**/README.md`;
* code changes hidden inside a documentation PR;
* any change that can affect executable documentation, packaging, or runtime
  behavior.

See `docs/sources/devguide/contributing.rst` for the full PR workflow.

The PR description should begin with:

1. a summary of changes;
2. what is intentionally out of scope;
3. references to related issues and PRs when applicable.

If a pull request adds or exposes a new public API symbol at top level
(``spectrochempy.<name>`` / ``scp.<name>``), update
`docs/sources/reference/index.rst` in the appropriate section so the symbol
appears in the public API reference.

---

## Changelog entries

Edit:

```text
docs/sources/whatsnew/changelog.rst
```

Never edit:

```text
docs/sources/whatsnew/latest.rst
```

manually.

### User-facing changes

Place in the appropriate section **without a prefix**:

* `New Features`
* `Bug Fixes`
* `Dependency Updates`
* `Breaking Changes`
* `Deprecations`

### Developer changes

Place developer-oriented entries in the `Developer` section.

Use the same prefix vocabulary defined above.

Examples:

```text
MAINT: Refactored CoordSet internals around group-aware lookup and lifecycle
adapters in preparation for future storage migration. (#1234)

TEST: Added comprehensive regression coverage for CoordSet lookup,
serialization, aliases, references, and lifecycle operations. (#1234)

CI: Added a dedicated core-only validation workflow. (#1234)
```

Include the related GitHub issue or pull request number when available:

```text
(#1234)
```

Full changelog guide:

`docs/sources/devguide/contributing_codebase.rst`
(section *Documenting change log*).

### Multi-PR projects

For long-running projects implemented across multiple pull requests, avoid
adding one changelog entry per intermediate refactoring step.

Detailed implementation history belongs in audit notes, not in the release
changelog.

Developer changelog entries should summarize meaningful release-level outcomes
rather than individual implementation steps.

Prefer describing the architectural or maintenance result achieved.

Good:

```text
MAINT: Refactored CoordSet internals around group-aware lookup, lifecycle
adapters, and storage-migration preparation while preserving public behavior.

TEST: Added comprehensive regression coverage for CoordSet lookup,
serialization, references, aliases, and lifecycle operations.

CI: Simplified the validation matrix and introduced a dedicated core-only
workflow.
```

Avoid:

```text
MAINT: Extracted helper A
MAINT: Added adapter B
MAINT: Migrated internal function C
MAINT: Added metadata D
TEST: Added test for helper E
TEST: Added test for helper F
```

unless the individual change is independently meaningful to downstream
developers, contributors, or maintainers.

As a rule of thumb:

* audit files record implementation history;
* changelog entries record release-level outcomes.

---

## Linting & formatting

```bash
ruff check src/spectrochempy/
ruff format src/spectrochempy/
```

Configuration is defined in `pyproject.toml`.

SpectroChemPy uses Ruff only (no standalone Black or isort).

---

## Tests

```bash
# single test file
pytest tests/path/to/test_file.py

# single test
pytest tests/path/to/test_file.py::TestClass::test_method

# with markers
pytest tests -m "not slow and not network"
```

External test data requires:

```bash
export SCP_TEST_DATA_DOWNLOAD=1
```

See `tests/conftest.py` for available fixtures.

---

## Pre-commit

Install hooks:

```bash
pip install pre-commit
pre-commit install
```

Run once near PR packaging time (not repeatedly during intermediate work):

```bash
pre-commit run --all-files
```

Hooks currently include:

* Ruff linting and formatting;
* codespell repository spelling checks (configured by `.codespellrc`);
* requirements regeneration;
* lazy stub regeneration;
* version and release-note updates.

Generated files updated by pre-commit (for example `latest.rst`) belong in the
final PR state and should not be discarded after pre-commit runs.

---

## Plugin development

Plugins are separate packages located in `plugins/`.

Example:

```bash
pip install -e plugins/spectrochempy-nmr
```

Each plugin:

* has its own `pyproject.toml`;
* registers through
  `[project.entry-points."spectrochempy.plugins"]`;
* if it is an **official plugin** (published and maintained by the
  SpectroChemPy team), must declare its official status via:

  ```toml
  [tool.spectrochempy]
  official-plugin = true
  ```

  in its ``pyproject.toml`` — this marker is the sole registration point
  for CI workflows (publishing, testing, docs builds, release validation).
  Do **not** use a Trove classifier for this, as PyPI only accepts
  registered classifiers.

See:

```text
docs/sources/devguide/plugins/
```

for details.

---

## Additional references

* Full developer guide: `docs/sources/devguide/`
* Maintainer documentation: `maintainers/` (release and recovery procedures)
* AGENTS.md: concise guidance for AI-assisted development
