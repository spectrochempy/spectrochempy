# Contributing to SpectroChemPy

Full developer guide: [`docs/sources/devguide/`](docs/sources/devguide/)

---

## Quick start

```bash
git clone https://github.com/your-user-name/spectrochempy.git
cd spectrochempy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Commit message prefixes

```
ENH:    Enhancement / new feature
FIX:    Bug fix
DOC:    Documentation only
TEST:   Test addition or change
BUILD:  Build system, packaging, CI
PERF:   Performance improvement
MAINT:  Refactoring, cleanup, maintenance
```

See `docs/sources/devguide/contributing.rst` for the full PR workflow.

---

## PR title prefixes

Use the same prefix style as commit messages:

| Prefix     | Usage                              |
|------------|------------------------------------|
| `MAINT:`   | Refactoring, cleanup               |
| `FIX:`     | Bug fix (including test/CI fixes)  |
| `DOC:`     | Documentation changes              |
| `TEST:`    | Test addition or change            |
| `CI:`      | CI/CD workflow changes             |
| `DEV:`     | Developer tooling                   |
| `FEATURE:` | New feature                        |

The PR description should begin with a summary of changes, state what is out
of scope, and reference the related issue/PR when applicable.

---

## Changelog entries

Edit `docs/sources/whatsnew/changelog.rst`. Never edit `latest.rst` manually.

### User-facing changes

Place in the appropriate section **without a prefix**:

- `New Features`
- `Bug Fixes`
- `Dependency Updates`
- `Breaking Changes`
- `Deprecations`

### Developer changes

Place in the `Developer` section with one of these prefixes:

| Prefix   | When to use                          |
|----------|--------------------------------------|
| `FEATURE`| New dev-facing capability            |
| `FIX`    | Test, CI, or internal bug fix        |
| `MAINT`  | Refactoring, cleanup                 |
| `CI`     | CI/CD workflow changes               |
| `DEV`    | Developer tooling (bump scripts, …)  |

Include the GitHub issue/PR number: `(#1234)`.

Full changelog guide: `docs/sources/devguide/contributing_codebase.rst` (section *Documenting change log*).

---

## Linting & formatting

```bash
ruff check src/spectrochempy/
ruff format src/spectrochempy/
```

Config in `pyproject.toml` — ruff only (no black, no isort standalone).

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

External test data requires `SCP_TEST_DATA_DOWNLOAD=1`. See `tests/conftest.py` for available fixtures.

---

## Pre-commit

Install hooks:
```bash
pip install pre-commit && pre-commit install
```

Run once near PR packaging time (not repeatedly during intermediate work):
```bash
pre-commit run --all-files
```

Hooks: ruff (lint+format), regenerate requirements, regenerate lazy stubs, update version/release notes.

Generated files updated by pre-commit (e.g. ``latest.rst``) belong in the final
PR state — do not discard them after pre-commit runs.

---

## Plugin development

Plugins are separate packages in `plugins/`. Install with:
```bash
pip install -e plugins/spectrochempy-nmr
```

Each plugin has its own `pyproject.toml` and registers via `[project.entry-points."spectrochempy.plugins"]`. See `docs/sources/devguide/plugins/` for details.

---

## Additional references

- [Full developer guide](docs/sources/devguide/)
- [Maintainer docs](maintainers/) (release process, package publication, incident recovery)
- [AGENTS.md](AGENTS.md) (concise reference for AI-assisted development — covers workflow, conventions, and common pitfalls)
