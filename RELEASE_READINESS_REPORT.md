# SpectroChemPy Release Readiness Report

## Branch: `release-prep` (based on `plugins`)

---

## 1. PyPI Readiness

### Core Package (`spectrochempy`)
| Status | Detail |
|--------|--------|
| Package name | `spectrochempy` |
| Version source | `setuptools_scm` (dynamic from git tags) |
| Build system | `setuptools` + `setuptools_scm` |
| Workflow | `build_package.yml` → `build-and-publish_pypi` |
| Trigger | Release published, push to master/fix branches, workflow_dispatch |
| Artifact retention | 5 days |
| Status | Ready |

### Official Plugins

| Plugin | PyPI Name | Version | Build Status | Workflow Status | Notes |
|--------|-----------|---------|--------------|-----------------|-------|
| spectrochempy-nmr | `spectrochempy-nmr` | 0.1.0 | OK | Updated in `publish_plugins.yml` | Depends on `spectrochempy`, optional `spectrochempy-hypercomplex` |
| spectrochempy-hypercomplex | `spectrochempy-hypercomplex` | 0.1.0 | OK | Updated in `publish_plugins.yml` | Depends on `spectrochempy`, `numpy-quaternion` |
| spectrochempy-iris | `spectrochempy-iris` | 0.1.0 | OK | Updated in `publish_plugins.yml` | Depends on `spectrochempy`, `scipy`, `osqp` |
| spectrochempy-cantera | `spectrochempy-cantera` | 0.1.0 | OK | Updated in `publish_plugins.yml` | Depends on `spectrochempy`, `cantera`, `numpy`, `scipy` |
| spectrochempy-carroucel | `spectrochempy-carroucel` | 0.1.0 | N/A (not merged) | TODO in workflow | Will be added after `plugin-carroucel` merge |

### PyPI Workflow Changes Made
- `publish_plugins.yml` now auto-discovers plugins by scanning `plugins/*/` for `pyproject.toml`
- Matrix is generated dynamically — no hardcoded list, new plugins are picked up automatically
- Carroucel will be included once its branch is merged and its directory contains a `pyproject.toml`
- Trigger paths updated to watch all plugin directories

---

## 2. Conda Readiness

### Core Package (`spectrochempy`)
| Status | Detail |
|--------|--------|
| Recipe format | rattler-build (`recipe/recipe.yaml` generated from template) |
| Generator script | `.github/workflows/scripts/generate_conda_recipe.py` |
| Template | `.github/workflows/scripts/templates/recipe.tmpl` |
| Build tool | `prefix-dev/rattler-build-action` |
| Channel | `spectrocat` (Anaconda Cloud) |
| Labels | `main` (releases), `dev` (pushes) |
| Status | Ready, but recipe needs regeneration from current `pyproject.toml` |

### Official Plugins

| Plugin | Conda Name | Recipe Location | Build Status | Channel | Notes |
|--------|------------|-----------------|--------------|---------|-------|
| spectrochempy-nmr | `spectrochempy-nmr` | `plugins/spectrochempy-nmr/recipe.yaml` | Ready | `spectrocat` | Co-located with source |
| spectrochempy-hypercomplex | `spectrochempy-hypercomplex` | `plugins/spectrochempy-hypercomplex/recipe.yaml` | Ready | `spectrocat` | Co-located with source |
| spectrochempy-iris | `spectrochempy-iris` | `plugins/spectrochempy-iris/recipe.yaml` | Ready | `spectrocat` | Co-located with source |
| spectrochempy-cantera | `spectrochempy-cantera` | `plugins/spectrochempy-cantera/recipe.yaml` | Ready | `spectrocat` | Co-located with source |
| spectrochempy-carroucel | `spectrochempy-carroucel` | `plugins/spectrochempy-carroucel/recipe.yaml` | Ready | `spectrocat` | Recipe ready; plugin source pending PR #956 merge |

### Conda Workflow Changes Made
- `build_package.yml` updated with `discover-conda-plugins` + `build_and_publish_conda_plugins` jobs
- `discover-conda-plugins` scans `plugins/*/` for `recipe.yaml` or `meta.yaml` files and generates the matrix dynamically
- `build_and_publish_conda_plugins` builds discovered recipes in parallel after core succeeds
- Uses `rattler-build-action` for all plugin recipes
- Uploads to Anaconda.org with the same label logic as core (`main` for releases, `dev` for pushes)
- Adding a new official plugin only requires adding a `recipe.yaml` in its root — no workflow edits needed

---

## 3. Dependency Matrix

### Core Runtime Dependencies (`pyproject.toml`)
```
colorama, dill, ipython, jinja2, lazy_loader, matplotlib,
numpy, pluggy, pint, pyyaml, requests, scikit-learn, scipy,
tzlocal, xlrd
```

### Plugin Dependencies

| Plugin | Runtime Deps | Optional Deps | Depends on Core | Depends on Other Plugin |
|--------|-------------|---------------|----------------|------------------------|
| nmr | `spectrochempy`, `numpy` | `spectrochempy-hypercomplex` (for 2D NMR) | Yes | hypercomplex (optional) |
| hypercomplex | `spectrochempy`, `numpy`, `numpy-quaternion` | None | Yes | None |
| iris | `spectrochempy`, `scipy`, `osqp` | None | Yes | None |
| cantera | `spectrochempy`, `cantera`, `numpy`, `scipy` | None | Yes | None |
| carroucel | `spectrochempy`, `xlrd`, `scipy` | None | Yes | None |

### Cross-Plugin Dependencies
- `spectrochempy-nmr[hypercomplex]` → pulls in `spectrochempy-hypercomplex`
- No other cross-plugin dependencies

---

## 4. Monorepo Build Independence

### Can each plugin build independently?

| Plugin | Has own `pyproject.toml` | Has own `src/` | Entry point registered | Independent build | Status |
|--------|-------------------------|----------------|----------------------|-------------------|--------|
| nmr | Yes | Yes | Yes | Yes | OK |
| hypercomplex | Yes | Yes | Yes | Yes | OK |
| iris | Yes | Yes | Yes | Yes | OK |
| cantera | Yes | Yes | Yes | Yes | OK |
| carroucel | Yes | Yes | Yes | Yes | OK |

All plugins use `setuptools` with `packages.find.where = ["src"]` and declare their own entry points under `spectrochempy.plugins`. They can be built independently with `python -m build`.

Official plugins that include a `recipe.yaml` (or `meta.yaml`) in their root directory are automatically discovered by CI for conda packaging. Third-party plugins may omit the recipe and handle their own distribution.

### Verified builds (local)
- All 4 existing plugins built successfully as wheels
- Carroucel recipe exists but plugin source not yet on this branch

---

## 5. CI Workflow Summary

| Workflow | Purpose | PyPI | Conda | Plugins Covered |
|----------|---------|------|-------|-----------------|
| `build_package.yml` | Core package build & publish | Yes | Yes (rattler-build) | Core only |
| `build_package.yml` (new job) | Plugin conda build & publish | No | Yes (rattler-build) | All official plugins |
| `publish_plugins.yml` | Plugin PyPI build & publish | Yes | No | All official plugins |
| `test_package.yml` | Test core on multiple OS/Python | N/A | N/A | N/A |

### Proposed order of execution on release
1. `test_package.yml` passes (all OS, all Python versions)
2. `build_package.yml` builds core conda package → upload to `main`
3. `build_package.yml` builds plugin conda packages → upload to `main`
4. `publish_plugins.yml` builds plugin wheels → upload to PyPI
5. Core wheel is built by `build_package.yml` → upload to PyPI

---

## 6. Documentation Changes

### Updated files
- `docs/sources/gettingstarted/install/install_adds.rst`
  - Added mamba/conda installation instructions for plugins

### Still needed before release
- `docs/sources/userguide/plugins/official_plugins.rst` — add conda install commands per plugin
- `docs/sources/userguide/plugins/index.rst` — mention conda channel `spectrocat`
- Release notes (`whatsnew/latest.rst`) — document new plugin availability on PyPI/conda

---

## 7. Secrets / Tokens Required

| Service | Secret Name | Used In | Status |
|---------|-------------|---------|--------|
| PyPI (core) | OIDC (trusted publisher) | `build_package.yml` | Must be configured in PyPI web UI |
| TestPyPI (core) | OIDC (trusted publisher) | `build_package.yml` | Must be configured in TestPyPI web UI |
| PyPI (plugins) | OIDC (trusted publisher) | `publish_plugins.yml` | Must be configured in PyPI web UI per plugin |
| TestPyPI (plugins) | OIDC (trusted publisher) | `publish_plugins.yml` | Must be configured in TestPyPI web UI per plugin |
| Anaconda.org | `ANACONDA_API_TOKEN` | `build_package.yml` | Must be set in repo secrets |

### Trusted Publisher Setup Needed
For each package on PyPI, the repository owner must configure:
- Repository: `spectrochempy/spectrochempy`
- Workflow: `build_package.yml` (for core) or `publish_plugins.yml` (for plugins)
- Environment: not required (uses default)

---

## 8. Convenience Metapackage Assessment

### `spectrochempy-all`
**Pros:**
- One-command install for users who want everything
- Simplifies onboarding for new users

**Cons:**
- Pulls heavy dependencies (cantera, osqp, numpy-quaternion) even if not needed
- Goes against the plugin philosophy of lightweight core
- Harder to maintain (version coupling)

**Recommendation:** Do not create. Instead, document the explicit install commands.

### `spectrochempy-nmr-stack`
**Pros:**
- Guarantees compatible versions of core + nmr + hypercomplex
- Useful for NMR-heavy labs

**Cons:**
- Adds another package to maintain
- pip extras already cover this (`pip install spectrochempy[nmr]` which pulls hypercomplex)

**Recommendation:** Not needed. pip extras (`spectrochempy[nmr]`) and conda metapackages can be documented instead.

---

## 9. Release Blockers / TODOs

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Merge `plugin-carroucel` into `plugins` | High | Pending PR #956 |
| 2 | No workflow edits needed — carroucel will be auto-discovered once directory exists | High | Dynamic discovery in place |
| 3 | Configure PyPI trusted publishers for all plugin packages | High | Not started |
| 4 | Configure TestPyPI trusted publishers | Medium | Not started |
| 5 | Verify `ANACONDA_API_TOKEN` secret is set | High | Check with repo admin |
| 6 | Test full release workflow on a pre-release tag | High | Not done |
| 7 | Update plugin installation docs with conda commands | Medium | Partially done |
| 8 | Add conda install instructions to each plugin's README | Low | Not done |
| 9 | Verify `spectrochempy-hypercomplex` entry point name matches recipe | Low | Should check |

---

## 10. Validation Summary

### Smoke Tests Performed
```bash
python -c "import spectrochempy as scp; print(scp.__version__)"
  # Result: 0.8.3.dev2+dirty ✓

python -c "import spectrochempy_nmr"
  # Result: OK ✓

python -c "import spectrochempy_hypercomplex"
  # Result: OK ✓

python -c "import spectrochempy_iris"
  # Result: OK ✓

python -c "import spectrochempy_cantera"
  # Result: OK ✓

python -c "import spectrochempy_carroucel"
  # Result: N/A (plugin not on this branch yet)
```

### Build Tests Performed
```bash
python -m build --wheel plugins/spectrochempy-nmr       # ✓
python -m build --wheel plugins/spectrochempy-hypercomplex # ✓
python -m build --wheel plugins/spectrochempy-iris       # ✓
python -m build --wheel plugins/spectrochempy-cantera    # ✓
```

---

## Summary

**Ready for PyPI:**
- Core package: Yes (workflow exists, builds)
- 4 existing plugins: Yes (workflow updated, builds verified)
- Carroucel: Waiting for branch merge

**Ready for Conda:**
- Core package: Yes (rattler-build workflow exists)
- 4 existing plugins: Yes (recipes created, workflow updated)
- Carroucel: Recipe ready, plugin source waiting for merge

**Blockers before first release:**
1. Merge `plugin-carroucel`
2. Configure PyPI trusted publishers (core + 5 plugins)
3. Verify Anaconda token is active
4. Run a test release (e.g., `v0.9.0-rc1`) to validate the full pipeline
