# Tensor plugin migration audit

## Scope

Move CP/PARAFAC decomposition out of core into a dedicated official tensor
plugin while keeping compatibility paths for existing users.

## Decisions

- Use the existing official plugin package convention:
  `plugins/spectrochempy-tensor`, Python package `spectrochempy_tensor`, public
  namespace `scp.tensor`.
- Keep the copied CP implementation behavior-preserving and place it under
  `spectrochempy_tensor.decompositions.cp`.
- Reserve `spectrochempy_tensor.adapters` for future bridges between TensorLy
  objects and SpectroChemPy datasets.
- Remove `CP` from the core root lazy-import map so `scp.CP` is no longer a
  core symbol. The tensor plugin provides a deprecated root export for
  compatibility.
- Keep `spectrochempy.analysis.decomposition.cp.CP` as a deprecated shim that
  imports from `spectrochempy_tensor`.
- Keep `spectrochempy[cp]` as a compatibility extra, redirected to
  `spectrochempy[tensor]`.

## Dependency boundary

- Core no longer imports TensorLy and no longer declares TensorLy directly in
  `pyproject.toml`.
- TensorLy is declared by `plugins/spectrochempy-tensor/pyproject.toml` and the
  plugin conda recipe.
- `requirements/requirements_cp.txt` and `environments/environment_cp.yml` are
  generated from `pyproject.toml`; they still need regeneration by the standard
  pre-commit workflow.

## Tests

- Core CP tests now cover compatibility and missing-plugin behavior only.
- CP behavior tests moved to `plugins/spectrochempy-tensor/tests/test_cp.py`.
- Tensor plugin lifecycle and registration tests live in
  `plugins/spectrochempy-tensor/tests/test_plugin.py`.

## Remaining follow-up

- Regenerate generated dependency files with the normal pre-commit workflow.
- Add future TensorLy-derived classes under `spectrochempy_tensor.decompositions`
  and shared conversion helpers under `spectrochempy_tensor.adapters`.
