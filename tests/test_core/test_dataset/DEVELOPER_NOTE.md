# Test Organization — `tests/test_core/test_dataset/`

## Layout

The original monolithic `test_dataset.py` was split into focused modules:

| File | Covers |
|------|--------|
| `test_dataset_creation.py` | Init/construction (1D, 2D, dtype, copy, filename, title, docstring) |
| `test_dataset_slicing.py` | All slicing variants (index, label, value, quantity, coords, out-of-limits) |
| `test_dataset_masking.py` | Mask creation, validation, propagation, operations with masked arrays |
| `test_dataset_units.py` | Unit assignment, conversion, absorbance↔transmittance |
| `test_dataset_coords.py` | CoordSet init, indexing, manipulation, sorting, multi-axis |
| `test_dataset_metadata.py` | `str`, `repr`, `_repr_html_`, `meta`, `timezone` |
| `test_dataset_complex.py` | Complex data: real/imag, slicing, operations, init with mask |
| `test_dataset_math.py` | Basic math, reductions, broadcasting, swapdims, transpose, apply, take |
| `test_dataset_squeeze.py` | Squeeze coord propagation (all dims, partial, multiples) |
| `test_dataset_regressions.py` | Historical bug reproducers (bug_462, bug_arnaud, etc.) |

## Rules for adding tests

1. **Put the test in the most specific file.** If a test spans categories, use judgment on the primary concern (e.g., a slicing test that also checks units goes in `test_dataset_slicing.py`).

2. **Do not add new external-data dependencies.** All tests should use synthetic fixtures or module-level constants. Tests that require downloaded data (`IR_dataset_1D`, `wodger.spg`) stay in `test_dataset_regressions.py` and are skipped when data is absent.

3. **Fixtures** live in two conftest files:
   - `tests/conftest.py` — dataset-level fixtures (`ds1`, `nd1d`, `nd2d`, `dsm`, `ref_ds`, `coord0`–`coord2`, `IR_dataset_1D`)
   - `tests/test_core/conftest.py` — additional fixtures for core tests

4. **Avoid circular imports.** Each test file imports `scp` (the package) and the specific utilities it needs. Do not import from sibling test files.

5. **Preserve the parametrize pattern.** Parametrized tests (e.g., `test_nddataset_1D_NDDataset`) keep their parametrize decorator in the new file.

6. **Do not import from `test_dataset_regressions.py`** into other test files. Regression tests are standalone reproducers.

## Before/After comparison (Phase 6)

- Files: 1 → 10 (+test_coord.py, +test_coordset.py unchanged)
- Test count: 83 → 83 (identical)
- Pass count: 83 → 83 (identical)
- Runtime: 49.66s → 52.48s (within normal variation for 10-file discovery)
