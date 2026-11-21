# Documentation data

This directory contains the public input files used by the tutorials and galleries.
The documentation build reads them directly, without downloading or extracting
`spectrochempy_data`. Internal paths match the original `testdata/` layout.

`manifest.json` records each fixture's size and SHA-256, the source data repository
and its exact base commit. The build checks these hashes before execution.
To update a fixture, copy the canonical file, preserve its notices, update the
manifest and rebuild the documentation. Keep the selection limited to files
needed by the documentation. Do not edit or generate output files here.

Most fixtures are copied unchanged from
[spectrochempy_data](https://github.com/spectrochempy/spectrochempy_data)
at the revision recorded in the manifest. The Renishaw WDF files originate from
[py-wdf-reader](https://github.com/alchem0x2A/py-wdf-reader) (MIT license).
The Hoyer and Hess Raman fixture is copied unchanged from
[Zenodo record 13929684](https://doi.org/10.5281/zenodo.13929684), under CC BY-SA 4.0;
see its [provenance notice](ramandata/hoyer_hess_2024/README.md).
Existing per-directory notices apply to their respective data.

Corn is not included. Supply it separately using `SCPY_CORN_DATA` and
`SCPY_CORN_ZIP`, as described in the documentation contributor guide.
These fixtures are excluded from the Python wheel and source distribution.
