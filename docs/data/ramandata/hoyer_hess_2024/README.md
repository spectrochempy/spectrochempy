# Hoyer and Hess Raman modulation-excitation data (2024)

This directory contains the unchanged raw Raman series used by SpectroChemPy's
modulation-excitation and neighbouring-acquisition despiking examples.

## Attribution and source

Henrik Hoyer and Christian Hess, *Dataset of "Identifying Active Oxygen and
Copper Species in Cu/CeO2 Catalysts using Raman and UV-Vis Modulation Excitation
Spectroscopy"*, version 1.0.0, Zenodo, published 15 October 2024.

- Dataset DOI: https://doi.org/10.5281/zenodo.13929684
- Related publication: https://doi.org/10.1021/acs.jpcc.4c05826
- Source archive: https://zenodo.org/records/13929684/files/data.zip
- Licence: **Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0)**, https://creativecommons.org/licenses/by-sa/4.0/
- Zenodo metadata recording the licence: https://zenodo.org/api/records/13929684

Only the two files below were extracted. Their bytes have not been modified.
The original `description.txt` also describes files that remain available in
the complete Zenodo archive but are not mirrored here. This licence applies
to this dataset, not to other data in spectrochempy_data.

| Local file | Original archive member | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `Fig_2-a-c_raw-data.txt` | `data/Fig_2-a-c_raw-data.txt` | 9782758 | `38eef3bf7f197c7d34045d78eeac9236bca80ffad02ae2b2bed0e3ca9d2988c5` |
| `description.txt` | `data/description.txt` | 5015 | `5fde07ed4f30f502b90f9094548c11503b3486852931059d416fa951e8a860a2` |

Archive checksums, retrieved 20 November 2026:

- Zenodo-supplied MD5: `c5c091dcc693ccd372d3ad20f3b24e3e`
- SHA-256: `ddd5d6d711a9b2f8d04e91b106ece1d22050caca28467fb605261e467de4b56a`
- Size: 10835841 bytes.

## Layout

The first row gives Raman shifts in cm^-1. Each later row starts with the
acquisition time as a fraction of a day, followed by the Raman intensities.
See the original `description.txt` for the experiment and processing context.

The examples read this local file from
`scp.preferences.datadir / "ramandata/hoyer_hess_2024/Fig_2-a-c_raw-data.txt"`.
The shared documentation-data preparation supplies the file once per build;
the two examples do not need to download the complete Zenodo archive separately.
