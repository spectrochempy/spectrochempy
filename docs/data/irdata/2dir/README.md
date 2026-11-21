# Experimental 2D-IR: D-silicalite at 210 °C

`silicalite_210C_0ps_2DIR.csv` contains the parallel-polarization detector
signal measured at a waiting time of 0.000 ps. The first row contains the pump
wavenumbers; the first column contains the probe wavenumbers. The remaining
values are the measured difference absorbance.

`silicalite_210C_0ps_2DIR.scp` contains the same cropped matrix in native
SpectroChemPy format, with probe and pump coordinates and signal units attached.
It is derived directly from the CSV above without further signal processing.

## Provenance and license

Donaldson, Hawkins & Howe, “Distinctive signatures and ultrafast dynamics of
Brønsted sites, silanol nests and adsorbed water in zeolites revealed by 2D-IR
spectroscopy,” *Chemical Science* (2025),
[doi:10.1039/D4SC08093A](https://doi.org/10.1039/D4SC08093A).

The source data are from [STFC eData, DOI 10.5286/edata/938](https://doi.org/10.5286/edata/938),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source file is `Figure 3/D-Silicalite 210C act 1500s av_waiting times_23-05-2023_2DIR/D-Silicalite_210C_0.000ps_2DIR.csv`.

## Preparation

The source file contains parallel and perpendicular detector blocks; this
example keeps the first (parallel) 128 probe rows. The pump axis is restricted
to 2176–2831 cm⁻¹ to retain the measured positive-frequency region and remove
the mirrored half. No baseline correction, smoothing, or normalization has
been applied. The diagonal bleach is negative in this dataset.
