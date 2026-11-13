These files are original infrared polystyrene reference spectra from the
JCAMP-DX Test Disk (versions 1.04 and 1.05), mirrored in the
[Robert J. Lancashire collection](https://github.com/nzhagen/jcamp/tree/9a4af7d/data/Robert_J_Lancashire_test_data).
Both headers declare `OWNER= Public Domain`. The export metadata identify
the Digilab 3240 instrument. The files represent the same 1,844 encoded
ordinates as FIX (explicit ASCII numbers) and DIFDUP (compressed differences).
They declare different YFACTORS and slightly different X endpoints.
The copies use the equivalent `.dx` extension, LF line endings, and no
trailing whitespace. The extension keeps them outside the test suite’s
cleanup of generated `.jdx` exports.

| File | SHA-256 of upstream bytes (CRLF) |
|---|---|
| jtpolys.jdx | `7fda98c9ce64e39c92fb627de3440568f6bed6efa883a0e1e326458761bfb7ed` |
| jtpolysd.jdx | `7bd79681a82ab5363a38384c4d0f1dd4d77dd0b72829814d3f3a51d0d73f19b9` |
