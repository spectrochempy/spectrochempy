# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import struct


def synthetic_spa_with_optical_velocity(
    mirror,
    canonical=None,
    *,
    xunits=1,
    reference_frequency=15798.0,
    raman_frequency=0.0,
    timestamp=0,
    library=False,
    scan_points=0,
    peak_position=0,
    sample_scans=0,
    fft_points=0,
    trailing_geometry=0,
    background_scans=0,
    background_gain=0.0,
    aperture=0.0,
    digitizer_bits=0,
    high_pass=0.0,
    low_pass=0.0,
    sample_gain=0.0,
    native_history=None,
):
    """Build a minimal SPA byte stream with optional native metadata blocks."""
    content = bytearray(1024)
    content[:18] = b"Spectral Data File"
    content[30:42] = b"synthetic.spa"

    records = [(2, 400, 140)]
    if canonical is not None:
        records.append((106, 700, 56))
    if library:
        records.append((0x53, 0, 0))
    payload_position = 756 if canonical is not None else 700
    if library and canonical is None:
        payload_position = 700
    records.append((3, payload_position, 8))
    if native_history is not None:
        history = native_history.encode("latin-1") + b"\x00"
        records.append((27, 900, len(history)))
        content[900 : 900 + len(history)] = history
    struct.pack_into("<H", content, 294, len(records))
    struct.pack_into("<I", content, 296, timestamp)
    for offset, (key, position, length) in zip(
        range(304, 304 + 16 * len(records), 16), records, strict=True
    ):
        struct.pack_into("<BBII", content, offset, key, 0, position, length)
    if library:
        content[304 + 16 * len(records)] = 1

    header = 400
    struct.pack_into("<I", content, header + 4, 2)
    content[header + 8] = xunits
    content[header + 12] = 17
    struct.pack_into("<ff", content, header + 16, 4000.0, 3999.0)
    struct.pack_into("<I", content, header + 28, scan_points)
    struct.pack_into("<I", content, header + 32, peak_position)
    struct.pack_into("<I", content, header + 36, sample_scans)
    struct.pack_into("<I", content, header + 44, fft_points)
    struct.pack_into("<I", content, header + 48, trailing_geometry)
    struct.pack_into("<I", content, header + 52, background_scans)
    struct.pack_into("<f", content, header + 56, background_gain)
    struct.pack_into("<I", content, header + 68, 100)
    struct.pack_into("<f", content, header + 80, reference_frequency)
    struct.pack_into("<f", content, header + 84, 1.0)
    struct.pack_into("<f", content, header + 92, aperture)
    struct.pack_into("<f", content, header + 96, raman_frequency)
    struct.pack_into("<f", content, header + 188, mirror)
    if canonical is not None:
        struct.pack_into("<I", content, 700 + 16, digitizer_bits)
        struct.pack_into("<f", content, 700 + 20, high_pass)
        struct.pack_into("<f", content, 700 + 24, low_pass)
        struct.pack_into("<f", content, 700 + 44, sample_gain)
        struct.pack_into("<f", content, 700 + 48, canonical)
    struct.pack_into("<ff", content, payload_position, 1.0, 2.0)
    return bytes(content)
