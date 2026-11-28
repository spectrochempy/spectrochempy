# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import struct


def synthetic_spg_members(members):
    """Build an SPG container with independently declared member headers."""
    start = 304 + 48 * len(members)
    content = bytearray(start + 768 * len(members))
    content[:18] = b"Spectral Data File"
    struct.pack_into("<H", content, 294, 3 * len(members))
    for index, member in enumerate(members):
        header = start + 768 * index
        values = member.get("data", [index + 1, index + 1.5])
        payload = header + 320
        label = header + 448
        records = [(2, header, 140), (3, payload, 4 * len(values)), (107, label, 260)]
        for offset, (key, position, length) in enumerate(records):
            entry = 304 + 16 * (3 * index + offset)
            struct.pack_into("<BBII", content, entry, key, 0, position, length)
        struct.pack_into("<I", content, header + 4, len(values))
        content[header + 8] = member.get("xcode", 1)
        content[header + 12] = member.get("ycode", 17)
        struct.pack_into(
            "<ff",
            content,
            header + 16,
            member.get("firstx", 4000),
            member.get("lastx", 3999),
        )
        struct.pack_into("<f", content, header + 80, 15798)
        struct.pack_into("<f", content, header + 188, 8.8617)
        struct.pack_into(f"<{len(values)}f", content, payload, *values)
        title = member.get("label", f"spectrum {index}").encode("latin-1")
        content[label : label + len(title)] = title
        struct.pack_into(
            "<I", content, label + 256, member.get("timestamp", 4_000_000_000 + index)
        )
    return bytes(content)


def synthetic_spa_with_optical_velocity(
    mirror,
    canonical=None,
    *,
    xunits=1,
    yunits=17,
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
    content[header + 12] = yunits
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


def synthetic_srs(*, xcode=1, ycode=17, background_xcode=1, background_ycode=17):
    """Build a minimal TG-style SRS with independent series/background headers."""
    content = bytearray(4096)
    content[:18] = b"Spectral Exte File"
    marker = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00"
    for header, xunit, yunit in (
        (400, xcode, ycode),
        (1800, background_xcode, background_ycode),
    ):
        struct.pack_into("<I", content, header + 4, 2)
        content[header + 8] = xunit
        content[header + 12] = yunit
        struct.pack_into("<ff", content, header + 16, 3999, 4000)
        struct.pack_into("<ff", content, header + 80, 15798, 1)
        struct.pack_into("<f", content, header + 188, 8.8617)
        content[header + 152 : header + 152 + len(marker)] = marker
    struct.pack_into("<ff", content, 400 + 1002, 0, 2)
    struct.pack_into("<I", content, 400 + 1026, 3)
    struct.pack_into("<ff", content, 1800 + 208, 7, 8)
    content[3200 : 3200 + len(marker)] = marker
    for index in range(3):
        position = 3260 + index * 108
        name = f"observation {index}".encode()
        content[position : position + len(name)] = name
        struct.pack_into("<ff", content, position + 84, index + 1, index + 2)
    return bytes(content)
