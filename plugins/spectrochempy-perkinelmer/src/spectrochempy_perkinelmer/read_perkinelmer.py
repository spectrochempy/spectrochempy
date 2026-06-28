# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
PerkinElmer .sp file reader for SpectroChemPy.

Parsing logic adapted from the BSD-3-Clause licensed specio project.
Original implementation Copyright (c) 2017 Guillaume Lemaitre.
See: https://pypi.org/project/specio/

The PerkinElmer .sp format is a binary format storing IR spectroscopic data.
It uses nested blocks identified by unsigned-short IDs with signed-int sizes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid

__all__ = ["read_perkinelmer"]


# ------------------------------------------------------------------------------
# Block helpers
# ------------------------------------------------------------------------------


def _read_block_header(content: bytes, offset: int) -> tuple[int, int]:
    """Read a block ID (unsigned short) and size (signed int)."""
    return struct.unpack("<Hi", content[offset : offset + 6])


# ------------------------------------------------------------------------------
# Block decoders for metadata blocks
# ------------------------------------------------------------------------------


def _decode_5104(data: bytes) -> dict:
    """
    Decode block 5104 (metadata strings).

    Adapted from specio (BSD-3-Clause).
    """
    text = []
    start_byte = 0
    while start_byte + 2 < len(data):
        tag = data[start_byte : start_byte + 2]
        if tag == b"#u":
            start_byte += 2
            text_size = struct.unpack("<h", data[start_byte : start_byte + 2])[0]
            start_byte += 2
            text.append(
                data[start_byte : start_byte + text_size].decode(
                    "utf-8", errors="replace"
                )
            )
            start_byte += text_size
            start_byte += 6
        elif tag == b"$u":
            start_byte += 2
            text.append(struct.unpack("<h", data[start_byte : start_byte + 2])[0])
            start_byte += 2
            start_byte += 6
        elif tag == b",u":
            start_byte += 2
            text.append(struct.unpack("<h", data[start_byte : start_byte + 2])[0])
            start_byte += 2
        else:
            start_byte += 1

    # Defensive indexing – files may have fewer fields than expected.
    def _get(idx, default=""):
        try:
            return text[idx]
        except IndexError:
            return default

    return {
        "analyst": _get(0),
        "date": _get(2),
        "image_name": _get(4),
        "instrument_model": _get(5),
        "instrument_serial_number": _get(6),
        "instrument_software_version": _get(7),
        "accumulations": _get(9),
        "detector": _get(11),
        "source": _get(12),
        "beam_splitter": _get(13),
        "apodization": _get(15),
        "spectrum_type": _get(16),
        "beam_type": _get(17),
        "phase_correction": _get(20),
        "ir_accessory": _get(26),
        "igram_type": _get(28),
        "scan_direction": _get(29),
        "background_scans": _get(32),
    }


def _decode_25739(data: bytes) -> dict:
    """Decode block 25739 (file path)."""
    start_byte = 0
    n_bytes = 2
    var_id = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
    if var_id == 29987:
        start_byte += n_bytes
        n_bytes = 2
        var_size = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
        start_byte += n_bytes
        n_bytes = var_size
        return {
            "file_path": data[start_byte : start_byte + n_bytes].decode(
                "utf-8", errors="replace"
            )
        }
    return {}


def _decode_35698(data: bytes) -> dict:
    """Decode block 35698 (wavelength range)."""
    start_byte = 0
    n_bytes = 2
    var_id = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
    if var_id == 29981:
        start_byte += n_bytes
        n_bytes = 16
        min_wavelength, max_wavelength = struct.unpack(
            "<dd", data[start_byte : start_byte + n_bytes]
        )
        return {"min_wavelength": min_wavelength, "max_wavelength": max_wavelength}
    return {}


def _decode_35699(data: bytes) -> dict:
    """Decode block 35699 (absolute range)."""
    start_byte = 0
    n_bytes = 2
    var_id = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
    if var_id == 29981:
        start_byte += n_bytes
        n_bytes = 16
        min_absolute, max_absolute = struct.unpack(
            "<dd", data[start_byte : start_byte + n_bytes]
        )
        return {"min_absolute": min_absolute, "max_absolute": max_absolute}
    return {}


def _decode_35700(data: bytes) -> dict:
    """Decode block 35700 (wavelength step)."""
    start_byte = 0
    n_bytes = 2
    var_id = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
    if var_id == 29979:
        start_byte += n_bytes
        n_bytes = 8
        wavelength_step = struct.unpack("<d", data[start_byte : start_byte + n_bytes])[
            0
        ]
        return {"wavelength_step": wavelength_step}
    return {}


def _decode_35701(data: bytes) -> dict:
    """Decode block 35701 (number of points)."""
    start_byte = 0
    n_bytes = 2
    var_id = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
    if var_id == 29995:
        start_byte += n_bytes
        n_bytes = 4
        n_points = struct.unpack("<I", data[start_byte : start_byte + n_bytes])[0]
        return {"n_points": n_points}
    return {}


def _decode_35708(data: bytes) -> np.ndarray:
    """Decode block 35708 (spectrum data)."""
    start_byte = 0
    n_bytes = 2
    var_id = struct.unpack("<H", data[start_byte : start_byte + n_bytes])[0]
    if var_id == 29974:
        start_byte += n_bytes
        n_bytes = 4
        var_size = struct.unpack("<I", data[start_byte : start_byte + n_bytes])[0]
        start_byte += n_bytes
        n_bytes = var_size
        return np.frombuffer(data[start_byte : start_byte + n_bytes], dtype=np.float64)
    return np.array([])


_FUNC_DECODE = {
    25739: _decode_25739,
    35698: _decode_35698,
    35699: _decode_35699,
    35700: _decode_35700,
    35701: _decode_35701,
    35708: _decode_35708,
}


# ------------------------------------------------------------------------------
# File parser
# ------------------------------------------------------------------------------


class _SpFile:
    """Parser for a single PerkinElmer .sp file."""

    def __init__(self, content: bytes):
        self.signature = content[:4]
        if self.signature != b"PEPE":
            raise ValueError(
                f"Not a valid PerkinElmer .sp file: expected signature b'PEPE', got {self.signature!r}"
            )

        self.description = content[4:44].decode("utf-8", errors="replace").strip("\x00")

        self.meta: dict = {
            "signature": self.signature.decode("ascii"),
            "description": self.description,
        }
        self.spectrum: np.ndarray = np.array([])

        # Navigate nested blocks starting after the description.
        self._parse_blocks(content)

        # Build wavelength axis if the required blocks were found.
        if all(
            k in self.meta for k in ("min_wavelength", "max_wavelength", "n_points")
        ):
            self.wavelength = np.linspace(
                self.meta["min_wavelength"],
                self.meta["max_wavelength"],
                self.meta["n_points"],
            )
        else:
            self.wavelength = np.array([])

    def _parse_blocks(self, content: bytes) -> None:
        """
        Walk nested blocks using a hybrid strategy.

        The specio stack algorithm correctly navigates the nested hierarchy
        to find block 122/5104 (metadata strings), but it can terminate
        early and miss the flat numeric blocks (35698, 35700, 35701, 35708)
        that follow block 122 in the same outer container.  We therefore
        combine the stack traversal for metadata with a linear scan for
        the remaining data blocks.
        """
        start_byte = 44  # after signature + description
        n_bytes = 6

        block_id, block_size = _read_block_header(content, start_byte)
        start_byte += n_bytes
        nbp = [start_byte + block_size]

        # Search for block 122, which typically contains the metadata block 5104.
        # The algorithm uses a stack (nbp) to handle nested blocks.
        # When the second byte of the next two bytes equals 117 ('u'),
        # we pop the stack (end of current nesting level).
        while block_id != 122 and start_byte < len(content) - 2:
            next_block_id_bytes = content[start_byte : start_byte + 2]
            if len(next_block_id_bytes) < 2:
                break
            if next_block_id_bytes[1] == 117:
                # Pop stack – end of current block's children.
                start_byte = nbp[-1]
                nbp = nbp[:-1]
                if not nbp:
                    break
                while start_byte >= nbp[-1]:
                    nbp = nbp[:-1]
                    if not nbp:
                        break
                if not nbp:
                    break
            else:
                block_id, block_size = _read_block_header(content, start_byte)
                start_byte += n_bytes
                nbp.append(start_byte + block_size)

        # Decode metadata block 5104 if we stopped at block 122.
        if block_id == 122 and start_byte + block_size <= len(content):
            meta_5104 = _decode_5104(content[start_byte : start_byte + block_size])
            self.meta.update(meta_5104)

        # Scan remaining blocks for numeric metadata and spectrum data.
        # We restart from just after the outer block header and walk linearly,
        # which is sufficient for the flat blocks that follow block 122.
        self._scan_flat_blocks(content, 50)

    def _scan_flat_blocks(self, content: bytes, start: int) -> None:
        """Linear scan for known flat blocks inside the outer container."""
        offset = start
        while offset + 6 <= len(content):
            block_id, block_size = _read_block_header(content, offset)
            if block_size < 0 or offset + 6 + block_size > len(content):
                break
            data = content[offset + 6 : offset + 6 + block_size]
            if block_id in _FUNC_DECODE:
                decoded = _FUNC_DECODE[block_id](data)
                if isinstance(decoded, dict):
                    self.meta.update(decoded)
                elif isinstance(decoded, np.ndarray):
                    self.spectrum = decoded
            offset += 6 + block_size


# ------------------------------------------------------------------------------
# Public / private reader functions
# ------------------------------------------------------------------------------


def read_perkinelmer(*paths, **kwargs):
    r"""
    Open PerkinElmer `.sp` files.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s). When several datasets are returned, the
        result is a list-like `ScpObjectList`.

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
    description : `str`, optional
        A custom description.
    directory : `~pathlib.Path` object or valid urls, optional
        From where to read the files.
    merge : `bool`, optional, default: `False`
        If `True` and several filenames have been provided with compatible dimensions,
        they are merged into a single `NDDataset`.
    protocol : `str`, optional
        Protocol used for reading.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.

    Examples
    --------
    >>> ds = scp.perkinelmer.read("irdata/perkinelmer/spectra.sp")
    NDDataset: [float64] a.u. (shape: (y:1, x:3301))

    """
    kwargs["filetypes"] = ["PerkinElmer SP files (*.sp)"]
    kwargs["protocol"] = ["perkinelmer"]
    importer = Importer()
    return importer(*paths, **kwargs)


@_importer_method
def _read_sp(*args, **kwargs):
    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)
    content = fid.read()
    fid.close()

    spf = _SpFile(content)

    if spf.spectrum.size == 0:
        raise ValueError(f"No spectrum data found in PerkinElmer file {filename}")

    if spf.wavelength.size == 0:
        raise ValueError(
            f"No wavelength information found in PerkinElmer file {filename}"
        )

    if spf.spectrum.size != spf.wavelength.size:
        raise ValueError(
            f"Mismatch between spectrum size ({spf.spectrum.size}) and wavelength size ({spf.wavelength.size})"
        )

    # Ensure 2D shape (y:1, x:n_points)
    data = spf.spectrum[np.newaxis, :]

    dataset.data = data
    dataset.title = "intensity"
    dataset.units = "absorbance"
    dataset.origin = "perkinelmer"
    dataset.name = Path(filename).stem
    dataset.filename = filename
    dataset.history = f"Imported from PerkinElmer .sp file {filename}."

    # Coordinates
    x = Coord(spf.wavelength, title="wavelength", units="nm")
    y = Coord([0.0], title="spectrum index")
    dataset.set_coordset(y=y, x=x)

    # Metadata — keep only well-defined, useful fields.
    _CORE_META_KEYS = (
        "analyst",
        "date",
        "instrument_model",
        "detector",
        "source",
        "accumulations",
        "spectrum_type",
    )
    _EXTRA_META_KEYS = (
        "instrument_serial_number",
        "instrument_software_version",
        "ir_accessory",
        "image_name",
    )
    for key in _CORE_META_KEYS + _EXTRA_META_KEYS:
        value = spf.meta.get(key)
        if value not in (None, ""):
            setattr(dataset.meta, key, value)

    # Use the vendor image_name as a fallback description when none is provided.
    image_name = spf.meta.get("image_name")
    if image_name and not dataset.description:
        dataset.description = str(image_name)

    dataset.meta.readonly = True

    return dataset
