# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Persistence migration utilities for legacy SCP/PSCP files."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import warnings
import zipfile

__all__ = ["migrate_legacy_file"]


def _peek_raw_document(filepath):
    """
    Read the raw JSON document from an SCP/ZIP archive without decoding arrays.

    Returns a dict with the raw payload structure intact, or None if the file
    cannot be read as a valid SCP/ZIP archive.
    """
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            member = zf.namelist()[0]
            return json.loads(zf.read(member).decode("utf-8"))
    except Exception:
        return None


def _has_legacy_payload(document):
    """
    Recursively check whether a raw SCP document contains legacy pickle payloads.

    Inspects all nested dicts for payloads with ``__class__`` keys that lack
    ``raw-base64`` encoding, indicating pickle-based persistence.
    """
    if not isinstance(document, dict):
        return False

    # Check if this dict looks like a NUMPY_ARRAY payload.
    if (
        document.get("__class__") == "NUMPY_ARRAY"
        and document.get("encoding") != "raw-base64"
    ):
        return True

    for value in document.values():
        if isinstance(value, dict) and _has_legacy_payload(value):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _has_legacy_payload(item):
                    return True

    return False


def _is_safe_document(document):
    """
    Check whether a raw SCP document uses the safe raw-base64 encoding.

    Validates document markers (``__format__``, ``__version__``) and
    recursively inspects all array payloads for ``raw-base64`` encoding.
    """
    if not isinstance(document, dict):
        return False

    # Validate document markers using the official validation.
    has_format = document.get("__format__") in ("scp", "pscp")
    has_version = document.get("__version__") is not None
    if not (has_format and has_version):
        return False

    # Check for any legacy (non-raw-base64) payloads recursively.
    if _has_legacy_payload(document):
        return False

    return True


def migrate_legacy_file(
    source,
    destination=None,
    *,
    allow_unsafe_legacy=False,
    overwrite=False,
    verbose=False,
):
    """
    Migrate a legacy SCP/PSCP file to the safe raw-base64 format.

    Legacy SCP/PSCP files produced before SpectroChemPy 0.8 may contain
    pickle-based array payloads.  Loading such files with the default
    ``allow_unsafe_legacy=False`` raises an error.  This function converts
    the file to the current safe format (``raw-base64``).

    .. warning::

        Migration **does** execute ``pickle.loads`` on the legacy payload
        to reconstruct the in-memory object.  Only migrate files whose
        provenance you fully trust.  Set ``allow_unsafe_legacy=True`` to
        acknowledge this risk explicitly.

    The migrated file is a proper SCP/ZIP archive and can be loaded with
    the default ``allow_unsafe_legacy=False``.  The source file is never
    modified or deleted.

    Writes are atomic: the destination is only replaced after the migrated
    file has been written, verified as loadable with safe defaults,
    and then moved into place via ``os.replace()``.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to the ``.scp`` or ``.pscp`` file to migrate.
    destination : str or pathlib.Path, optional
        Path for the migrated output file.  When *None*, a
        ``_migrated`` suffix is inserted before the extension in the
        source path.
    allow_unsafe_legacy : bool, default False
        **Required to be ``True``.**  Acknowledges that the source file
        will be deserialised via ``pickle.loads`` during migration.
        Set this to ``True`` only when the source file comes from a
        known and trusted origin.
    overwrite : bool, default False
        Allow overwriting an existing *destination* file.  When
        *False* and the destination already exists, a
        ``FileExistsError`` is raised.
    verbose : bool, default False
        Print a summary of the migration to stdout.

    Returns
    -------
    pathlib.Path
        Path of the migrated file.

    Raises
    ------
    FileNotFoundError
        If *source* does not exist.
    ValueError
        If *source* does not have a ``.scp`` or ``.pscp`` extension,
        if *source* and *destination* resolve to the same path, or if
        the *destination* suffix does not match the *source* suffix.
    FileExistsError
        If the destination already exists and *overwrite* is False.
    spectrochempy.utils.exceptions.SpectroChemPyError
        If *allow_unsafe_legacy* is ``False``, if the source is
        already in safe format, or if the migration fails for any
        other reason.

    See Also
    --------
    spectrochempy.load : Load an SCP/PSCP file.
    spectrochempy.NDDataset.save : Save an NDDataset to SCP format.
    spectrochempy.Project.save : Save a Project to PSCP format.

    Examples
    --------
    >>> from spectrochempy import migrate_legacy_file
    >>> # migrate a single file (requires trust acknowledgement)
    >>> migrated = migrate_legacy_file("old_data.scp",
    ...                                allow_unsafe_legacy=True)
    >>> # migrate to a specific path
    >>> migrated = migrate_legacy_file("old_data.scp", "new_data.scp",
    ...                                allow_unsafe_legacy=True,
    ...                                overwrite=True)
    """
    from spectrochempy.core.dataset.arraymixins.ndio import load as scp_load
    from spectrochempy.utils.exceptions import SpectroChemPyError

    source = pathlib.Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in (".scp", ".pscp"):
        raise ValueError(
            f"Source file must have a .scp or .pscp extension, got: {suffix}"
        )

    if destination is None:
        destination = source.parent / f"{source.stem}_migrated{source.suffix}"
    else:
        destination = pathlib.Path(destination)

    # Reject source == destination (resolved).
    if source.resolve() == destination.resolve():
        raise ValueError(
            "Source and destination resolve to the same path. "
            "Migration would overwrite the original file."
        )

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination file already exists: {destination}. "
            "Use overwrite=True to replace it."
        )

    dest_suffix = destination.suffix.lower()
    if dest_suffix not in (".scp", ".pscp"):
        raise ValueError(
            f"Destination must have a .scp or .pscp extension, got: {dest_suffix}"
        )
    if suffix != dest_suffix:
        raise ValueError(
            f"Destination suffix {dest_suffix} does not match source suffix "
            f"{suffix}.  Cross-format migration is not supported."
        )

    # Peek at the raw JSON to detect encoding without executing pickle.
    raw = _peek_raw_document(source)
    if _is_safe_document(raw):
        raise SpectroChemPyError(
            f"{source.name} is already in safe format. No migration needed."
        )

    # Explicit trust acknowledgement required.
    if not allow_unsafe_legacy:
        raise SpectroChemPyError(
            "Migrating this file requires deserialising pickle payloads. "
            "Set allow_unsafe_legacy=True only if the source file comes "
            "from a known and trusted source."
        )

    warnings.warn(
        f"Deserialising legacy pickle payload in {source.name}. "
        "Only do this for files from trusted sources.",
        stacklevel=2,
    )

    try:
        obj = scp_load(source, allow_unsafe_legacy=True)

        # Write to a temporary file in the same directory, then atomically
        # replace the destination.  This preserves the existing destination
        # on failure.
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=f"-{destination.name}",
            dir=destination.parent,
        )
        os.close(tmp_fd)
        tmp_path = pathlib.Path(tmp_path)
        try:
            obj.dump(tmp_path)

            # Verify the migrated file is loadable with safe defaults.
            verification = scp_load(tmp_path, allow_unsafe_legacy=False)
            if verification is None:
                raise SpectroChemPyError(
                    "Verification failed: migrated file could not be loaded "
                    "with safe defaults."
                )

            # Atomic replace: destination is only touched after success.
            os.replace(tmp_path, destination)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        if verbose:
            print(
                f"Migrated {source.name} -> {destination.name} "
                f"(safe raw-base64 format)"
            )
    except Exception:
        # Clean up partial output on failure (only if it was created
        # outside the tmp block, which should not happen, but defensive).
        raise

    return destination
