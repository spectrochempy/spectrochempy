# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Shared dataset-level metadata handling for NMR readers."""


def apply_reader_metadata_options(dataset, kwargs, *, default_origin):
    """Apply documented user overrides without changing vendor metadata."""
    origin = kwargs.get("origin")
    dataset.origin = default_origin if origin is None else origin

    description = kwargs.get("description")
    if description is not None:
        dataset.description = description


__all__ = ["apply_reader_metadata_options"]
