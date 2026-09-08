# ======================================================================================
# Copyright (©) 2015-2026 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a framework
# for processing, analysing and modelling *Spectro*scopic
# data for *Chem*istry with *Py*thon (SpectroChemPy).
#
# This software is governed by the CeCILL-B license under French law.
# ======================================================================================

"""
Reserved public root symbols of ``spectrochempy``.

This module computes, from existing sources of truth, the set of names that
are public at the ``scp`` root.  These names are protected against plugin
collisions in two ways:

- the plugin manager rejects a plugin I/O namespace whose name collides with
  one of these symbols (see ``register_io_namespace``);
- the root ``__getattr__`` always resolves these names to the core symbol,
  regardless of plugin installation or discovery order.

Other plugin-provided root surfaces (``root_exports``, reader exports,
``analysis``/``simulation`` extensions) are not rejected at registration
time; they are protected only by the resolution priority above.

Sources of truth used:

- ``_LAZY_IMPORTS`` (``spectrochempy.lazyimport.api_methods``) — public
  functions, classes, constants and compatibility aliases exposed at root;
- ``_LAZY_DATASETS_IMPORTS``
  (``spectrochempy.lazyimport.dataset_methods``) — ``NDDataset`` methods
  exposed at root;
- ``_PLOT_PROFILE_FUNCTIONS``
  (``spectrochempy.lazyimport.plot_profile``) — plotting profile API;
- ``_CORE_IO_NAMESPACES`` (``spectrochempy.core.io_namespaces``) — core I/O
  namespace names;
- the public submodules declared in ``spectrochempy/__init__.pyi`` (the same
  source that ``lazy_loader.attach_stub`` parses at import time).
"""

from __future__ import annotations

import functools
from pathlib import Path

from spectrochempy.lazyimport.api_methods import _LAZY_IMPORTS
from spectrochempy.lazyimport.dataset_methods import _LAZY_DATASETS_IMPORTS
from spectrochempy.lazyimport.plot_profile import _PLOT_PROFILE_FUNCTIONS

_ROOT_STUB = Path(__file__).resolve().parent.parent / "__init__.pyi"


def _submodule_names() -> frozenset[str]:
    """
    Return the public submodule names declared in the root stub.

    The stub is the same source that ``lazy_loader.attach_stub`` uses at
    import time, so the result never diverges from the runtime module list.
    """
    import ast

    if _ROOT_STUB.exists():
        try:
            tree = ast.parse(_ROOT_STUB.read_text(encoding="utf-8"))
        except SyntaxError:
            return frozenset()
        names: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 1
                and node.module is None
            ):
                names.update(alias.name for alias in node.names)
        return frozenset(names)
    return frozenset()


def _core_io_namespaces() -> frozenset[str]:
    from spectrochempy.core.io_namespaces import _CORE_IO_NAMESPACES

    return frozenset(_CORE_IO_NAMESPACES)


@functools.lru_cache(maxsize=1)
def reserved_root_symbols() -> frozenset[str]:
    """
    Return the set of names that are public ``scp`` symbols.

    The result is cached; it depends only on the core package layout, which is
    static after import.  Core I/O namespaces are imported lazily to avoid an
    import cycle at startup.
    """
    return frozenset(
        set(_LAZY_IMPORTS)
        | set(_LAZY_DATASETS_IMPORTS)
        | set(_PLOT_PROFILE_FUNCTIONS)
        | _core_io_namespaces()
        | _submodule_names()
    )


def is_reserved_root_symbol(name: str) -> bool:
    """Return ``True`` when *name* is a public ``scp`` root symbol."""
    return name in reserved_root_symbols()
