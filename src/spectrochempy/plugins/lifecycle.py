# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Plugin lifecycle model.

Defines the explicit states a plugin can be in, and a lightweight
descriptor for introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PluginState(Enum):
    """
    Lifecycle state of a plugin managed by :class:`~.PluginManager`.

    The typical transition is::

        DISCOVERED  →  LOADED  →  ACTIVE
                                           ↘  FAILED
        DISABLED  →  ACTIVE

    * **DISCOVERED** — entry point found, not yet instantiated.
    * **LOADED** — instantiated and validated, registered in pluggy.
    * **ACTIVE** — all contributions registered in the registry.
    * **FAILED** — an error occurred during load or registration.
    * **DISABLED** — explicitly deactivated by the user.
    """

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class PluginDescriptor:
    """
    Snapshot of a plugin's identity and current state.

    Returned by :meth:`~.PluginManager.get_plugin_descriptor`.
    """

    name: str
    version: str = ""
    state: PluginState = PluginState.DISCOVERED
    error: str | None = None
    entry_point: str | None = None
