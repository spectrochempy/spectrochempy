# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import pluggy

hookspec = pluggy.HookspecMarker("spectrochempy")
hookimpl = pluggy.HookimplMarker("spectrochempy")


class SpectroChemPyHookSpec:
    @hookspec
    def get_filetype_info(self) -> dict:
        ...

    @hookspec
    def can_read(self, files: dict) -> bool:
        ...

    @hookspec
    def read_file(self, files: dict, protocol, **kwargs):
        ...

    @hookspec
    def plugin_info(self) -> dict:
        """
        Return plugin metadata.

        Expected return format::

            {
                "name": "my-plugin",
                "version": "0.1.0",
                "plugin_api_version": "1.0",
                "spectrochempy_min_version": "1.2",
                "description": "...",
                "capabilities": ["reader"],
            }

        Implementations that omit the method or return an empty dict
        are treated conservatively (no capability advertised).
        """

    # ------------------------------------------------------------------
    # Declarative contribution hooks (prototype)
    #
    # These hooks allow a plugin to declare its contributions without
    # calling ``registry`` methods directly.  Each hook should return
    # a ``list[dict]`` where every dict contains at least ``"name"``
    # and ``"func"`` keys.
    #
    # They are optional: the old imperative ``register(registry)``
    # pattern continues to work.
    # ------------------------------------------------------------------

    @hookspec
    def register_readers(self) -> list[dict]:
        """
        Return reader contributions declared by the plugin.

        Expected return format::

            [
                {
                    "name": "myformat",
                    "func": read_myformat,
                    "description": "Read MyFormat files",
                    "extensions": [".myf"],
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no reader contribution".
        """

    @hookspec
    def register_writers(self) -> list[dict]:
        """
        Return writer contributions declared by the plugin.

        Expected return format::

            [
                {
                    "name": "myformat",
                    "func": write_myformat,
                    "description": "Write MyFormat files",
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no writer contribution".
        """

    @hookspec
    def register_visualizers(self) -> list[dict]:
        """
        Return visualizer contributions declared by the plugin.

        Expected return format::

            [
                {
                    "name": "myplot",
                    "func": plot_data,
                    "description": "Plot data",
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no visualizer contribution".
        """

    @hookspec
    def register_processors(self) -> list[dict]:
        """
        Return processor contributions declared by the plugin.

        Expected return format::

            [
                {
                    "name": "smooth",
                    "func": smooth_data,
                    "description": "Smooth data",
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no processor contribution".
        """

    @hookspec
    def register_accessors(self) -> list[dict]:
        """
        Return dataset accessor contributions declared by the plugin.

        Accessor contributions are callables intended to be exposed as
        methods on SpectroChemPy objects such as ``NDDataset``.

        Expected return format::

            [
                {
                    "name": "my_accessor",
                    "func": my_accessor,
                    "description": "Run my plugin accessor",
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no accessor contribution".
        """

    # ------------------------------------------------------------------
    # Analysis hooks
    # ------------------------------------------------------------------

    @hookspec
    def register_analyses(self) -> list[dict]:
        """
        Return analysis contributions declared by the plugin.

        Analysis contributions are high-level scientific workflows
        such as decomposition, multivariate analysis, curve fitting,
        or kinetic modelling.

        Expected return format::

            [
                {
                    "name": "pca",
                    "func": perform_pca,
                    "description": "Principal Component Analysis",
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no analysis contribution".
        """

    # ------------------------------------------------------------------
    # Simulation hooks
    # ------------------------------------------------------------------

    @hookspec
    def register_simulations(self) -> list[dict]:
        """
        Return simulation contributions declared by the plugin.

        Simulation contributions wrap external computational engines
        such as thermodynamics packages, reactor simulators, or
        kinetic solvers.

        Expected return format::

            [
                {
                    "name": "equilibrium",
                    "func": compute_equilibrium,
                    "description": "Chemical equilibrium calculation",
                },
            ]

        Returning ``None`` or an empty list is treated as
        "no simulation contribution".
        """
