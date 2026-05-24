# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from collections.abc import Callable

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

    @hookspec
    def register_unit_contexts(self) -> list[dict]:
        """
        Return Pint unit-context contributions declared by the plugin.

        Expected return format::

            [
                {
                    "name": "nmr",
                    "func": set_nmr_context,
                    "predicate": applies_to_coord,
                    "argument_extractor": get_larmor,
                    "description": "NMR ppm/frequency conversion context",
                },
            ]

        ``predicate`` is optional.  When provided, the core calls it with the
        object being converted and uses the context only when it returns
        ``True``.  ``argument_extractor`` is also optional; when provided, it
        receives the same object and returns either positional arguments,
        keyword arguments, or ``(args, kwargs)`` for the setup function.

        Returning ``None`` or an empty list is treated as
        "no unit-context contribution".
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

    # ------------------------------------------------------------------
    # Generic handler overrides
    #
    # A single hook that lets plugins override named behaviour in core
    # classes.  Keys follow the convention ``"<domain>.<action>"``.
    #
    # Unlike the list-based register_* hooks above, this one returns a
    # single dict whose values are callables.  Each callable receives
    # arguments specific to its extension point and returns either a
    # result or ``None`` (meaning "use the core default").
    # ------------------------------------------------------------------

    @hookspec
    def register_handlers(self) -> dict[str, Callable]:
        """
        Return a dict of named handler overrides for core extension points.

        Example::

            def register_handlers(self):
                return {
                    "coord.reversed": _my_coord_reversed,
                    "concatenate.extract_metadata": _my_extract_metadata,
                    "concatenate.postprocess": _my_concat_postprocess,
                    "ndmath.execution_branch": _my_ndmath_branch,
                    "ndmath.execute": _my_ndmath_execute,
                }

        Recognised handler names
        ------------------------
        ``"coord.reversed"``
            ``callable(self: Coord) -> bool | None``
            Return True/False for whether the coordinate axis should be
            displayed in decreasing order, or ``None`` to use the core default.
        ``"concatenate.extract_metadata"``
            ``callable(datasets: list) -> dict | None``
            Extract metadata coordinates (e.g. TopSpin parameters) from
            a list of datasets before concatenation.  Return a dict of
            ``{name: [values, ...]}`` or ``None``.
        ``"concatenate.postprocess"``
            ``callable(out: NDDataset, datasets: list, metacoords: dict) -> NDDataset | None``
            Post-process the result of a concatenation.  Return the
            modified dataset or ``None``.
        ``"ndmath.execution_branch"``
            ``callable(fname: str, data: np.ndarray, args: list) -> str | None``
            Return the execution branch name (``"real"``, ``"quaternion"``, …)
            for the current math operation, or ``None`` to use the core default.
            This allows a plugin to define non-real numeric execution paths
            (e.g. quaternion decomposition) without the core knowing about
            those types.
        ``"ndmath.execute"``
            ``callable(branch: str, f: Callable, d: np.ndarray, args: list) -> np.ndarray | None``
            Execute the math operation *f* on data *d* for the given
            *branch*.  Should return the result array or ``None`` to fall
            back to the core default implementation.
        ``"ndmath.numpy_method.<name>"``
            ``callable(dataset: NDDataset, *args, **kwargs) -> NDDataset | None``
            Override the ``@_from_numpy_method`` decorated method *name*
            (e.g. ``"absolute"``, ``"amax"``, ``"conjugate"``) when the
            data requires plugin-specific handling.  Return the modified
            dataset or ``None`` to fall back to the standard numpy path.
        ``"importer.resolve_directory_target"``
            ``callable(path: Path, **kwargs) -> Path | list[Path] | None``
            Resolve a directory passed to ``read`` into one or more concrete
            files for a plugin-owned format.  Return ``None`` to use the core
            directory globbing behavior.
        ``"importer.infer_filetype_key"``
            ``callable(path: Path, **kwargs) -> str | None``
            Return a filetype key such as ``".myformat"`` for extensionless
            files owned by a plugin format.  Return ``None`` when the plugin
            does not recognize the path.
        ``"importer.remote_download_target"``
            ``callable(path: Path, **kwargs) -> Path | None``
            Return the path that should be downloaded for formats whose data
            lives in a surrounding directory.  Return ``None`` to download the
            requested path directly.

        Returning ``None`` or an empty dict is treated as
        "no handler override".
        """
