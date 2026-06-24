# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Result object infrastructure for analysis and fit outputs."""

__all__ = ["ResultBase", "AnalysisResult", "FitResult"]


class ResultBase:
    """
    Base class for result objects produced by fitted estimators.

    Provides common identity, provenance, named output access,
    diagnostics, and a compact text representation.
    """

    def __init__(self, *, estimator, parameters=None, outputs=None, diagnostics=None):
        self._estimator = (
            str(estimator) if not isinstance(estimator, str) else estimator
        )
        self._parameters = dict(parameters or {})
        self._outputs = dict(outputs or {})
        self._diagnostics = dict(diagnostics or {})

    # ----------------------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------------------
    @property
    def estimator(self):
        """Name of the estimator that produced this result."""
        return self._estimator

    @property
    def parameters(self):
        """Public estimator parameters used for the run."""
        return self._parameters

    @property
    def outputs(self):
        """Named output datasets produced by the operation."""
        return self._outputs

    @property
    def diagnostics(self):
        """Named diagnostic values produced by the operation."""
        return self._diagnostics

    def __getattr__(self, name):
        """
        Return a named output or diagnostic using attribute-style access.

        Normal attributes and methods are resolved before this fallback.
        Named outputs take precedence over diagnostics when both mappings use
        the same key. Parameters remain available through :attr:`parameters`.
        """
        outputs = self.__dict__.get("_outputs", {})
        if name in outputs:
            return outputs[name]

        diagnostics = self.__dict__.get("_diagnostics", {})
        if name in diagnostics:
            return diagnostics[name]

        raise AttributeError(
            f"{type(self).__name__!s} object has no attribute {name!r}"
        )

    def __dir__(self):
        """Include named outputs and diagnostics in attribute discovery."""
        names = set(super().__dir__())
        for mapping_name in ("_outputs", "_diagnostics"):
            mapping = self.__dict__.get(mapping_name, {})
            names.update(
                name
                for name in mapping
                if isinstance(name, str) and name.isidentifier()
            )
        return sorted(names)

    # ----------------------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------------------
    def __repr__(self):
        lines = [type(self).__name__]
        lines.append(f"  estimator: {self._estimator}")
        if self._parameters:
            lines.append("  parameters:")
            for name, value in self._parameters.items():
                lines.append(f"      {name}: {value}")
        if self._outputs:
            lines.append("  outputs:")
            for name, obj in self._outputs.items():
                shape = getattr(obj, "shape", None)
                if shape is not None:
                    lines.append(f"      {name} {shape}")
                else:
                    lines.append(f"      {name}")
        if self._diagnostics:
            lines.append("  diagnostics:")
            for name, obj in self._diagnostics.items():
                shape = getattr(obj, "shape", None)
                if shape is not None:
                    lines.append(f"      {name} {shape}")
                else:
                    lines.append(f"      {name}")
        return "\n".join(lines)


class AnalysisResult(ResultBase):
    """
    Result of a decomposition or projection analysis.

    Typical outputs include scores, loadings, and components.
    Typical diagnostics include explained variance.
    """

    pass


class FitResult(ResultBase):
    """
    Result of a fit or optimization operation.

    Typical outputs include fitted curves, components, and residuals.
    Typical diagnostics include convergence and goodness-of-fit metrics.
    """

    pass
