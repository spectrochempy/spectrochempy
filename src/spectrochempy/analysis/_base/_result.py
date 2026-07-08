# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Result object infrastructure for analysis and fit outputs."""

import numpy as np
from scipy import stats

from spectrochempy.utils.print import DisplayItem
from spectrochempy.utils.print import DisplaySection
from spectrochempy.utils.print import _html_heading
from spectrochempy.utils.print import _render_sections

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
    # HTML representation (Jupyter notebooks)
    # ----------------------------------------------------------------------------------
    def _repr_sections(self):
        sections: list[DisplaySection] = []

        # --- Summary : estimator ---
        sections.append(
            DisplaySection(
                "summary", "", [DisplayItem("field", self._estimator, "estimator")]
            )
        )

        # --- Parameters ---
        if self._parameters:
            params = [
                DisplayItem("field", str(v), k) for k, v in self._parameters.items()
            ]
            sections.append(
                DisplaySection("data", f"Parameters ({len(params)})", params)
            )

        # --- Outputs ---
        if self._outputs:
            outputs = []
            for name, obj in self._outputs.items():
                if hasattr(obj, "_repr_html_"):
                    outputs.append(DisplayItem("html", obj._repr_html_(), name))
                else:
                    shape = getattr(obj, "shape", None)
                    label = str(shape) if shape is not None else type(obj).__name__
                    outputs.append(DisplayItem("field", label, name))
            sections.append(
                DisplaySection("data", f"Outputs ({len(outputs)})", outputs)
            )

        # --- Diagnostics ---
        if self._diagnostics:
            diags = []
            for name, obj in self._diagnostics.items():
                if hasattr(obj, "_repr_html_"):
                    diags.append(DisplayItem("html", obj._repr_html_(), name))
                else:
                    diags.append(DisplayItem("field", str(obj), name))
            sections.append(
                DisplaySection("data", f"Diagnostics ({len(diags)})", diags)
            )

        return sections

    def _repr_html_(self):
        sections = self._repr_sections()
        body = _render_sections(sections)
        heading = _html_heading(self)
        return (
            '<div class="scp-output">'
            f"<details open><summary>{heading}</summary>\n{body}\n"
            "</details></div>"
        )

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

    def __init__(
        self,
        *,
        estimator,
        parameters=None,
        outputs=None,
        diagnostics=None,
        covariance=None,
        variance=None,
        stderr=None,
        correlation=None,
        parameter_values=None,
        confidence_intervals=None,
        confidence_level=0.95,
    ):
        super().__init__(
            estimator=estimator,
            parameters=parameters,
            outputs=outputs,
            diagnostics=diagnostics,
        )
        self._covariance = covariance
        self._variance = variance
        self._stderr = stderr
        self._correlation = correlation
        self._parameter_values = parameter_values
        self._confidence_intervals = confidence_intervals
        self._confidence_level = confidence_level

    @property
    def covariance(self):
        """
        Approximate covariance matrix of the fitted varying parameters.

        Returns
        -------
        ndarray or None
            Approximate covariance matrix computed from the retained least-squares
            Jacobian when available and mathematically meaningful. Returns ``None``
            when covariance is unavailable.
        """
        return self._covariance

    @property
    def variance(self):
        """
        Approximate per-parameter variances derived from the covariance matrix.

        Returns
        -------
        ndarray or None
            Immutable 1D array containing the diagonal of
            :attr:`covariance`. Returns ``None`` when covariance is unavailable.
        """
        if self._variance is None and self._covariance is not None:
            variance = np.array(np.diag(self._covariance), copy=True, dtype=np.float64)
            tiny_negative = (variance < 0.0) & np.isclose(variance, 0.0)
            variance[tiny_negative] = 0.0
            variance.flags.writeable = False
            self._variance = variance
        return self._variance

    @property
    def stderr(self):
        """
        Approximate per-parameter standard errors derived from covariance.

        Returns
        -------
        ndarray or None
            Immutable 1D array containing the square root of
            :attr:`variance` when available. Returns ``None`` when covariance is
            unavailable.
        """
        variance = self.variance
        if self._stderr is None and variance is not None:
            stderr = np.full_like(variance, np.nan, dtype=np.float64)
            valid = np.isfinite(variance) & (variance >= 0.0)
            stderr[valid] = np.sqrt(variance[valid])
            stderr.flags.writeable = False
            self._stderr = stderr
        return self._stderr

    @property
    def correlation(self):
        """
        Approximate parameter correlation matrix derived from covariance.

        Returns
        -------
        ndarray or None
            Immutable correlation matrix computed from :attr:`covariance` and
            :attr:`stderr`. Returns ``None`` when covariance is unavailable.
            Entries whose normalization is undefined remain ``nan``.
        """
        covariance = self.covariance
        stderr = self.stderr
        if self._correlation is None and covariance is not None and stderr is not None:
            denom = np.outer(stderr, stderr)
            correlation = np.full_like(covariance, np.nan, dtype=np.float64)
            valid = np.isfinite(covariance) & np.isfinite(denom) & (denom > 0.0)
            correlation[valid] = covariance[valid] / denom[valid]

            zero_std = np.isfinite(stderr) & np.isclose(stderr, 0.0)
            for index, is_zero in enumerate(zero_std):
                if is_zero:
                    correlation[index, index] = 1.0

            nonzero = np.isfinite(stderr) & (stderr > 0.0)
            diag_idx = np.where(nonzero)[0]
            correlation[diag_idx, diag_idx] = 1.0

            correlation = 0.5 * (correlation + correlation.T)
            correlation.flags.writeable = False
            self._correlation = correlation
        return self._correlation

    @property
    def confidence_level(self):
        """Confidence level used for :attr:`confidence_intervals`."""
        return self._confidence_level

    @property
    def confidence_intervals(self):
        """
        Approximate two-sided confidence intervals for varying parameters.

        Returns
        -------
        ndarray or None
            Immutable array of shape ``(n_parameters, 2)`` containing lower and
            upper bounds for approximate 95% confidence intervals derived from
            the fitted parameter values, standard errors, and Student-t critical
            value based on the residual degrees of freedom. Returns ``None``
            when the required uncertainty information is unavailable.
        """
        stderr = self.stderr
        values = self._parameter_values
        degrees_of_freedom = int(self.diagnostics.get("degrees_of_freedom", 0))

        if (
            self._confidence_intervals is None
            and stderr is not None
            and values is not None
            and degrees_of_freedom > 0
        ):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != stderr.shape:
                return None

            alpha = 1.0 - float(self._confidence_level)
            t_value = stats.t.ppf(1.0 - (alpha / 2.0), degrees_of_freedom)
            if not np.isfinite(t_value):
                return None

            delta = t_value * stderr
            intervals = np.column_stack((values - delta, values + delta))
            intervals.flags.writeable = False
            self._confidence_intervals = intervals
        return self._confidence_intervals
