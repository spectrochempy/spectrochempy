# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Minimal linear Pipeline for SpectroChemPy estimators."""

from __future__ import annotations

from types import MappingProxyType

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils._estimator import parameter_values_equal
from spectrochempy.utils._estimator import pipeline_v1_step_kind
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.exceptions import SpectroChemPyError

__all__ = ["Pipeline"]
__configurables__ = ["Pipeline"]

_RESERVED_STEP_NAMES = frozenset(
    {
        "steps",
        "named_steps",
        "fitted_steps_",
        "fitted_named_steps_",
        "fit",
        "fit_transform",
        "transform",
        "predict",
        "score",
        "get_params",
        "set_params",
    }
)


class Pipeline:
    """
    Linear composition of allowlisted SpectroChemPy preprocessing and estimators.

    The supplied steps are templates. Calling :meth:`fit` clones those
    templates, fits the clones, and exposes fitted runtime state through
    :attr:`fitted_steps_` and :attr:`fitted_named_steps_`.
    """

    def __init__(self, steps):
        self._steps = self._validate_steps(steps)
        self._fitted = False
        self._fitted_steps = None

    @property
    def steps(self):
        """Return ordered template ``(name, step)`` pairs."""
        return self._steps

    @property
    def named_steps(self):
        """Return a read-only mapping of template step names to steps."""
        return MappingProxyType(dict(self._steps))

    @property
    def fitted_steps_(self):
        """Return ordered fitted ``(name, step)`` pairs."""
        if not self._fitted or self._fitted_steps is None:
            raise NotFittedError(attr="fitted_steps_")
        return self._fitted_steps

    @property
    def fitted_named_steps_(self):
        """Return a read-only mapping of fitted step names to fitted steps."""
        if not self._fitted or self._fitted_steps is None:
            raise NotFittedError(attr="fitted_named_steps_")
        return MappingProxyType(dict(self._fitted_steps))

    def fit(self, X, y=None):
        """
        Fit the pipeline on *X* and optional supervised target *y*.

        Intermediate preprocessing transformers receive only the current
        dataset. A supervised final estimator receives the transformed dataset
        and *y*.
        """
        self._clear_fitted_state()
        final_name, final_step = self._steps[-1]
        final_kind = self._step_kind(final_step, final=True)
        if y is not None and final_kind == "transformer":
            raise SpectroChemPyError(
                "Pipeline.fit received y, but final step "
                f"'{final_name}' is a transformer and does not accept y."
            )
        if y is None and final_kind == "estimator":
            raise SpectroChemPyError(
                "Pipeline.fit requires y because final step "
                f"'{final_name}' is a supervised estimator."
            )

        fitted_steps = []
        current = X
        try:
            for position, (name, template) in enumerate(self._steps):
                step = self._clone_step(template, name, position)
                is_final = position == len(self._steps) - 1
                if is_final:
                    if final_kind == "estimator":
                        self._fit_step(step, current, y, name, position)
                    else:
                        self._fit_step(step, current, None, name, position)
                    fitted_steps.append((name, step))
                    break

                self._fit_step(step, current, None, name, position)
                current = self._transform_step(step, current, name, position)
                fitted_steps.append((name, step))
        except Exception:
            self._clear_fitted_state()
            raise

        self._fitted_steps = tuple(fitted_steps)
        self._fitted = True
        return self

    def fit_transform(self, X, y=None):
        """Fit the pipeline and transform *X* with the fitted pipeline."""
        final_name, final_step = self._steps[-1]
        if self._step_kind(final_step, final=True) != "transformer":
            raise SpectroChemPyError(
                "Pipeline.fit_transform is not available because final step "
                f"'{final_name}' is a supervised estimator."
            )
        return self.fit(X, y).transform(X)

    def transform(self, X):
        """Transform *X* with a transformer-final fitted pipeline."""
        self._require_fitted("transform")
        final_name, final_step = self._steps[-1]
        if self._step_kind(final_step, final=True) != "transformer":
            raise SpectroChemPyError(
                "Pipeline.transform is not available because final step "
                f"'{final_name}' is a supervised estimator."
            )
        current = X
        for position, (name, step) in enumerate(self.fitted_steps_):
            current = self._transform_step(step, current, name, position)
        return current

    def predict(self, X):
        """Predict from *X* with an estimator-final fitted pipeline."""
        self._require_fitted("predict")
        final_name, final_step = self._steps[-1]
        if self._step_kind(final_step, final=True) != "estimator":
            raise SpectroChemPyError(
                "Pipeline.predict is not available because final step "
                f"'{final_name}' is a transformer."
            )
        current = self._transform_intermediates(X, operation="predict")
        fitted_final = self.fitted_steps_[-1][1]
        try:
            return fitted_final.predict(current)
        except Exception as exc:
            raise self._step_error(
                "predict", final_name, len(self._steps) - 1, fitted_final, exc
            ) from exc

    def score(self, X, y=None):
        """Score predictions from *X* against *y* with a fitted final estimator."""
        self._require_fitted("score")
        final_name, final_step = self._steps[-1]
        if self._step_kind(final_step, final=True) != "estimator":
            raise SpectroChemPyError(
                "Pipeline.score is not available because final step "
                f"'{final_name}' is a transformer."
            )
        fitted_final = self.fitted_steps_[-1][1]
        if not hasattr(fitted_final, "score"):
            raise SpectroChemPyError(
                f"Final step '{final_name}' does not provide score()."
            )
        if y is None:
            raise SpectroChemPyError(
                f"Pipeline.score requires y for final step '{final_name}'."
            )
        current = self._transform_intermediates(X, operation="score")
        try:
            return fitted_final.score(current, y)
        except Exception as exc:
            raise self._step_error(
                "score", final_name, len(self._steps) - 1, fitted_final, exc
            ) from exc

    def get_params(self, deep=True):
        """
        Return template configuration parameters.

        With ``deep=True``, each template is exposed under its step name and
        its constructor parameters under ``step__parameter`` keys.
        """
        params = {"steps": self.steps}
        if not deep:
            return params
        for name, step in self._steps:
            params[name] = step
            if hasattr(step, "get_params"):
                for key, value in step.get_params(deep=False).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        """Transactionally update template steps or nested template parameters."""
        if not params:
            return self

        candidate_steps = (
            self._validate_steps(params["steps"]) if "steps" in params else self._steps
        )
        candidate_by_name = dict(candidate_steps)
        step_replacements = {}
        nested_updates = {}

        valid_top_level = {"steps", *(name for name, _ in self._steps)}
        for key, value in params.items():
            if key == "steps":
                continue
            if "__" in key:
                step_name, parameter = key.split("__", 1)
                if not step_name or not parameter:
                    raise SpectroChemPyError(
                        f"Invalid nested parameter '{key}' for Pipeline."
                    )
                if step_name not in candidate_by_name:
                    valid = ", ".join(sorted(candidate_by_name))
                    raise SpectroChemPyError(
                        f"Invalid step name '{step_name}' for Pipeline. "
                        f"Valid step names: {valid}."
                    )
                nested_updates.setdefault(step_name, {})[parameter] = value
            elif key in candidate_by_name:
                step_replacements[key] = value
            else:
                valid = ", ".join(sorted(valid_top_level))
                raise SpectroChemPyError(
                    f"Invalid parameter '{key}' for Pipeline. Valid parameters: {valid}."
                )

        if step_replacements:
            replaced = []
            for name, step in candidate_steps:
                replaced.append((name, step_replacements.get(name, step)))
            candidate_steps = self._validate_steps(replaced)

        candidate_steps = self._apply_nested_updates(candidate_steps, nested_updates)
        changed = not self._steps_identical(self._steps, candidate_steps)
        if changed:
            self._steps = candidate_steps
            self._clear_fitted_state()
        return self

    def _apply_nested_updates(self, candidate_steps, nested_updates):
        if not nested_updates:
            return candidate_steps
        updated = []
        for name, step in candidate_steps:
            updates = nested_updates.get(name)
            if updates:
                original_step = step
                try:
                    step = clone_unfitted(step)
                    step.set_params(**updates)
                except Exception as exc:
                    raise SpectroChemPyError(
                        f"Invalid nested parameter update for Pipeline step '{name}'."
                    ) from exc
                if self._step_equal(original_step, step):
                    step = original_step
            updated.append((name, step))
        return self._validate_steps(updated)

    def _clear_fitted_state(self):
        self._fitted = False
        self._fitted_steps = None

    def _require_fitted(self, attr):
        if not self._fitted or self._fitted_steps is None:
            raise NotFittedError(attr=attr)

    def _transform_intermediates(self, X, *, operation):
        current = X
        for position, (name, step) in enumerate(self.fitted_steps_[:-1]):
            current = self._transform_step(step, current, name, position, operation)
        return current

    def _clone_step(self, template, name, position):
        try:
            return clone_unfitted(template)
        except Exception as exc:
            raise self._step_error("clone", name, position, template, exc) from exc

    def _fit_step(self, step, X, y, name, position):
        try:
            if y is None:
                step.fit(X)
            else:
                step.fit(X, y)
        except Exception as exc:
            raise self._step_error("fit", name, position, step, exc) from exc

    def _transform_step(self, step, X, name, position, operation="transform"):
        try:
            result = step.transform(X)
        except Exception as exc:
            raise self._step_error(operation, name, position, step, exc) from exc
        if not isinstance(result, NDDataset):
            raise SpectroChemPyError(
                f"Pipeline {operation} step '{name}' at position {position} "
                f"returned {result.__class__.__name__}, expected NDDataset."
            )
        return result

    @staticmethod
    def _step_error(operation, name, position, step, exc):
        return SpectroChemPyError(
            f"Pipeline {operation} failed at step '{name}' "
            f"(position {position}, class {step.__class__.__name__}): {exc}"
        )

    @classmethod
    def _validate_steps(cls, steps):
        if not isinstance(steps, list | tuple):
            raise SpectroChemPyError("Pipeline steps must be an ordered list or tuple.")
        if not steps:
            raise SpectroChemPyError("Pipeline requires at least one step.")

        validated = []
        seen = set()
        for position, item in enumerate(steps):
            if not isinstance(item, tuple) or len(item) != 2:
                raise SpectroChemPyError(
                    "Pipeline steps must be ordered (name, step) pairs."
                )
            name, step = item
            cls._validate_step_name(name, seen)
            cls._validate_step_object(step, name, position, position == len(steps) - 1)
            validated.append((name, step))
            seen.add(name)
        return tuple(validated)

    @classmethod
    def _validate_step_name(cls, name, seen):
        if not isinstance(name, str):
            raise SpectroChemPyError("Pipeline step names must be strings.")
        if not name:
            raise SpectroChemPyError("Pipeline step names must be non-empty strings.")
        if "__" in name:
            raise SpectroChemPyError(
                f"Pipeline step name '{name}' cannot contain '__'."
            )
        if name in seen:
            raise SpectroChemPyError(f"Pipeline step name '{name}' is duplicated.")
        if name in _RESERVED_STEP_NAMES:
            raise SpectroChemPyError(
                f"Pipeline step name '{name}' is reserved by Pipeline."
            )

    @classmethod
    def _validate_step_object(cls, step, name, position, final):
        if step is None:
            raise SpectroChemPyError(
                f"Pipeline step '{name}' at position {position} cannot be None."
            )
        if isinstance(step, str):
            raise SpectroChemPyError(
                f"Pipeline step '{name}' at position {position} cannot be a string."
            )

        kind = cls._step_kind(step, final=final)
        if not final and kind != "intermediate":
            raise SpectroChemPyError(
                f"Pipeline step '{name}' at position {position} with class "
                f"{step.__class__.__name__} is not an allowlisted intermediate "
                "transformer."
            )
        if final and kind == "unsupported":
            raise SpectroChemPyError(
                f"Pipeline final step '{name}' at position {position} with class "
                f"{step.__class__.__name__} is not in the v1 allowlist."
            )

    @staticmethod
    def _step_kind(step, *, final):
        return pipeline_v1_step_kind(step, final=final)

    @staticmethod
    def _steps_identical(left, right):
        if len(left) != len(right):
            return False
        for (left_name, left_step), (right_name, right_step) in zip(
            left, right, strict=True
        ):
            if left_name != right_name or left_step is not right_step:
                return False
        return True

    @staticmethod
    def _step_equal(left_step, right_step):
        if left_step is right_step:
            return True
        if type(left_step) is not type(right_step):
            return False
        left_params = left_step.get_params(deep=False)
        right_params = right_step.get_params(deep=False)
        if left_params.keys() != right_params.keys():
            return False
        for key, left_value in left_params.items():
            if not parameter_values_equal(left_value, right_params[key]):
                return False
        return True

    def __repr__(self):
        inner = ", ".join(f"({name!r}, {step!r})" for name, step in self._steps)
        return f"Pipeline([{inner}])"
