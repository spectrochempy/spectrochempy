# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module implementing the base abstract classes to define estimators such as PCA, ..."""

import copy
import logging
import warnings
from types import SimpleNamespace

import numpy as np
import traitlets as tr
from sklearn import linear_model

from spectrochempy.application.application import app
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.constants import NOMASK
from spectrochempy.utils.decorators import _wrap_ndarray_output_to_nddataset
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.meta import Meta
from spectrochempy.utils.traits import NDDatasetType


# ======================================================================================
# Base class AnalysisConfigurable
# ======================================================================================
class AnalysisSourceMetadata:
    """
    Internal snapshot of the metadata of a scientific source dataset.

    It is captured from the raw ``X`` / ``Y`` input *before* any
    ``NDDatasetType`` / ``NDDataset(value)`` coercion or internal preprocessing,
    so that model outputs can reuse the exact source metadata (e.g. ``author``)
    instead of the values recreated by the coercion step.

    This is deliberately not a second dataset: it only holds a disconnected
    copy of the metadata fields, independent of the source object (no shared
    mutable reference to ``meta``, ``coordset``, ``history`` or ``mask``).

    Only ``author`` is currently consumed by the output wrapping decorator.
    The other fields are preserved for the deterministic identity/history
    alignment (see the accepted analysis output metadata policy, PR 2).
    """

    def __init__(self, source: NDDataset):
        self.name = source.name
        self.title = source.title
        self.description = source.description
        self.author = source.author
        self.origin = source.origin
        self.filename = source.filename
        self.meta = copy.deepcopy(source.meta)
        self.units = source.units
        self.created = source._created
        self.modified = source._modified
        self.acquisition_date = source._acquisition_date
        self.history = list(source.history or [])
        self.shape = tuple(source.shape)
        self.dims = tuple(source.dims)
        self.coordset = source.coordset.copy() if source.coordset is not None else None
        self.mask = source.mask.copy() if source.mask is not None else None


# ======================================================================================
# Base class AnalysisConfigurable
# ======================================================================================
class AnalysisConfigurable(BaseConfigurable):
    """
    Abstract class to write analysis model estimators.

    Analysis model class must subclass this to get a minimal structure.

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    warm_start : `bool`, optional, default: `False`
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        reuse the solution of the previous call to fit and add more components
        (if available) in a sequential manner.

        When `warm_start` is `True`, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to `fit`.

    """

    # Get doc sections for reuse in subclass

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _fitted = tr.Bool(False, help="False if the model was not yet fitted")
    _outfit = tr.Any(help="the output of the _fit method - generally a tuple")

    _X_source_metadata = tr.Instance(
        AnalysisSourceMetadata,
        allow_none=True,
        help="Snapshot of the metadata of the scientific source assigned to _X",
    )
    _Y_source_metadata = tr.Instance(
        AnalysisSourceMetadata,
        allow_none=True,
        help="Snapshot of the metadata of the scientific source assigned to _Y",
    )

    _ANALYSIS_ROLE_TITLES = {
        "scores": "scores",
        "y_scores": "y scores",
        "components": "components",
        "loadings": "loadings",
        "weights": "weights",
        "rotations": "rotations",
        "concentration_profiles": "concentration profiles",
        "spectral_profiles": "spectral profiles",
        "reconstruction": "reconstruction",
        "fitted_data": "fitted data",
        "prediction": "prediction",
        "residuals": "residuals",
        "diagnostic": "diagnostic",
        "singular_values": "singular values",
        "explained_variance": "explained variance",
        "explained_variance_ratio": "explained variance ratio",
        "cumulative_explained_variance": "cumulative explained variance",
    }

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------

    # Write here traits like e.g.,
    #     A = Unicode("A", help='description").tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level=logging.WARNING,
        warm_start=False,
        **kwargs,
    ):
        self._warm_start = warm_start

        super().__init__(log_level=log_level, **kwargs)

        if not warm_start:
            # We should not be able to use any methods requiring fit results
            # until the fit method has been executed
            self._fitted = False

    # ----------------------------------------------------------------------------------
    # Private metadata snapshot helpers
    # ----------------------------------------------------------------------------------
    def _capture_source_metadata(self, role, value, force=False):
        """
        Capture (or clear) the metadata snapshot of a scientific source input.

        This helper must be called with the raw input value *before* any
        ``NDDatasetType`` / ``NDDataset(value)`` coercion or internal
        preprocessing, at each ``_X`` / ``_Y`` assignment site of a ``fit``.

        Persistent snapshots are only replaced or cleared by ``fit``.
        Argument-less methods reuse them, and direct-call methods never touch
        them: a direct ``NDDataset`` argument is used as temporary authority
        for that call only.

        Parameters
        ----------
        role : `str`
            ``"_X"`` or ``"_Y"``.
        value : `NDDataset`, array-like or `None`
            Raw value about to be assigned to the ``_X`` / ``_Y`` trait.
        force : `bool`, optional
            ``fit`` semantics: ``True`` clears the snapshot when `value` is
            ``None`` instead of keeping it.

        Notes
        -----
        - an ``NDDataset`` source replaces the current snapshot;
        - an array-like source clears the snapshot (no scientific source
          metadata to preserve);
        - ``None`` keeps the current snapshot unless ``force`` is ``True``.
        """
        if value is None and not force:
            return
        if isinstance(value, NDDataset):
            snapshot = AnalysisSourceMetadata(value)
        else:
            snapshot = None
        setattr(self, f"{role}_source_metadata", snapshot)

    def _normalize_analysis_role(self, role_id):
        if role_id == "reconstruction" and type(self).__name__ == "Optimize":
            return "fitted_data"
        return role_id

    def _analysis_role_title(self, role_id):
        return self._ANALYSIS_ROLE_TITLES[role_id]

    @staticmethod
    def _analysis_ordered_unique_text(values):
        ordered = []
        seen = set()
        for value in values:
            if value in (None, ""):
                continue
            if value in seen:
                continue
            ordered.append(value)
            seen.add(value)
        if not ordered:
            return None
        if len(ordered) == 1:
            return ordered[0]
        return " & ".join(ordered)

    @staticmethod
    def _analysis_source_label(source_metadata):
        if source_metadata is None or source_metadata.name in (None, ""):
            return "<unnamed>"
        return source_metadata.name

    @staticmethod
    def _analysis_source_token(source_metadata):
        if source_metadata is None or source_metadata.name in (None, ""):
            return None
        return source_metadata.name

    def _analysis_source_list(self, sources):
        items = [
            self._analysis_source_label(source)
            for source in sources
            if source is not None
        ]
        if not items:
            return "<unnamed>"
        return " + ".join(items)

    @staticmethod
    def _analysis_acquisition_date_consensus(sources):
        sources = [source for source in sources if source is not None]
        if not sources:
            return None
        if len(sources) == 1:
            return sources[0].acquisition_date
        values = [source.acquisition_date for source in sources]
        if any(value is None for value in values):
            return None
        if all(value == values[0] for value in values[1:]):
            return values[0]
        return None

    @staticmethod
    def _analysis_snapshot_from_direct_source(source):
        if isinstance(source, NDDataset):
            return AnalysisSourceMetadata(source)
        return None

    def _analysis_resolve_source_metadata(self, role, direct_kind, direct_source):
        if direct_kind == "dataset":
            return self._analysis_snapshot_from_direct_source(direct_source)
        if direct_kind == "arraylike":
            return None
        return getattr(self, f"{role}_source_metadata", None)

    def _analysis_reset_history(self, dataset, payload):
        dataset._history = []
        dataset.history = payload

    def _analysis_generated_name(self, role_id, source_metadata=None):
        token = self._analysis_source_token(source_metadata)
        estimator = type(self).__name__
        if token is not None:
            return f"{token}_{estimator}.{role_id}"
        return f"{estimator}.{role_id}"

    def _analysis_generated_description(
        self,
        role_id,
        *,
        sources=None,
        fit_sources=None,
        prediction_source=None,
    ):
        estimator = type(self).__name__
        role_title = self._analysis_role_title(role_id)
        source_list = self._analysis_source_list(sources or [])
        if role_id == "prediction":
            fit_source_list = self._analysis_source_list(fit_sources or [])
            prediction_source_list = self._analysis_source_list([prediction_source])
            return (
                f"Prediction from {estimator} fit of {fit_source_list} "
                f"applied to {prediction_source_list}."
            )
        if role_id == "fitted_data":
            return f"Fitted data from {estimator} fit of {source_list}."
        if role_id == "residuals":
            return f"Residuals from {estimator} fit of {source_list}."
        if role_id in {
            "diagnostic",
            "singular_values",
            "explained_variance",
            "explained_variance_ratio",
            "cumulative_explained_variance",
        }:
            return f"{role_title} diagnostic from {estimator} fit of {source_list}."
        return f"{role_title} from {estimator} fit of {source_list}."

    def _analysis_generated_history(
        self,
        role_id,
        *,
        sources=None,
        fit_sources=None,
        prediction_source=None,
    ):
        estimator = type(self).__name__
        source_list = self._analysis_source_list(sources or [])
        if role_id == "prediction":
            fit_source_list = self._analysis_source_list(fit_sources or [])
            prediction_source_list = self._analysis_source_list([prediction_source])
            return (
                f"Created analysis output prediction with {estimator} from "
                f"{fit_source_list}; applied to {prediction_source_list}."
            )
        if role_id == "fitted_data":
            return f"Created fitted data with {estimator} from {source_list}."
        if role_id == "residuals":
            return f"Created residuals with {estimator} from {source_list}."
        if role_id in {
            "diagnostic",
            "singular_values",
            "explained_variance",
            "explained_variance_ratio",
            "cumulative_explained_variance",
        }:
            return f"Created diagnostic {role_id} with {estimator} from {source_list}."
        return f"Created analysis output {role_id} with {estimator} from {source_list}."

    def _analysis_additional_provenance_sources(
        self,
        role_id,
        *,
        x_source,
        y_source,
        direct_x,
        direct_y,
        direct_x_kind,
        direct_y_kind,
    ):
        return ()

    def _analysis_meta_authority_source(
        self,
        role_id,
        *,
        single_source,
        provenance_sources,
        x_source,
        y_source,
    ):
        return single_source

    def _analysis_units_for_role(
        self,
        role_id,
        *,
        dataset,
        x_source,
        y_source,
        single_source,
        fit_x_source=None,
        fit_y_source=None,
    ):
        role_id = self._normalize_analysis_role(role_id)
        if role_id in {
            "scores",
            "y_scores",
            "components",
            "loadings",
            "weights",
            "rotations",
            "concentration_profiles",
            "spectral_profiles",
            "diagnostic",
            "singular_values",
            "explained_variance",
        }:
            return None
        if role_id in {"explained_variance_ratio", "cumulative_explained_variance"}:
            return "percent"
        if role_id in {"reconstruction", "fitted_data", "residuals"}:
            return x_source.units if x_source is not None else None
        if role_id == "prediction":
            return fit_y_source.units if fit_y_source is not None else None
        return dataset.units

    @staticmethod
    def _analysis_full_rows(mask):
        if mask is None:
            return None
        array = np.asarray(mask)
        if array.ndim < 2:
            return None
        return np.all(array, axis=-1)

    @staticmethod
    def _analysis_full_columns(mask):
        if mask is None:
            return None
        array = np.asarray(mask)
        if array.ndim < 2:
            return None
        return np.all(array, axis=-2)

    @staticmethod
    def _analysis_copy_mask(mask):
        if mask is None or np.isscalar(mask):
            return np.False_
        return np.array(mask, copy=True)

    def _analysis_set_unmasked(self, dataset):
        dataset._mask = NOMASK
        return dataset

    def _analysis_restore_svd_diagnostic_axis(self, dataset):
        return self._analysis_set_unmasked(dataset)

    def _analysis_copy_coordset(self, source):
        if source is None or source.coordset is None:
            return None
        return source.coordset.copy()

    def _analysis_copy_exact_geometry(self, dataset, source):
        if source is None:
            return self._analysis_set_unmasked(dataset)
        dataset.dims = list(source.dims)
        dataset._coordset = self._analysis_copy_coordset(source)
        dataset._mask = self._analysis_copy_mask(source.mask)
        return dataset

    def _analysis_restore_axis_from_source(self, dataset, source, axis):
        if source is None:
            return self._analysis_set_unmasked(dataset)
        full_mask = (
            self._analysis_full_rows(source.mask)
            if axis == 0
            else self._analysis_full_columns(source.mask)
        )
        dataset = self._analysis_apply_full_axis_mask(dataset, full_mask, axis=axis)
        if source.coordset is None or dataset.coordset is None:
            return dataset
        if axis == 0:
            dataset.coordset[dataset.dims[0]] = source.coordset[source.dims[0]].copy()
        elif axis == -1:
            dataset.coordset[dataset.dims[-1]] = source.coordset[source.dims[-1]].copy()
        return dataset

    def _analysis_apply_full_axis_mask(self, dataset, full_mask, axis):
        if full_mask is None:
            return self._analysis_set_unmasked(dataset)
        full_mask = np.asarray(full_mask, dtype=bool)
        if not np.any(full_mask):
            return self._analysis_set_unmasked(dataset)
        if dataset.ndim == 1:
            if axis != 0:
                return self._analysis_set_unmasked(dataset)
            if full_mask.shape[0] == dataset.shape[0]:
                dataset._mask = full_mask.copy()
                return dataset
            if np.count_nonzero(~full_mask) != dataset.shape[0]:
                return self._analysis_set_unmasked(dataset)
            data = np.ma.zeros(full_mask.shape[0], dtype=dataset.dtype)
            data[~full_mask] = np.ma.asarray(dataset.masked_data)
            data[full_mask] = np.ma.masked
            dataset.data = data
            dataset._mask = np.asarray(data.mask)
            return dataset
        mask = np.zeros(dataset.shape, dtype=bool)
        if axis == 0:
            if full_mask.shape[0] == dataset.shape[0]:
                mask[full_mask, ...] = True
            elif np.count_nonzero(~full_mask) == dataset.shape[0]:
                data = np.ma.zeros(
                    (full_mask.shape[0], dataset.shape[1]), dtype=dataset.dtype
                )
                data[~full_mask, :] = np.ma.asarray(dataset.masked_data)
                data[full_mask, :] = np.ma.masked
                dataset.data = data
                mask = np.asarray(data.mask)
            else:
                return self._analysis_set_unmasked(dataset)
        elif axis == -1:
            if full_mask.shape[0] == dataset.shape[-1]:
                mask[..., full_mask] = True
            elif np.count_nonzero(~full_mask) == dataset.shape[-1]:
                data = np.ma.zeros(
                    (dataset.shape[0], full_mask.shape[0]), dtype=dataset.dtype
                )
                data[:, ~full_mask] = np.ma.asarray(dataset.masked_data)
                data[:, full_mask] = np.ma.masked
                dataset.data = data
                mask = np.asarray(data.mask)
            else:
                return self._analysis_set_unmasked(dataset)
        else:
            return self._analysis_set_unmasked(dataset)
        dataset._mask = mask
        return dataset

    def _analysis_restore_full_geometry(self, dataset, source):
        if source is None:
            return self._analysis_set_unmasked(dataset)

        if tuple(dataset.shape) == tuple(source.shape):
            return self._analysis_copy_exact_geometry(dataset, source)

        source_mask = source.mask
        if source_mask is None:
            return self._analysis_set_unmasked(dataset)

        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.ndim == 1:
            if dataset.ndim != 1 or np.count_nonzero(~source_mask) != dataset.shape[0]:
                return self._analysis_set_unmasked(dataset)
            data = np.ma.zeros(source.shape[0], dtype=dataset.dtype)
            data[~source_mask] = np.ma.asarray(dataset.masked_data)
            data[source_mask] = np.ma.masked
            dataset.data = data
            dataset.dims = list(source.dims)
            dataset._coordset = self._analysis_copy_coordset(source)
            dataset._mask = np.asarray(data.mask)
            return dataset

        rows = self._analysis_full_rows(source_mask)
        cols = self._analysis_full_columns(source_mask)
        data = np.ma.asarray(dataset.masked_data)

        if rows.shape[0] == data.shape[0]:
            pass
        elif np.count_nonzero(~rows) == data.shape[0]:
            expanded = np.ma.zeros((rows.shape[0], data.shape[1]), dtype=dataset.dtype)
            expanded[~rows, :] = data
            expanded[rows, :] = np.ma.masked
            data = expanded
        else:
            return self._analysis_set_unmasked(dataset)

        if cols.shape[0] == data.shape[1]:
            pass
        elif np.count_nonzero(~cols) == data.shape[1]:
            expanded = np.ma.zeros((data.shape[0], cols.shape[0]), dtype=dataset.dtype)
            expanded[:, ~cols] = data
            expanded[:, cols] = np.ma.masked
            data = expanded
        else:
            return self._analysis_set_unmasked(dataset)

        data[source_mask] = np.ma.masked
        dataset.data = data
        dataset.dims = list(source.dims)
        dataset._coordset = self._analysis_copy_coordset(source)
        dataset._mask = np.asarray(data.mask)
        return dataset

    def _analysis_reconstruction_geometry_source(
        self,
        role_id,
        *,
        x_source,
        direct_x_kind,
    ):
        if role_id not in {"reconstruction", "fitted_data", "residuals"}:
            return None
        if direct_x_kind != "none":
            return None
        if getattr(self, "_X_original_ndim", 2) == 1:
            return SimpleNamespace(
                shape=tuple(self._X_shape),
                dims=tuple(self._X.dims),
                coordset=copy.copy(self._X_coordset),
                mask=copy.copy(self._X_mask),
            )
        return AnalysisSourceMetadata(self.X)

    def _analysis_prediction_dims_and_coordset(self, dataset, x_predict, y_train):
        if dataset.ndim == 1:
            obs_dim = dataset.dims[0]
            obs_coord = None
            if x_predict is not None and x_predict.coordset is not None:
                source_dim = x_predict.dims[0]
                obs_dim = source_dim
                obs_coord = x_predict.coordset[source_dim].copy()
            dataset.dims = [obs_dim]
            dataset.set_coordset({obs_dim: obs_coord})
            return dataset

        default_obs_dim, default_target_dim = dataset.dims
        obs_dim = default_obs_dim
        target_dim = default_target_dim
        obs_coord = None
        target_coord = None

        if x_predict is not None and len(x_predict.dims) >= 1:
            source_obs_dim = x_predict.dims[0]
            if x_predict.coordset is not None:
                obs_coord = x_predict.coordset[source_obs_dim].copy()
            obs_dim = source_obs_dim

        if y_train is not None and len(y_train.dims) >= 1:
            source_target_dim = y_train.dims[-1]
            if y_train.coordset is not None:
                target_coord = y_train.coordset[source_target_dim].copy()
            target_dim = source_target_dim

        if obs_dim == target_dim:
            obs_dim = default_obs_dim
            target_dim = default_target_dim

        dataset.dims = [obs_dim, target_dim]
        dataset.set_coordset({obs_dim: obs_coord, target_dim: target_coord})
        return dataset

    def _analysis_prediction_mask(self, dataset, x_predict, y_train):
        if dataset.ndim == 1:
            obs_rows = (
                self._analysis_full_rows(x_predict.mask)
                if x_predict is not None
                else None
            )
            return self._analysis_apply_full_axis_mask(dataset, obs_rows, axis=0)

        mask = np.zeros(dataset.shape, dtype=bool)
        contributed = False

        obs_rows = (
            self._analysis_full_rows(x_predict.mask) if x_predict is not None else None
        )
        if (
            obs_rows is not None
            and obs_rows.shape[0] == dataset.shape[0]
            and np.any(obs_rows)
        ):
            mask[obs_rows, :] = True
            contributed = True

        target_cols = (
            self._analysis_full_columns(y_train.mask) if y_train is not None else None
        )
        if (
            target_cols is not None
            and target_cols.shape[0] == dataset.shape[-1]
            and np.any(target_cols)
        ):
            mask[:, target_cols] = True
            contributed = True

        dataset._mask = mask if contributed else NOMASK
        return dataset

    def _apply_analysis_output_geometry(
        self,
        dataset,
        *,
        role_id,
        meta_from,
        direct_x,
        direct_y,
        direct_x_kind,
        direct_y_kind,
    ):
        role_id = self._normalize_analysis_role(role_id)
        x_source = self._analysis_resolve_source_metadata("_X", direct_x_kind, direct_x)
        y_source = self._analysis_resolve_source_metadata("_Y", direct_y_kind, direct_y)

        if role_id in {
            "diagnostic",
            "singular_values",
            "explained_variance",
            "explained_variance_ratio",
            "cumulative_explained_variance",
        }:
            return self._analysis_restore_svd_diagnostic_axis(dataset)

        if role_id == "prediction":
            dataset = self._analysis_prediction_dims_and_coordset(
                dataset, x_source, y_source
            )
            return self._analysis_prediction_mask(dataset, x_source, y_source)

        if (
            role_id in {"reconstruction", "fitted_data", "residuals"}
            and direct_x_kind == "arraylike"
        ):
            dataset._coordset = None
            return self._analysis_set_unmasked(dataset)

        geometry_source = self._analysis_reconstruction_geometry_source(
            role_id,
            x_source=x_source,
            direct_x_kind=direct_x_kind,
        )
        if geometry_source is not None:
            return self._analysis_restore_full_geometry(dataset, geometry_source)

        if role_id in {"scores", "concentration_profiles"}:
            return self._analysis_restore_axis_from_source(dataset, x_source, axis=0)

        if role_id == "y_scores":
            return self._analysis_restore_axis_from_source(dataset, y_source, axis=0)

        if role_id in {
            "components",
            "loadings",
            "weights",
            "rotations",
            "spectral_profiles",
        }:
            authority = x_source
            if role_id in {"loadings", "weights", "rotations"} and meta_from == "_Y":
                authority = y_source
            return self._analysis_restore_axis_from_source(dataset, authority, axis=-1)

        return self._analysis_set_unmasked(dataset)

    def _apply_analysis_output_metadata(
        self,
        dataset,
        *,
        role_id,
        meta_from,
        direct_x,
        direct_y,
        direct_x_kind,
        direct_y_kind,
    ):
        role_id = self._normalize_analysis_role(role_id)
        x_source = self._analysis_resolve_source_metadata("_X", direct_x_kind, direct_x)
        y_source = self._analysis_resolve_source_metadata("_Y", direct_y_kind, direct_y)
        single_source = x_source if meta_from == "_X" else y_source

        if role_id == "prediction":
            x_train = getattr(self, "_X_source_metadata", None)
            y_train = getattr(self, "_Y_source_metadata", None)
            x_predict = x_source
            provenance_sources = [
                source for source in (x_train, y_train, x_predict) if source is not None
            ]
            merged_author = self._analysis_ordered_unique_text(
                [source.author for source in provenance_sources]
            )
            merged_origin = self._analysis_ordered_unique_text(
                [source.origin for source in provenance_sources]
            )
            dataset.author = merged_author
            dataset.origin = "" if merged_origin is None else merged_origin
            dataset.meta = (
                copy.deepcopy(y_train.meta) if y_train is not None else Meta()
            )
            dataset.name = self._analysis_generated_name(role_id, x_predict)
            dataset.title = self._analysis_role_title(role_id)
            dataset.description = self._analysis_generated_description(
                role_id,
                fit_sources=[
                    source for source in (x_train, y_train) if source is not None
                ],
                prediction_source=x_predict,
            )
            dataset.filename = None
            dataset.acquisition_date = self._analysis_acquisition_date_consensus(
                provenance_sources
            )
            self._analysis_reset_history(
                dataset,
                self._analysis_generated_history(
                    role_id,
                    fit_sources=[
                        source for source in (x_train, y_train) if source is not None
                    ],
                    prediction_source=x_predict,
                ),
            )
            dataset.units = self._analysis_units_for_role(
                role_id,
                dataset=dataset,
                x_source=x_predict,
                y_source=y_train,
                single_source=None,
                fit_x_source=x_train,
                fit_y_source=y_train,
            )
            return dataset

        additional_sources = list(
            self._analysis_additional_provenance_sources(
                role_id,
                x_source=x_source,
                y_source=y_source,
                direct_x=direct_x,
                direct_y=direct_y,
                direct_x_kind=direct_x_kind,
                direct_y_kind=direct_y_kind,
            )
        )
        provenance_sources = [single_source] if single_source is not None else []
        provenance_sources.extend(
            source for source in additional_sources if source is not None
        )
        meta_authority = self._analysis_meta_authority_source(
            role_id,
            single_source=single_source,
            provenance_sources=provenance_sources,
            x_source=x_source,
            y_source=y_source,
        )

        if provenance_sources:
            merged_author = self._analysis_ordered_unique_text(
                [source.author for source in provenance_sources]
            )
            merged_origin = self._analysis_ordered_unique_text(
                [source.origin for source in provenance_sources]
            )
            dataset.author = copy.copy(merged_author)
            dataset.origin = "" if merged_origin is None else copy.copy(merged_origin)
            dataset.acquisition_date = self._analysis_acquisition_date_consensus(
                provenance_sources
            )
        else:
            dataset.author = copy.copy(getattr(dataset, "author", None))
            dataset.origin = ""
            dataset.acquisition_date = None

        if meta_authority is not None:
            dataset.meta = copy.deepcopy(meta_authority.meta)
        else:
            dataset.meta = Meta()

        dataset.name = self._analysis_generated_name(role_id, single_source)
        dataset.title = self._analysis_role_title(role_id)
        dataset.description = self._analysis_generated_description(
            role_id, sources=provenance_sources
        )
        dataset.filename = None
        self._analysis_reset_history(
            dataset,
            self._analysis_generated_history(role_id, sources=provenance_sources),
        )
        dataset.units = self._analysis_units_for_role(
            role_id,
            dataset=dataset,
            x_source=x_source,
            y_source=y_source,
            single_source=single_source,
        )
        return dataset

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("_X")
    def _X_default(self):
        raise NotFittedError

    @property
    def _X_is_missing(self):
        # check whether X has been already defined
        try:
            if self._X is None:
                return True
        except NotFittedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods that should be, most of the time, overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):  # pragma: no cover
        #  Intended to be replaced in the subclasses by user defined function
        #  (with the same name)
        raise NotImplementedError("fit method has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods and property
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y=None):
        r"""
        Fit the model with ``X`` as input dataset.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : any
            Depends on the model.

        Returns
        -------
        self
            The fitted instance itself.

        See Also
        --------
        fit_transform :  Fit the model with an input dataset ``X`` and apply the dimensionality reduction on ``X``.
        fit_reduce : Alias of `fit_transform` (Deprecated).

        """
        self._fitted = False  # reinit this flag

        # fire the X and eventually Y validation and preprocessing.
        # X and Y are expected to be resp. NDDataset and NDDataset or list of NDDataset.
        self._capture_source_metadata("_X", X, force=True)
        self._X = X
        self._capture_source_metadata("_Y", Y, force=True)
        if Y is not None:
            self._Y = Y

        # _X_preprocessed has been computed when X was set, as well as _Y_preprocessed.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed
        newY = self._Y_preprocessed if Y is not None else None

        # Call to the actual _fit method (overloaded in the subclass)
        # warning : _fit must take ndarray arguments not NDDataset arguments.
        # when method must return NDDataset from the calculated data,
        # we use the decorator _wrap_ndarray_output_to_nddataset, as in the PCA
        # model for example.
        try:
            self._outfit = self._fit(newX, newY)
        except TypeError:
            # in case Y s not used in _fit
            self._outfit = self._fit(newX)

        # if the process was successful, _fitted is set to True so that other method
        # which needs fit will be possibly used.
        self._fitted = True
        return self

    # we do not use this method as a decorator as in this case signature of subclasses
    # extract useful individual parameters doc

    @property
    def log(self):
        """Return ``log`` output."""
        # A string handler (#1) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[1].stream.getvalue().rstrip()

    @property
    def X(self):
        """Return the X input dataset (eventually modified by the model)."""
        if self._X_is_missing:
            raise NotFittedError
        # We use X property only to show this information to the end user. Internally
        # we use _X attribute to refer to the input data
        X = self._X.copy()
        if np.any(self._X_mask):
            # restore masked row and column if necessary
            X = self._restore_masked_data(X, axis="both")
        if self._is_dataset or self._output_type == "NDDataset":
            return X
        return np.asarray(X)

    def get_params(self, deep=True):
        r"""
        Get the configuration parameters of this estimator.

        Parameters
        ----------
        deep : `bool`, optional, default:`True`
            Ignored.  Present for compatibility with scikit-learn conventions.

        Returns
        -------
        `dict`
            Mapping of parameter name -> current value.

        """
        return dict(self.params())

    def set_params(self, **params):
        r"""
        Set configuration parameters on this estimator.

        Returns `self` so that calls can be chained.

        Parameters
        ----------
        **params
            Parameter names and values to update.

        Returns
        -------
        self
            The estimator instance.

        Raises
        ------
        SpectroChemPyError
            If a parameter name does not correspond to a configurable trait.

        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise SpectroChemPyError(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        params = self.get_params()
        # Show a concise subset for readability
        display = {k: v for k, v in params.items() if not k.startswith("_")}
        if not display:
            return f"{cls}()"
        items = ", ".join(f"{k}={v!r}" for k, v in display.items())
        return f"{cls}({items})"


# ======================================================================================
# Base class DecompositionAnalysis
# ======================================================================================
class DecompositionAnalysis(AnalysisConfigurable):
    """
    Abstract class to write analysis decomposition models such as `PCA`, ...

    Subclass this to get a minimal structure

    See Also
    --------
    EFA : Perform an Evolving Factor Analysis (forward and reverse).
    FastICA : Perform Independent Component Analysis with a fast algorithm.
    IRIS : Integral inversion solver for spectroscopic data.
    MCRALS : Perform MCR-ALS of a dataset knowing the initial :math:`C` or :math:`S^T` matrix.
    NMF : Non-Negative Matrix Factorization.
    PCA : Perform Principal Components Analysis.
    SIMPLISMA : SIMPLe to use Interactive Self-modeling Mixture Analysis.
    SVD : Perform a Singular Value Decomposition.

    """

    # This class is subclass AnalysisConfigurable, so we define only additional
    # attributes and methods necessary for decomposition model.

    # Get doc sections for reuse in subclass

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    _Y = tr.Union(
        (
            tr.Tuple(NDDatasetType(), NDDatasetType()),
            NDDatasetType(),
        ),
        default_value=None,
        allow_none=True,
        help="Target/profiles taken into account to fit a model",
    )
    _Y_preprocessed = tr.Union((tr.List(Array()), Array()), help="preprocessed Y")
    _n_components = tr.Integer(help="""The actual number of components.""")
    _components = Array(help="the array of (n_components, n_features) components")

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.validate("_Y")
    def _Y_validate(self, proposal):
        # validation of the _Y attribute: fired when self._Y is assigned
        Y = proposal.value

        # we need a dataset or a list of NDDataset
        return self._make_dataset(Y)

    @property
    def _Y_is_missing(self):
        # check whether or not Y has been already defined
        try:
            if self._Y is None:
                return True
        except NotFittedError:
            return True
        return False

    @tr.default("_n_components")
    def _n_components_default(self):
        # ensure model fitted before using this value
        if not self._fitted:
            raise NotFittedError("_n_components")

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        Y = change.new
        # optional preprocessing as scaling, centering, ...
        # return a np.ndarray
        self._Y_preprocessed = Y.data

    def _transform(self, *args, **kwargs):  # pragma:  no cover
        # to be overridden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("transform has not yet been implemented")

    def _inverse_transform(self, *args, **kwargs):  # pragma:  no cover
        # to be overridden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("inverse_transform has not yet been implemented")

    def _get_components(self, n_components=None):  # pragma:  no cover
        # to be overridden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("get_components has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typex="components",
        analysis_role="scores",
    )
    def transform(self, X=None, **kwargs):
        r"""
        Apply dimensionality reduction to `X`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction. If not given
            the number of components is eventually the one specified or determined
            in the `fit` process.

        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X.copy()

        # Get the processed ndarray data
        newX = self._X_preprocessed

        X_transform = self._transform(newX)

        # Slice according to n_components
        n_components = kwargs.pop("n_components", self._n_components)
        if n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value.",
                stacklevel=2,
            )
        if n_components < self._n_components:
            X_transform = X_transform[:, :n_components]

        return X_transform

    # Get doc sections for reuse in subclass

    @_wrap_ndarray_output_to_nddataset(analysis_role="reconstruction")
    def inverse_transform(self, X_transform=None, **kwargs):
        r"""
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would
        be `X_transform`.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`, :term:`n_components`), optional
            Reduced `X` data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If `X_transform` is not
            provided, a transform of `X` provided in `fit` is performed first.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_features`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reconstruction.

        See Also
        --------
        reconstruct : Alias of inverse_transform (Deprecated).

        """
        if not self._fitted:
            raise NotFittedError

        # get optional n_components
        n_components = kwargs.pop("n_components", self._n_components)
        if n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value.",
                stacklevel=2,
            )

        if isinstance(X_transform, NDDataset):
            X_transform = X_transform.data
            if n_components > X_transform.shape[1]:
                warnings.warn(
                    "The number of components required for reduction "
                    "cannot be greater than the X_transform size : "
                    f"{X_transform.shape[1]}. We then use this latter value.",
                    stacklevel=2,
                )
        elif X_transform is None:
            X_transform = self.transform(**kwargs).data

        return self._inverse_transform(X_transform)

    def fit_transform(self, X, Y=None, **kwargs):
        r"""
        Fit the model with `X` and apply the dimensionality reduction on `X`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : any
            Depends on the model.
        **kwargs : keyword arguments, optional
            Additional keyword arguments passed to the underlying implementation.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction.

        """
        try:
            self.fit(X, Y)
        except TypeError:
            # the current model does not use Y
            self.fit(X)
        return self.transform(X, **kwargs)

    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typey="components",
        analysis_role="components",
    )
    def get_components(self, n_components=None):
        r"""
        Return the component's dataset: (selected :term:`n_components`, :term:`n_features`).

        Parameters
        ----------
        n_components : `int`, optional, default: `None`
            The number of components to keep in the output dataset.
            If `None`, all calculated components are returned.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            Dataset with shape (:term:`n_components`, :term:`n_features`)

        """
        if n_components is None or n_components > self._n_components:
            n_components = self._n_components

        # we call the specific _get_components method defined in subclasses
        return self._get_components()[:n_components]

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title="keep",
        typey="components",
        analysis_role="components",
    )
    def components(self):
        r"""
        `NDDataset` with components in feature space (:term:`n_components`, :term:`n_features`).

        See Also
        --------
        get_components : Retrieve only the specified number of components.

        """
        return self._get_components()

    @property
    def n_components(self):
        """Number of components that were fitted."""
        if self._fitted:
            return self._n_components
        raise NotFittedError("n_components")

    def _get_component_labels(self, n):
        """
        Return default labels for *n* components.

        Subclasses may override this to provide domain-specific labels
        (e.g. ``PC1``, ``PC2``, ... for PCA).
        """
        return [f"#{i}" for i in range(n)]

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    def plot_merit(self, X=None, X_hat=None, **kwargs):
        r"""
        Plot the input (`X`), reconstructed (`X_hat`) and residuals.

        :math:`X` and :math:`\hat{X}` can be passed as arguments. If not,
        the `X` attribute is used for :math:`X`and :math:`\hat{X}`is computed by
        the `inverse_transform` method

        Parameters
        ----------
        X : `NDDataset`, optional
            Original dataset. If is not provided (default), the `X`
            attribute is used and X_hat is computed using `inverse_transform`.
        X_hat : `NDDataset`, optional
            Inverse transformed dataset. if `X` is provided, `X_hat`
            must also be provided as compuyed externally.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib subplot axe.

        Other Parameters
        ----------------
        exp_c : color, colormap, or list of colors, optional
            Color(s) for experimental spectra.
            - None: use unified semantic resolver (auto-detect categorical/sequential)
            - Single color: use for all experimental spectra
            - Colormap name/object: sample colors from colormap
            - List/tuple: use as explicit color cycle
        calc_c : color, colormap, or list of colors, optional
            Color(s) for calculated spectra.
            - None: use default blue "#2a6fbb"
            - Single color: use for all calculated spectra
            - Colormap name/object: sample colors from colormap
            - List/tuple: use as explicit color cycle
        resid_c : color, colormap, or list of colors, optional
            Color(s) for residual spectra.
            - None: use default grey "0.4"
            - Single color: use for all residual spectra
            - Colormap name/object: sample colors from colormap
            - List/tuple: use as explicit color cycle
        exp_linestyle : str, optional
            Line style for experimental spectra. Default: "-".
        calc_linestyle : str, optional
            Line style for calculated spectra. Default: "--".
        resid_linestyle : str, optional
            Line style for residual spectra. Default: "-".
        exp_linewidth : float, optional
            Line width for experimental spectra. Default: 1.2.
        calc_linewidth : float, optional
            Line width for calculated spectra. Default: 1.0.
        resid_linewidth : float, optional
            Line width for residual spectra. Default: 1.0.
        min_contrast : float, optional
            Minimum contrast ratio for sequential colormaps. Default: 1.5.
        offset : `float`, optional, default: `None`
            Specify the separation (in percent) between the
            :math:`X` , :math:`X_hat` and :math:`E`.
        nb_traces : `int` or ``'all'``, optional
            Number of lines to display. Default is ``'all'``.
        **others : Other keywords parameters
            Parameters passed to the internal `plot` method of the `X` dataset.
            Common options include ``color``, ``linewidth``, ``linestyle``,
            ``alpha``, and standard Matplotlib kwargs.

        """
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        return plot_merit(
            analysis_object=self,
            X=X,
            X_hat=X_hat,
            **kwargs,
        )

    # Backward compatibility alias
    @deprecated(replace="plot_merit", removed="0.13.0")
    def plotmerit(self, X=None, X_hat=None, **kwargs):
        """
        Backward-compatible alias for :meth:`plot_merit`. Deprecated.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes containing the plot.
        """
        return self.plot_merit(X, X_hat, **kwargs)

    @property
    def Y(self):
        r"""The `Y` input."""
        # We use Y property only to show this information to the end-user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        return self._Y


# ======================================================================================
# Base class CrossDecompositionAnalysis
# ======================================================================================
class CrossDecompositionAnalysis(DecompositionAnalysis):
    """
    Abstract class to write analysis cross decomposition models such as `PLSRegression`, ...

    Subclass this to get a minimal structure

    See Also
    --------
    PLSRegression : Perform a Partial Least Square Regression .

    """

    # This class is a subclass of DecompositionAnalysis, so we define only additional
    # attributes and methods necessary for cross decomposition model.

    # Get doc sections for reuse in subclass

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _predict(self, *args, **kwargs):  # pragma:  no cover
        # to be overridden in subclass such as PLSRegression, ...
        raise NotImplementedError("predict has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------

    @_wrap_ndarray_output_to_nddataset(
        meta_from="_Y",
        title=None,
        use_snapshot=False,
        analysis_role="prediction",
    )
    def predict(self, X=None):
        r"""
        Predict targets of given observations.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.

        Returns
        -------
        `NDDataset`
            Datasets with shape (:term:`n_observations`,) or ( :term:`n_observations`, :term:`n_targets`).

        """
        if not self._fitted:
            raise NotFittedError()

        if X is None:
            X = self._X_preprocessed
        elif isinstance(X, NDDataset):
            X = X.data

        return self._predict(X)

    def score(self, X=None, Y=None, sample_weight=None):
        r"""
        Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})` , where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is ``1.0`` and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `Y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            Test samples. If not given, the X attribute is used.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_targets`), optional
            True values for `X`.
        sample_weight : `NDDataset` or :term:`array-like` of shape (:term:`n_samples`,), default: `None`
            Sample weights.

        Returns
        -------
        `float`
            :math:`R^2` of `predict`(X) w.r.t `Y`.

        """
        if not self._fitted:
            raise NotFittedError()

        if X is None:
            X = self._X_preprocessed
        elif isinstance(X, NDDataset):
            X = X.data

        if Y is None:
            Y = self._Y_preprocessed
        elif isinstance(Y, NDDataset):
            Y = Y.data

        if isinstance(sample_weight, NDDataset):
            sample_weight = sample_weight.data

        return self._score(X, Y, sample_weight)

    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        meta_from=("_X", "_Y"),
        typex="components",
        analysis_role=("scores", "y_scores"),
    )
    def transform(self, X=None, Y=None, both=False, **kwargs):
        r"""
        Apply dimensionality reduction to `X`and `Y`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_targets`), optional
            New data, where :term:`n_targets` is the number of variables to predict.
            if not provided, the input dataset of the `fit` method will be used.
        both : `bool`, default: `False`
            Whether to also apply the dimensionality reduction to Y when neither X nor Y are provided.

        Returns
        -------
        x_score, y_score: `NDDataset` or tuple of `NDDataset`
            Datasets with shape (:term:`n_observations`, :term:`n_components`).

        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X
        self._Y = Y if Y is not None else self.Y

        # Get the processed ndarray data
        newX = self._X_preprocessed
        newY = self._Y_preprocessed

        if both or (Y is not None):
            return self._transform(newX, newY)
        return self._transform(newX, None)

    # Get doc sections for reuse in subclass

    @_wrap_ndarray_output_to_nddataset(meta_from=("_X", "_Y"))
    def inverse_transform(
        self,
        X_transform=None,
        Y_transform=None,
        both=False,
        **kwargs,
    ):
        r"""
        Transform data back to its original space.

        In other words, return reconstructed `X` and `Y` whose reduce/transform would
        be `X_transform` and `Y_transform`.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`, :term:`n_components`), optional
            Reduced `X` data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If `X_transform` is not
            provided, a transform of `X` provided in `fit` is performed first.
        Y_transform : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, `n_components`), optional
            New data, where :term:`n_targets` is the number of variables to predict. If `Y_transform` is not
            provided, a transform of `Y` provided in `fit` is performed first.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction.

        See Also
        --------
        reconstruct : Alias of inverse_transform (Deprecated).

        """
        if not self._fitted:
            raise NotFittedError

        if isinstance(X_transform, NDDataset):
            X_transform = X_transform.data

        elif X_transform is None:
            X_transform = self.transform(**kwargs).data

        if isinstance(Y_transform, NDDataset):
            Y_transform = Y_transform.data

        elif Y_transform is None and both is True:
            Y_transform = self.transform(**kwargs).data

        if Y_transform is None:
            return self._inverse_transform(X_transform)
        X, Y = self._inverse_transform(X_transform, X_transform)
        return X, Y

    def fit_transform(self, X, Y, both=False):
        r"""
        Fit the model with `X` and `Y` and apply the dimensionality reduction on `X` and optionally on `Y`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        both : `bool`, optional
            Whether to also apply the dimensionality reduction to Y when neither X nor Y are provided.

        Returns
        -------
        `NDDataset` or tuple of `NDDataset`
            Transformed data.

        """
        try:
            result = self.fit(X, Y).transform(X, Y, both=both)
            # fit_transform should return only x_scores by default (not a tuple)
            if both:
                return result
            # result could be a tuple from _transform - return only x_scores
            if isinstance(result, tuple):
                return result[0]
            return result
        except NotFittedError:
            # If transform failed, return None
            return None

    def plot_parity(
        self,
        Y=None,
        Y_hat=None,
        *,
        ax=None,
        clear=True,
        show=True,
        **kwargs,
    ):
        r"""
        Plot the predicted (:math:`\hat{Y}`) vs measured (:math:`Y`) values.

        :math:`Y` and :math:`\hat{Y}` can be passed as arguments. If not,
        the `Y` attribute is used for :math:`\hat{Y}` computed by
        the `predict` method.

        Parameters
        ----------
        Y : `NDDataset`, optional
            Measured values. If not provided, uses ``self.Y`` and computes
            ``Y_hat`` via ``self.predict(self.X)``.
        Y_hat : `NDDataset`, optional
            Predicted values. If ``Y`` is provided, ``Y_hat`` must also be
            provided as computed externally.
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. If None, a new figure is created.
        clear : `bool`, optional
            Whether to clear the axes before plotting. Default: True.
            Only used when ``ax`` is provided.
        show : `bool`, optional
            Whether to display the figure. Default: True.
        **kwargs : keyword arguments, optional
            Additional keyword arguments passed to
            `~matplotlib.axes.Axes.scatter`. Includes ``s``, ``c``, ``marker``,
            ``cmap``, ``norm``, ``vmin``, ``vmax``, ``alpha``, ``linewidths``,
            ``edgecolors``, ``plotnonfinite``.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes containing the parity plot.

        See Also
        --------
        parityplot : Deprecated alias for this method.
        """
        if Y is None:
            Y = self.Y
            if Y_hat is None:
                Y_hat = self.predict(self.X)
        elif Y_hat is None:
            raise ValueError(
                "If Y is provided, an externally computed Y_hat dataset "
                "must be also provided.",
            )

        from spectrochempy.plotting.composite.parity import plot_parity as _plot_parity

        return _plot_parity(Y, Y_hat, ax=ax, clear=clear, show=show, **kwargs)

    # Backward compatibility alias
    @deprecated(replace="plot_parity", removed="0.13.0")
    def parityplot(
        self, Y=None, Y_hat=None, *, ax=None, clear=True, show=True, **kwargs
    ):
        """
        Backward-compatible alias for :meth:`plot_parity`. Deprecated.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes containing the parity plot.
        """
        return self.plot_parity(Y, Y_hat, ax=ax, clear=clear, show=show, **kwargs)


# ======================================================================================
# Base class LinearRegressionAnalysis
# ======================================================================================
class LinearRegressionAnalysis(AnalysisConfigurable):
    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    fit_intercept = tr.Bool(
        default_value=True,
        help="Whether to calculate the `intercept` for this model. If set to `False`, "
        "no `intercept` will be used in calculations (*i.e.,* data is expected to be "
        "centered).",
    ).tag(config=True)

    positive = tr.Bool(
        default_value=False,
        help=r"When set to `True` , forces the coefficients (`coef`) "
        r"to be positive.",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    _Y = NDDatasetType()
    _Y_preprocessed = Array(help="preprocessed Y")

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # initialize sklearn LinearRegression
        self._linear_regression = linear_model.LinearRegression(
            fit_intercept=self.fit_intercept,
            n_jobs=None,  # not used for the moment (XXX: should we add this?)
            positive=self.positive,
        )

        # unlike decomposition methods, we output ndarray when the input
        # is not a dataset
        self._output_type = "ndarray"

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.validate("_Y")
    def _Y_validate(self, proposal):
        # validation of the _Y attribute: fired when self._Y is assigned
        Y = proposal.value

        # we need a dataset or a list of NDDataset
        return self._make_dataset(Y)

    @property
    def _Y_is_missing(self):
        # check whether or not Y has been already defined
        try:
            if self._Y is None:
                return True
        except NotFittedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        Y = change.new
        # optional preprocessing as scaling, centering, ...
        # return a np.ndarray
        self._Y_preprocessed = Y.data

    def _fit(self, X, Y=None, sample_weight=None):
        # this method is called by the abstract class fit.
        return self._linear_regression.fit(X, Y, sample_weight=sample_weight)

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y=None, sample_weight=None):
        r"""
        Fit linear model.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`,:term:`n_features`)
            Training data, where `n_observations` is the number of observations
            and `n_features` is the number of features.
        Y : :term:`array-like` of shape (:term:`n_observations`,) or (:term:`n_observations`,:term:`n_targets`)
            Target values. Will be cast to `X`'s dtype if necessary.
        sample_weight : :term:`array-like` of shape (:term:`n_observations`,), default: `None`
            Individual weights for each observation.

        Returns
        -------
        self
            Returns the instance itself.

        """
        self._fitted = False  # reiniit this flag

        # store if the original input type is a dataset (or at least a subclass instance
        # of NDArray)
        self._is_dataset = isinstance(X, NDArray)

        def _make2D(X):
            # For regression analysis we need X as a NDDataset with two dimensions
            # IF X is 1D, then we add a dimension at the end.
            X = NDDataset(X)
            if X.ndim == 1:
                coordset = X.coordset
                X._data = X._data[:, np.newaxis]
                if np.any(X.mask):
                    X._mask = X._mask[:, np.newaxis]
                X.dims = ["x", "a"]
                coordx = coordset[0] if coordset is not None else None
                X.set_coordset(x=coordx, a=None)
            return X

        # fire the X and Y validation and preprocessing.
        if Y is not None:
            self._capture_source_metadata("_X", X, force=True)
            self._X = _make2D(X)
            self._capture_source_metadata("_Y", Y, force=True)
            self._Y = Y
        else:
            # X should contain the X and Y information (X being the coord and Y the data)
            if X.coordset is None:
                raise ValueError(
                    "The passed argument must have a x coordinates,"
                    "or X input and Y target must be passed separately",
                )
            self._capture_source_metadata("_X", X.coord(0), force=True)
            self._X = _make2D(X.coord(0))
            self._capture_source_metadata("_Y", X, force=True)
            self._Y = X

        # _X_preprocessed has been computed when X was set, as well as _Y_preprocessed.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed
        newY = self._Y_preprocessed

        # call to the actual _fit method (overloaded in the subclass)
        # warning : _fit must take ndarray arguments not NDDataset arguments.
        # when method must return NDDataset from the calculated data,
        # we use the decorator _wrap_ndarray_output_to_nddataset, as below or in the PCA
        # model for example.
        self._outfit = self._fit(newX, newY, sample_weight=sample_weight)

        # if the process was successful,_fitted is set to True so that other method which
        # needs fit will be possibly used.
        self._fitted = True
        return self

    @property
    def Y(self):
        """Return the `Y` input dataset."""
        # We use Y property only to show this information to the end user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        Y = self._Y
        if self._is_dataset or self._output_type == "NDDataset":
            return Y
        return np.asarray(Y)

    @property
    def coef(self):
        r"""
        Estimated coefficients for the linear regression problem.

        If multiple targets are passed during the fit (Y 2D), this is a 2D array of
        shape (:term:`n_targets`, :term:`n_features`), while if only one target
        is passed, this is a 1D array of length :term:`n_features`.
        """
        if self._linear_regression.coef_.size == 1:
            # this is the result of the single equation, so only one value
            # should be returned
            if self._linear_regression.coef_.ndim == 0:
                A = float(self._linear_regression.coef_)
            else:
                A = float(self._linear_regression.coef_[0])
            if self._is_dataset and self._Y.has_units and self._X.has_units:
                A = A * self._Y.units / self._X.units
        elif self._is_dataset:
            unitsX = self._X.units if self._X.units is not None else 1.0
            unitsY = self._Y.units if self._Y.units is not None else 1.0
            if unitsX != 1 or unitsY != 1:
                units = self._Y.units / self._X.units
            else:
                units = None

            A = type(self._X)(
                data=self._linear_regression.coef_,
                coordset=self._Y.coordset,
                dims=self._Y.T.dims,
                units=units,
                title=f"{self._Y.title} / {self._X.title}",
                history="Computed from the LSTSQ model",
            )
        return A

    @property
    def intercept(self):
        r"""
        Return a float or an array of shape (:term:`n_targets`,).

        Independent term in the linear model. Set to ``0.0`` if `fit_intercept` is `False`.
        If `Y` has units, then `intercept` has the same units.
        """
        if self._linear_regression.intercept_.size == 1:
            # A single value, return the associated quantity
            B = self._linear_regression.intercept_
            if self._is_dataset and self._Y.has_units:
                B = B * self._Y.units
        elif self._is_dataset:
            # else, return a NDDataset with the same units has Y
            B = type(self._X)(
                data=self._linear_regression.intercept_,
                coordset=self._Y.coordset,
                dims=self._Y.dims,
                units=self._Y.units,
                title=f"{self._Y.title} at origin",
                history="Computed from the LSTSQ model",
            )
        return B

    def predict(self, X=None):
        r"""
        Predict features using the linear model.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` matrix, shape (:term:`n_observations`,:term:`n_features`)
            Observations. If `X` is not set, the input `X` for `fit` is used.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            Predicted values (object of type of the input) using a ahape (:term:`n_observations`,).

        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        if X is not None:
            # _is_dataset = hasattr(X, "_implements") and X._implements("NDDataset")
            self._X = X

        # Get the processed ndarray data
        newX = self._X_preprocessed

        predicted = self._linear_regression.predict(newX)

        if self._is_dataset:
            predicted = type(self._X)(
                predicted,
                coordset=self._Y.coordset,
                dims=self._Y._dims,
                units=self._Y.units,
                title=self._Y.title,
                history="Computed from a LSTSQ model",
            )

        return predicted

    def score(self, X=None, Y=None, sample_weight=None):
        r"""
        Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})` , where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()`` .
        The best possible score is ``1.0`` and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `Y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Test samples.

        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`,)
            True values for `X`.

        sample_weight : :term:`array-like` of shape (:term:`n_observations`,), default: `None`
            Sample weights.

        Returns
        -------
        `float`
            :math:`R^2` of `predict` (`X` ) wrt. `Y` .

        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X
        self._Y = Y if X is not None else self.Y

        # Get the processed ndarray data
        newX = self._X_preprocessed
        newY = self._Y_preprocessed

        return self._linear_regression.score(newX, newY, sample_weight=sample_weight)
