"""A Markov prediction model for forecasting fatalities.
Key differences from the original:

* Subclasses :class:`darts.models.forecasting.sklearn_model.SKLearnModel` so it
  plugs into the existing Darts pipeline (save/load, fit/predict contract).
  Internally, however, the Markov logic owns the train/predict flow — the
  inherited ``SKLearnModel.fit`` / ``SKLearnModel.predict`` are NOT used. We
  only inherit the darts ``ForecastingModel`` machinery (input validation,
  encoders, save/load pickle).
* No pandas dependency: the original used ``pd.DataFrame`` with a
  ``(month_id, country_id)`` MultiIndex everywhere. This implementation works
  with flat numpy arrays + parallel index arrays (``time_ids``,
  ``entity_ids``). The conversion from a list of Darts ``TimeSeries`` to the
  flat representation happens once, at the top of :meth:`MarkovModel.fit` and
  :meth:`MarkovModel.predict`.
* Multivariate forecasting: the original raised ``NotImplementedError`` for
  more than one target. This implementation lifts that restriction by
  training a separate :class:`MarkovFatalityModel` per target column. The
  Markov *state* model is shared across targets (the state is computed from
  the ``markov_target`` column, which is a single fatality column even in
  multi-target configurations).
* Uses sklearn's :class:`RandomForestClassifier` and
  :class:`RandomForestRegressor` directly. Darts' ``RandomForestModel`` is a
  thin wrapper around sklearn's ``RandomForestRegressor`` and does not expose
  a classifier variant; using sklearn directly avoids unnecessary indirection
  while preserving the user's intent ("use the sklearn RandomForest
  implementations that darts also uses internally").
"""

from __future__ import annotations

import logging
import pickle
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _check_is_fitted(obj: Any, attr: str = "is_fitted_") -> None:
    """Lightweight replacement for sklearn's ``check_is_fitted``.

    sklearn's ``check_is_fitted`` requires the estimator to implement
    ``__sklearn_tags__`` (via :class:`BaseEstimator`). Darts models do not
    inherit from :class:`BaseEstimator`, so we use a direct attribute check
    instead — same semantics, no sklearn tag plumbing required.
    """
    if not getattr(obj, attr, False):
        raise RuntimeError(
            f"This {type(obj).__name__} instance is not fitted yet. "
            f"Call `fit` before using this model."
        )

from darts import TimeSeries
from darts.models.forecasting.sklearn_model import SKLearnModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Markov state enum
# ----------------------------------------------------------------------


class MarkovState(str, Enum):
    """An enumeration of the Markov states used in the Markov model."""

    PEACE = "peace"
    DESC = "desc"
    ESC = "esc"
    WAR = "war"


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _as_markov_state(value: Any) -> Optional[MarkovState]:
    """Coerce a string / MarkovState / numpy value to ``MarkovState``.

    Returns ``None`` for ``None`` / NaN inputs (the numpy equivalent of the
    original ``pd.NA`` return path).
    """
    if value is None:
        return None
    if isinstance(value, MarkovState):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
    s = str(value)
    if s in ("", "nan", "None", "<NA>"):
        return None
    try:
        return MarkovState(s)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Markov state classifier
# ----------------------------------------------------------------------


class MarkovStateModel:
    """A Markov state prediction model.

    Predicts the probability of each Markov state in a future month, given the
    current month's state and a set of features, for a given step size. One
    :class:`RandomForestClassifier` is fitted per starting state — only
    samples whose ``markov_state`` equals the starting state are used to fit
    that classifier.
    """

    def __init__(
        self,
        step: int,
        train_start: int,
        train_end: int,
        rf_class_params: Optional[dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Store hyperparameters; no fitting is performed.

        Args:
            step: Number of months ahead to predict the Markov state for.
            train_start: Start month id of the training window (inclusive).
            train_end: End month id of the training window (inclusive).
            rf_class_params: Optional kwargs for the ``RandomForestClassifier``.
            random_state: Random seed.
            n_jobs: Parallelism for the underlying sklearn estimator.
        """
        self.step = step
        self.train_start = train_start
        self.train_end = train_end
        self._rf_class_params = rf_class_params if rf_class_params is not None else {}
        self._random_state = random_state
        self._n_jobs = n_jobs
        self.models: dict[MarkovState, RandomForestClassifier] = {}
        self._markov_states: list[MarkovState] = list(MarkovState)
        self.is_fitted_: bool = False
        self._feature_idx: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        *,
        values: np.ndarray,  # (N, F) float32 — feature matrix
        time_ids: np.ndarray,  # (N,) int64 — month_id per row
        entity_ids: np.ndarray,  # (N,) int64 — entity id per row
        markov_state: np.ndarray,  # (N,) object — MarkovState per row
        markov_target: np.ndarray,  # (N,) float32 — fatality count per row
        feature_idx: np.ndarray,  # (F_markov,) int — column indices for state features
    ) -> "MarkovStateModel":
        """Fit one classifier per starting state.

        Args:
            values: Flat ``(N, F)`` feature matrix (features + targets).
            time_ids: Per-row time id (month_id).
            entity_ids: Per-row entity id (country_id / priogrid_id).
            markov_state: Per-row current Markov state (as ``MarkovState``
                enum, string, or ``None``).
            markov_target: Per-row fatality count column (used to compute the
                shifted target state; this mirrors the original code, which
                shifted the ``markov_state`` column directly — here we recompute
                the target state from the shifted fatalities to keep the
                numpy path explicit).
            feature_idx: Column indices into ``values`` for the markov-state
                features.
        """
        # --- Sort by (entity, time) so we can shift within entity groups.
        order = np.lexsort((time_ids, entity_ids))
        values = values[order]
        time_ids = time_ids[order]
        entity_ids = entity_ids[order]
        markov_state = markov_state[order]
        markov_target = np.asarray(markov_target, dtype=np.float64)[order]

        # --- Compute target state: shift markov_target by -step within entity.
        target_time = time_ids + self.step
        # Build a (entity, time) -> row index map for the shift lookup.
        # For each row, we want the row in the same entity whose time == time + step.
        # Use a dict on (entity, time) -> row index.
        lookup: dict[tuple[int, int], int] = {}
        for i, (e, t) in enumerate(zip(entity_ids.tolist(), time_ids.tolist())):
            lookup[(e, t)] = i

        target_state = np.empty(len(values), dtype=object)
        target_state[:] = None
        for i, (e, t) in enumerate(zip(entity_ids.tolist(), time_ids.tolist())):
            tgt_row = lookup.get((e, t + self.step))
            if tgt_row is not None:
                # Markov state of the target month = state computed from
                # markov_target at target row (already computed externally).
                # Here, we recompute the target state from the shifted
                # markov_target and the current markov_target. But the
                # original code shifts the *markov_state* column directly —
                # so target_state[i] = markov_state[tgt_row].
                target_state[i] = markov_state[tgt_row]

        # --- Filter to training window (based on target_time).
        train_mask = (target_time >= self.train_start) & (target_time <= self.train_end)
        # Also drop rows where target_state is None (no future observation).
        has_target = np.array(
            [s is not None for s in target_state], dtype=bool
        )
        valid_mask = train_mask & has_target

        self._feature_idx = np.asarray(feature_idx, dtype=np.int64)
        X_all = values[:, self._feature_idx]

        # --- Train one classifier per starting state.
        for state in self._markov_states:
            state_mask = np.array(
                [s == state for s in markov_state], dtype=bool
            )
            sub_mask = valid_mask & state_mask
            if sub_mask.sum() == 0:
                logger.warning(
                    "MarkovStateModel(step=%d): no training samples for "
                    "starting state %s; fitting a dummy classifier.",
                    self.step, state,
                )
                # Fit a dummy classifier that always predicts PEACE so the
                # predict_proba path doesn't crash. This matches the
                # original behaviour (the RF would also fail to fit on
                # zero samples).
                rf = RandomForestClassifier(
                    n_estimators=1,
                    random_state=self._random_state,
                    n_jobs=self._n_jobs,
                )
                dummy_X = np.zeros((1, len(self._feature_idx)), dtype=np.float32)
                dummy_y = np.array([MarkovState.PEACE.value], dtype=object)
                rf.fit(dummy_X, dummy_y)
                self.models[state] = rf
                continue

            X_train = X_all[sub_mask]
            y_train = np.array(
                [target_state[i] for i in np.where(sub_mask)[0]], dtype=object
            )
            rf = RandomForestClassifier(
                random_state=self._random_state,
                n_jobs=self._n_jobs,
                **self._rf_class_params,
            )
            rf.fit(X_train, y_train)
            self.models[state] = rf

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        *,
        values: np.ndarray,  # (N, F)
        start_state: MarkovState,
    ) -> np.ndarray:
        """Predict the probability of each target state given a starting state.

        Args:
            values: ``(N, F)`` feature matrix.
            start_state: Current-month Markov state.

        Returns:
            ``(N, n_seen_classes)`` probability matrix. The class order is
            available on ``self.models[start_state].classes_``.
        """
        _check_is_fitted(self, "is_fitted_")
        model = self.models[start_state]
        X = values[:, self._feature_idx]
        return model.predict_proba(X)


# ----------------------------------------------------------------------
# Markov fatality regressor
# ----------------------------------------------------------------------


class MarkovFatalityModel:
    """A Markov fatality prediction model.

    Predicts the number of fatalities in a future month, given the predicted
    month's state and a set of features. One :class:`RandomForestRegressor`
    is fitted per state — but only for the escalation (``ESC``) and war
    (``WAR``) states, since peace and de-escalation are assumed to contribute
    zero fatalities.

    Numpy-only — no pandas dependency.
    """

    def __init__(
        self,
        step: int,
        train_start: int,
        train_end: int,
        rf_reg_params: Optional[dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Store hyperparameters; no fitting is performed.

        Args:
            step: Number of months ahead to predict fatalities for.
            train_start: Start month id of the training window (inclusive).
            train_end: End month id of the training window (inclusive).
            rf_reg_params: Optional kwargs for ``RandomForestRegressor``.
            random_state: Random seed.
            n_jobs: Parallelism for the underlying sklearn estimator.
        """
        self.step = step
        self.train_start = train_start
        self.train_end = train_end
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._rf_reg_params = rf_reg_params if rf_reg_params is not None else {}
        self.models: dict[MarkovState, RandomForestRegressor] = {}
        self.is_fitted_: bool = False
        self._feature_idx: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        *,
        values: np.ndarray,  # (N, F)
        time_ids: np.ndarray,  # (N,)
        entity_ids: np.ndarray,  # (N,)
        markov_state: np.ndarray,  # (N,) — current markov state
        fatalities_target: np.ndarray,  # (N,) — target fatalities column
        feature_idx: np.ndarray,  # column indices into ``values``
    ) -> "MarkovFatalityModel":
        """Fit one regressor per target state (ESC, WAR only).

        The target is the fatality count ``step`` months ahead. Training
        samples are filtered to those whose ``target_state`` (the Markov
        state of the target month) equals ESC or WAR — the original
        implementation trains regressors only for these escalation states.
        """
        # --- Sort by (entity, time).
        order = np.lexsort((time_ids, entity_ids))
        values = values[order]
        time_ids = time_ids[order]
        entity_ids = entity_ids[order]
        markov_state = markov_state[order]
        fatalities_target = np.asarray(fatalities_target, dtype=np.float64)[order]

        # --- Compute target state by shifting markov_state by -step.
        target_time = time_ids + self.step
        lookup: dict[tuple[int, int], int] = {}
        for i, (e, t) in enumerate(zip(entity_ids.tolist(), time_ids.tolist())):
            lookup[(e, t)] = i

        target_state = np.empty(len(values), dtype=object)
        target_state[:] = None
        target_fatalities = np.full(len(values), np.nan, dtype=np.float64)
        for i, (e, t) in enumerate(zip(entity_ids.tolist(), time_ids.tolist())):
            tgt_row = lookup.get((e, t + self.step))
            if tgt_row is not None:
                target_state[i] = markov_state[tgt_row]
                target_fatalities[i] = fatalities_target[tgt_row]

        # --- Filter to training window.
        train_mask = (target_time >= self.train_start) & (target_time <= self.train_end)
        has_target = np.array(
            [s is not None for s in target_state], dtype=bool
        ) & ~np.isnan(target_fatalities)
        valid_mask = train_mask & has_target

        self._feature_idx = np.asarray(feature_idx, dtype=np.int64)
        X_all = values[:, self._feature_idx]

        # --- Train one regressor per escalation state.
        for state in [MarkovState.ESC, MarkovState.WAR]:
            state_mask = np.array(
                [s == state for s in target_state], dtype=bool
            )
            sub_mask = valid_mask & state_mask
            if sub_mask.sum() == 0:
                logger.warning(
                    "MarkovFatalityModel(step=%d): no training samples for "
                    "target state %s; fitting a dummy regressor.",
                    self.step, state,
                )
                rf = RandomForestRegressor(
                    n_estimators=1,
                    random_state=self._random_state,
                    n_jobs=self._n_jobs,
                )
                dummy_X = np.zeros((1, len(self._feature_idx)), dtype=np.float32)
                dummy_y = np.zeros(1, dtype=np.float32)
                rf.fit(dummy_X, dummy_y)
                self.models[state] = rf
                continue

            X_train = X_all[sub_mask]
            y_train = target_fatalities[sub_mask]
            rf = RandomForestRegressor(
                random_state=self._random_state,
                n_jobs=self._n_jobs,
                **self._rf_reg_params,
            )
            rf.fit(X_train, y_train)
            self.models[state] = rf

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        *,
        values: np.ndarray,  # (N, F)
        start_state: MarkovState,
    ) -> np.ndarray:
        """Predict fatalities given the target month's state.

        Args:
            values: ``(N, F)`` feature matrix.
            start_state: Markov state of the *target* month (must be ESC or WAR).

        Returns:
            ``(N,)`` array of predicted fatalities.
        """
        _check_is_fitted(self, "is_fitted_")
        model = self.models[start_state]
        X = values[:, self._feature_idx]
        return model.predict(X)


# ----------------------------------------------------------------------
# Main MarkovModel — darts SKLearnModel subclass
# ----------------------------------------------------------------------


class MarkovModel(SKLearnModel):
    """A Markov prediction model for forecasting fatalities.

    Subclasses :class:`SKLearnModel` so it plugs into the darts model
    registry and inherits save/load. The Markov logic (state classification
    + escalation regression + transition matrix) is owned by this class —
    the inherited ``SKLearnModel.fit`` / ``SKLearnModel.predict`` are NOT
    used.

    Supports multivariate forecasting: a separate fatality regressor is
    trained per target column. The Markov state classifier is shared across
    targets (it is computed from the single ``markov_target`` column).

    The model accepts Darts :class:`TimeSeries` (one per entity) for both
    ``series`` (targets) and ``past_covariates`` (features). Internally, the
    series are flattened into a ``(N, F)`` numpy matrix with parallel index
    arrays (``time_ids``, ``entity_ids``) — no pandas dependency.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        steps: Union[int, List[int], range],
        targets: List[str],
        markov_target: str,
        state_features: Optional[List[str]] = None,
        fatalities_features: Optional[List[str]] = None,
        markov_method: Literal["direct", "transition"] = "direct",
        regression_method: Literal["single", "multi"] = "single",
        markov_threshold: int = 0,
        random_state: int = 42,
        n_jobs: int = -1,
        rf_class_params: Optional[Dict[str, Any]] = None,
        rf_reg_params: Optional[Dict[str, Any]] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        # SKLearnModel plumbing (kept for darts interface compliance):
        lags: Optional[int] = 1,
        lags_past_covariates: Optional[int] = 1,
        add_encoders: Optional[dict] = None,
        use_static_covariates: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the Markov model.

        Args:
            steps: Step(s) ahead to fit the model for. ``int``, ``list[int]``,
                or ``range``. All values must be positive integers.
            targets: List of target column names (multivariate forecasting
                supported — one fatality regressor is fitted per target).
            markov_target: Name of the column to compute Markov states from
                (typically a fatality column). Must be present in the target
                or past-covariate components.
            state_features: Optional list of feature column names to use for
                the state classifier. When ``None``, all feature + target
                columns are used.
            fatalities_features: Optional list of feature column names to use
                for the fatality regressor. When ``None``, all feature +
                target columns are used.
            markov_method: ``"direct"`` (fit one state classifier per step) or
                ``"transition"`` (fit a single step-1 classifier and apply
                the transition matrix K times for step K).
            regression_method: ``"single"`` (fit one regressor for step 1 and
                reuse for all steps) or ``"multi"`` (fit one regressor per
                step).
            markov_threshold: Threshold for the Markov state computation.
                Defaults to 0. Non-zero values emit a warning.
            random_state: Random seed.
            n_jobs: Parallelism for the sklearn estimators.
            rf_class_params: Optional kwargs for ``RandomForestClassifier``.
            rf_reg_params: Optional kwargs for ``RandomForestRegressor``.
            output_chunk_length: Darts interface plumbing — number of steps
                predicted per ``predict`` call. Defaults to ``max(steps)``.
            output_chunk_shift: Darts interface plumbing.
            lags: Darts interface plumbing — passed to ``SKLearnModel.__init__``.
            lags_past_covariates: Darts interface plumbing.
            add_encoders: Darts interface plumbing.
            use_static_covariates: Darts interface plumbing.
        """
        # --- Validate inputs.
        self._verify_class_input_data(markov_method, regression_method)

        if not isinstance(targets, list):
            raise ValueError("Dependent variable must be a list")
        if len(targets) == 0:
            raise ValueError("targets must contain at least one column name")
        self._targets = list(targets)
        self._markov_target = markov_target

        # --- Set model parameters.
        self._steps = self._get_list_of_steps(steps)
        self._markov_method = markov_method
        self._regression_method = regression_method
        self._markov_threshold = markov_threshold
        self._state_features_cfg = (
            list(state_features) if state_features is not None else None
        )
        self._fatalities_features_cfg = (
            list(fatalities_features) if fatalities_features is not None else None
        )
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._verbose = kwargs.pop("verbose", True)

        # --- Random Forest sub-model parameters (Ranger-compatible defaults).
        self._rf_class_params: dict[str, Any] = {
            "n_estimators": 500,
        }
        self._rf_reg_params: dict[str, Any] = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
        }
        if rf_class_params:
            self._rf_class_params.update(rf_class_params)
        if rf_reg_params:
            self._rf_reg_params.update(rf_reg_params)

        # --- Markov states.
        self._markov_states: list[MarkovState] = list(MarkovState)

        # --- Storage for fitted sub-models.
        self._state_models: dict[int, MarkovStateModel] = {}
        self._fatality_models: dict[int, MarkovFatalityModel] = {}
        # Per-target fatality model storage (multivariate support).
        # _fatality_models is kept for backward-compat (single-target case);
        # _fatality_models_per_target[target][step] = MarkovFatalityModel.
        self._fatality_models_per_target: dict[str, dict[int, MarkovFatalityModel]] = {}

        self.is_fitted_: bool = False

        # --- Internal schema (filled by fit).
        self._time_id: str = "month_id"
        self._entity_id: str = "country_id"
        self._feature_columns: list[str] = []
        self._target_columns: list[str] = []
        self._all_columns: list[str] = []
        self._state_feature_idx: np.ndarray = np.array([], dtype=np.int64)
        self._fatalities_feature_idx: np.ndarray = np.array([], dtype=np.int64)
        self._markov_target_idx: int = -1
        self._target_indices: dict[str, int] = {}
        # Train window (filled by fit):
        self._train_start: int = 0
        self._train_end: int = 0
        # Caches for predict:
        self._data_values: Optional[np.ndarray] = None
        self._data_time_ids: Optional[np.ndarray] = None
        self._data_entity_ids: Optional[np.ndarray] = None
        self._data_markov_state: Optional[np.ndarray] = None

        # --- Determine output_chunk_length if not provided.
        max_step = max(self._steps)
        if output_chunk_length <= 1:
            output_chunk_length = max_step

        # --- Initialise the SKLearnModel parent with a placeholder sklearn
        # --- estimator (the actual sub-models live on this instance; the
        # --- placeholder is required by SKLearnModel.__init__ but is never
        # --- used for predictions — MarkovModel overrides fit/predict).
        placeholder = RandomForestRegressor(
            n_estimators=1, random_state=random_state, n_jobs=n_jobs
        )
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            model=placeholder,
            multi_models=False,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
            **kwargs,
        )

    # ==================================================================
    # Public darts interface
    # ==================================================================

    @property
    def input_chunk_length(self) -> int:  # type: ignore[override]
        """Darts plumbing — Markov does not use chunks; return 1."""
        return 1

    @property
    def output_chunk_length(self) -> int:  # type: ignore[override]
        """Darts plumbing — return ``max(steps)``."""
        return max(self._steps) if self._steps else 1

    # ------------------------------------------------------------------
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        sample_weight: Any = None,
        stride: int = 1,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> "MarkovModel":
        """Fit the Markov model.

        Accepts the standard darts ``fit`` signature so the model can be
        driven by :class:`DartsForecaster`. Torch-specific kwargs
        (``val_series``, ``val_past_covariates``, ``dataloader_kwargs``) are
        accepted and ignored — Markov does not use a validation set or
        gradient-based training.

        Args:
            series: Target ``TimeSeries`` (one per entity). Each series must
                carry the target columns; the ``markov_target`` column may
                live here or in ``past_covariates``.
            past_covariates: Feature ``TimeSeries`` (one per entity). May be
                ``None`` if all features are already in ``series``.
            future_covariates: Ignored (Markov does not use future covariates).
            max_samples_per_ts: Ignored.
            n_jobs_multioutput_wrapper: Ignored.
            sample_weight: Ignored.
            stride: Ignored.
            verbose: Ignored.
            **kwargs: ``val_series``, ``val_past_covariates``,
                ``dataloader_kwargs`` are accepted and ignored.

        Returns:
            ``self`` (fitted).
        """
        # Extract torch-specific kwargs and ignore them.
        kwargs.pop("val_series", None)
        kwargs.pop("val_past_covariates", None)
        kwargs.pop("dataloader_kwargs", None)

        # --- Flatten the darts TimeSeries to (N, F) numpy + index arrays.
        flat = _flatten_timeseries_list(series, past_covariates)
        # flat: dict with keys 'values', 'time_ids', 'entity_ids',
        # 'columns' (list of column names in order).

        self._all_columns = list(flat["columns"])
        self._feature_columns = [
            c for c in self._all_columns if c not in self._targets
        ]
        self._target_columns = list(self._targets)

        # --- Resolve column indices.
        if self._markov_target not in self._all_columns:
            raise ValueError(
                f"Markov target column '{self._markov_target}' not found in "
                f"input columns: {self._all_columns}"
            )
        self._markov_target_idx = self._all_columns.index(self._markov_target)
        self._target_indices = {
            t: self._all_columns.index(t) for t in self._targets
        }

        # Determine state/fatalities feature indices.
        state_feat_names = (
            self._state_features_cfg
            if self._state_features_cfg is not None
            else self._all_columns
        )
        fat_feat_names = (
            self._fatalities_features_cfg
            if self._fatalities_features_cfg is not None
            else self._all_columns
        )
        missing_state = [f for f in state_feat_names if f not in self._all_columns]
        if missing_state:
            raise ValueError(
                f"State features {missing_state} not found in input columns."
            )
        missing_fat = [f for f in fat_feat_names if f not in self._all_columns]
        if missing_fat:
            raise ValueError(
                f"Fatalities features {missing_fat} not found in input columns."
            )
        self._state_feature_idx = np.array(
            [self._all_columns.index(f) for f in state_feat_names], dtype=np.int64
        )
        self._fatalities_feature_idx = np.array(
            [self._all_columns.index(f) for f in fat_feat_names], dtype=np.int64
        )

        # --- Process data: log1p the targets, compute Markov states.
        values = flat["values"].astype(np.float32, copy=True)
        time_ids = flat["time_ids"].astype(np.int64, copy=False)
        entity_ids = flat["entity_ids"].astype(np.int64, copy=False)

        # Fill missing (entity, time) combinations with 0 — mirrors the
        # original ``_process_data`` which extends the index to the cartesian
        # product of (months × entities existing in the last month).
        values, time_ids, entity_ids = _fill_missing_combinations(
            values, time_ids, entity_ids
        )

        # Apply log1p to each target column (mirrors original behaviour).
        for tgt_name, tgt_idx in self._target_indices.items():
            values[:, tgt_idx] = np.log1p(np.maximum(values[:, tgt_idx], 0.0))

        # Determine train window from the time_ids present in the input
        # series. The darts forecaster already filters to the train partition
        # before calling fit, so the min/max time_ids here ARE the train
        # window boundaries.
        self._train_start = int(time_ids.min())
        self._train_end = int(time_ids.max())

        # Compute Markov states from markov_target.
        markov_state = self._add_markov_states(
            values, time_ids, entity_ids, self._markov_target_idx
        )

        # --- Cache the processed data for predict().
        self._data_values = values
        self._data_time_ids = time_ids
        self._data_entity_ids = entity_ids
        self._data_markov_state = markov_state

        # --- Determine which steps to fit.
        if self._markov_method == "direct":
            markov_steps = self._steps
        elif self._markov_method == "transition":
            markov_steps = [1]
        else:
            raise ValueError(
                f"Invalid markov_method: {self._markov_method}. "
                "Expected 'direct' or 'transition'."
            )

        if self._regression_method == "single":
            regression_steps = [1]
        elif self._regression_method == "multi":
            regression_steps = self._steps
        else:
            raise ValueError(
                f"Invalid regression_method: {self._regression_method}. "
                "Expected 'single' or 'multi'."
            )

        logger.info(
            "Fitting Markov Model using %s method and %s regression:",
            self._markov_method, self._regression_method,
        )

        # --- Fit state models.
        for step in markov_steps:
            sm = MarkovStateModel(
                step=step,
                train_start=self._train_start,
                train_end=self._train_end,
                rf_class_params=self._rf_class_params,
                random_state=self._random_state,
                n_jobs=self._n_jobs,
            )
            sm.fit(
                values=values,
                time_ids=time_ids,
                entity_ids=entity_ids,
                markov_state=markov_state,
                markov_target=values[:, self._markov_target_idx],
                feature_idx=self._state_feature_idx,
            )
            self._state_models[step] = sm

        # --- Fit fatality models (per target, per step).
        for target_name in self._targets:
            tgt_idx = self._target_indices[target_name]
            self._fatality_models_per_target[target_name] = {}
            for step in regression_steps:
                fm = MarkovFatalityModel(
                    step=step,
                    train_start=self._train_start,
                    train_end=self._train_end,
                    rf_reg_params=self._rf_reg_params,
                    random_state=self._random_state,
                    n_jobs=self._n_jobs,
                )
                fm.fit(
                    values=values,
                    time_ids=time_ids,
                    entity_ids=entity_ids,
                    markov_state=markov_state,
                    fatalities_target=values[:, tgt_idx],
                    feature_idx=self._fatalities_feature_idx,
                )
                self._fatality_models_per_target[target_name][step] = fm
                # Backward-compat: _fatality_models[step] = the first
                # target's model (used by callers that pre-date multivariate
                # support).
                if target_name == self._targets[0]:
                    self._fatality_models[step] = fm

        logger.info("Finished fitting Markov model.")
        self.is_fitted_ = True
        # SKLearnModel plumbing — set the internal fit flag so darts'
        # save/load and predict checks pass.
        self._fit_called = True
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: Optional[bool] = None,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecast ``n`` steps ahead from the most recent input month.

        The Markov model predicts each step in ``self._steps`` independently.
        ``n`` is interpreted as the number of steps to forecast (capped at
        ``max(self._steps)``). The output ``TimeSeries`` carries one
        component per target column.

        Args:
            n: Number of steps to forecast. Capped at ``max(self._steps)``.
            series: Optional override of the input ``TimeSeries``. When
                ``None``, the cached training data is used (the forecast is
                issued from the last observed month).
            past_covariates: Optional override of the feature ``TimeSeries``.
            num_samples: Ignored (Markov is deterministic).
            verbose: Ignored.
            **kwargs: Accepted and ignored (darts interface compliance).
        """
        _check_is_fitted(self, "is_fitted_")

        # If the caller passed new series, use them; otherwise use the
        # cached training data.
        if series is not None or past_covariates is not None:
            flat = _flatten_timeseries_list(series, past_covariates)
            values = flat["values"].astype(np.float32, copy=True)
            time_ids = flat["time_ids"].astype(np.int64, copy=False)
            entity_ids = flat["entity_ids"].astype(np.int64, copy=False)
            values, time_ids, entity_ids = _fill_missing_combinations(
                values, time_ids, entity_ids
            )
            for tgt_name, tgt_idx in self._target_indices.items():
                values[:, tgt_idx] = np.log1p(np.maximum(values[:, tgt_idx], 0.0))
            markov_state = self._add_markov_states(
                values, time_ids, entity_ids, self._markov_target_idx
            )
        else:
            values = self._data_values
            time_ids = self._data_time_ids
            entity_ids = self._data_entity_ids
            markov_state = self._data_markov_state

        # Determine the "current" month (the most recent observed time id).
        current_time = int(time_ids.max())
        # Filter to rows for the current month.
        cur_mask = time_ids == current_time
        cur_values = values[cur_mask]
        cur_entity_ids = entity_ids[cur_mask]
        cur_markov_state = markov_state[cur_mask]

        # Determine which steps to predict.
        steps_to_predict = [s for s in self._steps if s <= n]
        if not steps_to_predict:
            steps_to_predict = [min(n, max(self._steps))]

        # Predict per step.
        per_step_preds: dict[int, np.ndarray] = {}  # step -> (n_entities, n_targets)
        for step in steps_to_predict:
            step_preds = self._predict_by_step(
                step=step,
                cur_values=cur_values,
                cur_markov_state=cur_markov_state,
            )
            per_step_preds[step] = step_preds  # (n_entities, n_targets)

        # Build a list of TimeSeries — one per entity.
        # Each TimeSeries has shape (n_steps, n_targets) with integer time
        # index starting at current_time + 1.
        n_entities = cur_values.shape[0]
        out: list[TimeSeries] = []
        for e_idx in range(n_entities):
            entity_id = int(cur_entity_ids[e_idx])
            # (n_steps, n_targets) — gather per step.
            step_arr = np.stack(
                [per_step_preds[s][e_idx] for s in steps_to_predict], axis=0
            )
            # Inverse log transform: expm1.
            step_arr = np.expm1(step_arr)
            # Clip negatives (fatalities can't be negative).
            step_arr = np.maximum(step_arr, 0.0)
            step_arr = step_arr.astype(np.float32)

            time_idx = np.array(
                [current_time + s for s in steps_to_predict], dtype=np.int64
            )
            import pandas as pd
            ts = TimeSeries.from_times_and_values(
                times=pd.Index(time_idx),
                values=step_arr,
                columns=self._targets,
                static_covariates=pd.DataFrame(
                    {self._entity_id: [float(entity_id)]}
                ),
                freq=1,
            )
            out.append(ts)

        # If a single TimeSeries was passed in, return a single TimeSeries.
        if len(out) == 1:
            return out[0]
        return out

    # ------------------------------------------------------------------
    def _predict_by_step(
        self,
        *,
        step: int,
        cur_values: np.ndarray,  # (n_entities, F)
        cur_markov_state: np.ndarray,  # (n_entities,) object
    ) -> np.ndarray:
        """Predict fatalities for each entity at the given step.

        Returns:
            ``(n_entities, n_targets)`` array of (still log-space) predictions.
        """
        # --- 1) Predict Markov state probabilities for the target month.
        if self._markov_method == "transition":
            state_model = self._state_models[1]
        else:
            state_model = self._state_models[step]

        # For each starting state, get probability of each target state.
        # state_probabilities[start_state] = (n_entities, n_classes)
        state_probabilities: dict[MarkovState, np.ndarray] = {}
        for start_state in self._markov_states:
            probs = state_model.predict(
                values=cur_values, start_state=start_state
            )
            state_probabilities[start_state] = probs

        # --- 2) Build a (n_entities, n_states, n_states) transition matrix.
        n_states = len(self._markov_states)
        # Map class labels to state indices.
        # Each classifier's classes_ may differ — build a canonical ordering.
        # For each (start_state, target_state), extract the probability column.
        # If a target_state is not in classes_, the probability is 0.
        P = np.zeros((cur_values.shape[0], n_states, n_states), dtype=np.float64)
        for start_idx, start_state in enumerate(self._markov_states):
            probs = state_probabilities[start_state]
            classes = state_model.models[start_state].classes_
            for tgt_idx, tgt_state in enumerate(self._markov_states):
                # Find tgt_state in classes (classes may be strings or enums).
                tgt_str = tgt_state.value
                if tgt_str in classes:
                    col = list(classes).index(tgt_str)
                    P[:, tgt_idx, start_idx] = probs[:, col]
                elif tgt_state in classes:
                    col = list(classes).index(tgt_state)
                    P[:, tgt_idx, start_idx] = probs[:, col]
                else:
                    P[:, tgt_idx, start_idx] = 0.0

        # --- 3) Apply transition matrix power if using transition method.
        if self._markov_method == "transition":
            P = self._matrix_power(P, step)

        # --- 4) Predict fatalities for ESC and WAR states.
        # For each target column, retrieve the right fatality model.
        n_entities = cur_values.shape[0]
        n_targets = len(self._targets)
        fatalities_per_state: dict[MarkovState, np.ndarray] = {}
        # (n_entities, n_targets) per state — only ESC and WAR are non-zero.
        for state in [MarkovState.ESC, MarkovState.WAR]:
            fat_arr = np.zeros((n_entities, n_targets), dtype=np.float64)
            for t_idx, target_name in enumerate(self._targets):
                if self._regression_method == "multi":
                    fm = self._fatality_models_per_target[target_name][step]
                else:
                    fm = self._fatality_models_per_target[target_name][1]
                preds = fm.predict(values=cur_values, start_state=state)
                fat_arr[:, t_idx] = preds
            fatalities_per_state[state] = fat_arr

        # --- 5) Compute weighted fatalities per entity.
        # For each entity i with current state s_i:
        # weighted = P[i, ESC, s_i] * fat_esc[i] + P[i, WAR, s_i] * fat_war[i]
        # (PEACE and DESC contribute 0 fatalities.)
        esc_idx = self._markov_states.index(MarkovState.ESC)
        war_idx = self._markov_states.index(MarkovState.WAR)

        weighted = np.zeros((n_entities, n_targets), dtype=np.float64)
        for i in range(n_entities):
            current_state = _as_markov_state(cur_markov_state[i])
            if current_state is None:
                # Default to PEACE if state couldn't be computed.
                current_state = MarkovState.PEACE
            s_idx = self._markov_states.index(current_state)
            p_esc = P[i, esc_idx, s_idx]
            p_war = P[i, war_idx, s_idx]
            weighted[i] = (
                p_esc * fatalities_per_state[MarkovState.ESC][i]
                + p_war * fatalities_per_state[MarkovState.WAR][i]
            )

        return weighted

    # ==================================================================
    # Internal helpers (parity with original MarkovModel)
    # ==================================================================

    @staticmethod
    def _verify_class_input_data(
        markov_method: str,
        regression_method: str,
    ) -> None:
        """Validate ``markov_method`` and ``regression_method``."""
        valid_markov_methods = ["direct", "transition"]
        valid_regression_methods = ["single", "multi"]
        if markov_method not in valid_markov_methods:
            raise ValueError(
                f"Invalid markov_method: {markov_method}. Valid options: "
                f"{valid_markov_methods}"
            )
        if regression_method not in valid_regression_methods:
            raise ValueError(
                f"Invalid regression_method: {regression_method}. Valid "
                f"options: {valid_regression_methods}"
            )

    @staticmethod
    def _get_list_of_steps(steps: Union[int, List[int], range]) -> List[int]:
        """Format ``steps`` into a list of positive integers."""
        if isinstance(steps, range):
            steps_list = list(steps)
        elif isinstance(steps, list):
            steps_list = steps
        elif isinstance(steps, int):
            steps_list = [steps]
        else:
            raise TypeError("Steps must be an int, list of ints, or range.")

        for s in steps_list:
            if not isinstance(s, int):
                raise TypeError(
                    f"All elements in steps list must be integers. {s} is "
                    f"of type {type(s)}"
                )
        if any(s <= 0 for s in steps_list):
            raise ValueError("All steps must be positive integers.")
        if any(s > 36 for s in steps_list):
            warnings.warn(
                "Found steps higher than 36 months. This may lead to "
                "unreliable predictions.",
                UserWarning,
            )
        return steps_list

    def _add_markov_states(
        self,
        values: np.ndarray,
        time_ids: np.ndarray,
        entity_ids: np.ndarray,
        target_idx: int,
    ) -> np.ndarray:
        """Compute the Markov state for each row.

        Adds no column to ``values`` — returns a parallel ``(N,)`` object
        array of ``MarkovState`` (or ``None`` for the first step of each
        entity, where the t-1 fatality is unknown).
        """
        # Sort by (entity, time) so we can shift within entity groups.
        order = np.lexsort((time_ids, entity_ids))
        sorted_vals = values[order]
        sorted_time = time_ids[order]
        sorted_entity = entity_ids[order]

        target_col = sorted_vals[:, target_idx].astype(np.float64)
        # t-1 within entity: shift by 1.
        target_t_min_1 = np.empty_like(target_col)
        target_t_min_1[:] = np.nan
        # Within each entity group (which is now contiguous), shift by 1.
        # Find group boundaries.
        if len(sorted_entity) > 0:
            # Use np.diff to find group change points.
            change = np.concatenate(
                ([True], sorted_entity[1:] != sorted_entity[:-1])
            )
            group_starts = np.where(change)[0]
            for gs in group_starts:
                # First row of group has no t-1; subsequent rows shift down.
                target_t_min_1[gs] = np.nan  # no t-1
                if gs + 1 < len(target_t_min_1):
                    target_t_min_1[gs + 1:] = target_col[gs:-1]

        # Compute Markov states.
        sorted_states = np.empty(len(sorted_vals), dtype=object)
        sorted_states[:] = None
        threshold = self._markov_threshold
        if threshold != 0:
            warnings.warn(
                f"Non-zero threshold of {threshold} may lead to unexpected "
                "Markov state assignments. Please confirm that this is "
                "intended.",
                UserWarning,
            )

        for i in range(len(sorted_vals)):
            t = target_col[i]
            t_min_1 = target_t_min_1[i]
            if np.isnan(t) or np.isnan(t_min_1):
                sorted_states[i] = None
                continue
            if t <= threshold:
                if t_min_1 <= threshold:
                    sorted_states[i] = MarkovState.PEACE
                else:
                    sorted_states[i] = MarkovState.DESC
            else:
                if t_min_1 <= threshold:
                    sorted_states[i] = MarkovState.ESC
                else:
                    sorted_states[i] = MarkovState.WAR

        # Unsort: restore the original order.
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        return sorted_states[inverse_order]

    @staticmethod
    def _matrix_power(transition_matrix: np.ndarray, K: int) -> np.ndarray:
        """Compute the K-th power of a batch of transition matrices.

        Args:
            transition_matrix: ``(n_samples, n_states, n_states)``.
            K: Power to raise the matrices to.

        Returns:
            Same shape as input.
        """
        if K <= 1:
            return transition_matrix.copy()
        result = transition_matrix.copy()
        for _ in range(K - 1):
            result = np.einsum("nij,njk->nik", result, transition_matrix)
        return result

    # ==================================================================
    # Darts plumbing — required overrides so the parent class doesn't
    # attempt to use the placeholder sklearn estimator for prediction.
    # ==================================================================

    @property
    def _model_encoder_settings(self) -> Any:  # type: ignore[override]
        """Return None — Markov does not use darts' lag-based encoder."""
        return None

    @property
    def supports_multivariate(self) -> bool:  # type: ignore[override]
        """Markov supports multivariate forecasting (multiple targets)."""
        return True

    @property
    def supports_transferable_series_prediction(self) -> bool:  # type: ignore[override]
        return False

    @property
    def min_train_samples(self) -> int:  # type: ignore[override]
        return 3

    @property
    def extreme_lags(self) -> tuple:  # type: ignore[override]
        # (lags, lags_past_cov, lags_future_cov, lags_past_cov_strict)
        return (1, None, None, 0, 1, None, None, None)

    @property
    def _target_window_lengths(self) -> tuple:  # type: ignore[override]
        return (1, 1)


# ----------------------------------------------------------------------
# Numpy helpers — flat representation for the Markov logic
# ----------------------------------------------------------------------


def _flatten_timeseries_list(
    series: Union[TimeSeries, Sequence[TimeSeries], None],
    past_covariates: Union[TimeSeries, Sequence[TimeSeries], None],
) -> dict:
    """Flatten a list of Darts ``TimeSeries`` to a single ``(N, F)`` matrix.

    Concatenates the target and past-covariate components column-wise (target
    columns first, then feature columns), and stacks entities row-wise. The
    output is a dict with:

    * ``values``: ``(N, F)`` float32 array (features + targets).
    * ``time_ids``: ``(N,)`` int64 array.
    * ``entity_ids``: ``(N,)`` int64 array.
    * ``columns``: list of column names in order (targets + features).

    The row order is entity-major then time-major within entity, matching
    the original pandas MultiIndex convention.
    """
    if series is None:
        raise ValueError("MarkovModel.fit requires at least one target series.")

    # Normalise to lists.
    if isinstance(series, TimeSeries):
        series_list = [series]
    else:
        series_list = list(series)
    if past_covariates is None:
        cov_list: list[TimeSeries] = []
    elif isinstance(past_covariates, TimeSeries):
        cov_list = [past_covariates]
    else:
        cov_list = list(past_covariates)

    if len(series_list) == 0:
        raise ValueError("MarkovModel.fit requires at least one target series.")
    if cov_list and len(cov_list) != len(series_list):
        raise ValueError(
            f"Length mismatch: {len(series_list)} target series but "
            f"{len(cov_list)} past_covariate series."
        )

    # Determine column order: targets first, then features.
    target_columns = list(series_list[0].components)
    feature_columns: list[str] = []
    if cov_list:
        feature_columns = list(cov_list[0].components)
        # Remove duplicates (a column may appear in both targets and features).
        feature_columns = [
            c for c in feature_columns if c not in target_columns
        ]
    all_columns = target_columns + feature_columns

    values_list: list[np.ndarray] = []
    time_ids_list: list[np.ndarray] = []
    entity_ids_list: list[np.ndarray] = []

    for i, ts in enumerate(series_list):
        # Time index (integer).
        time_idx = np.asarray(ts.time_index.values, dtype=np.int64)
        # Target values: shape (T, n_targets) or (T, n_targets, n_samples).
        tgt_vals = ts.values(copy=False)  # (T, n_targets) or (T, n_targets, 1)
        if tgt_vals.ndim == 3:
            tgt_vals = tgt_vals[:, :, 0]
        tgt_vals = tgt_vals.astype(np.float32, copy=False)

        if cov_list:
            cov_ts = cov_list[i]
            # Align time index — the covariate series may have a different
            # time range. We use the intersection.
            cov_time = np.asarray(cov_ts.time_index.values, dtype=np.int64)
            common_time = np.intersect1d(time_idx, cov_time)
            if len(common_time) == 0:
                raise ValueError(
                    f"Entity {i}: no overlapping time ids between target "
                    "and past_covariate series."
                )
            # Subset both series to the common time range.
            tgt_mask = np.isin(time_idx, common_time)
            cov_mask = np.isin(cov_time, common_time)
            # Sort common_time to preserve order.
            common_time.sort()
            # Reorder masks by sorted common_time.
            tgt_lookup = {t: k for k, t in enumerate(time_idx)}
            cov_lookup = {t: k for k, t in enumerate(cov_time)}
            tgt_indices = [tgt_lookup[t] for t in common_time.tolist()]
            cov_indices = [cov_lookup[t] for t in common_time.tolist()]

            tgt_vals = tgt_vals[tgt_indices]
            cov_vals = cov_ts.values(copy=False)
            if cov_vals.ndim == 3:
                cov_vals = cov_vals[:, :, 0]
            cov_vals = cov_vals[cov_indices].astype(np.float32, copy=False)
            # Drop duplicate columns (columns that appear in both).
            keep_cov_idx = [
                k for k, c in enumerate(cov_ts.components)
                if c not in target_columns
            ]
            if keep_cov_idx:
                cov_vals = cov_vals[:, keep_cov_idx]
            else:
                cov_vals = np.zeros((len(common_time), 0), dtype=np.float32)
            vals = np.concatenate([tgt_vals, cov_vals], axis=1)
            time_idx_local = common_time
        else:
            vals = tgt_vals
            time_idx_local = time_idx

        # Entity id from static covariates (or fallback to index i).
        entity_id = i + 1
        if ts.static_covariates is not None and not ts.static_covariates.empty:
            sc = ts.static_covariates
            # Find an "entity" column (country_id, priogrid_id, etc.).
            for col in ("country_id", "priogrid_id", "entity_id"):
                if col in sc.columns:
                    entity_id = int(sc[col].iloc[0])
                    break

        values_list.append(vals)
        time_ids_list.append(time_idx_local)
        entity_ids_list.append(np.full(len(time_idx_local), entity_id, dtype=np.int64))

    values = np.concatenate(values_list, axis=0).astype(np.float32, copy=False)
    time_ids = np.concatenate(time_ids_list, axis=0)
    entity_ids = np.concatenate(entity_ids_list, axis=0)

    return {
        "values": values,
        "time_ids": time_ids,
        "entity_ids": entity_ids,
        "columns": all_columns,
    }


def _fill_missing_combinations(
    values: np.ndarray,
    time_ids: np.ndarray,
    entity_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill missing (time, entity) combinations with 0.

    Mirrors the original ``_process_data`` method: countries that appear in
    the last month are kept; missing (time, entity) rows for those countries
    are inserted with zero values for all columns.
    """
    if len(values) == 0:
        return values, time_ids, entity_ids

    # Determine the entities present in the last month.
    max_time = int(time_ids.max())
    last_month_mask = time_ids == max_time
    existing_entities = np.unique(entity_ids[last_month_mask])

    # Filter to rows whose entity is in the existing_entities set.
    keep_mask = np.isin(entity_ids, existing_entities)
    values = values[keep_mask]
    time_ids = time_ids[keep_mask]
    entity_ids = entity_ids[keep_mask]

    # Build the cartesian product of (all_months × existing_entities).
    all_months = np.unique(time_ids)
    # Compute the set of existing (time, entity) pairs.
    existing_pairs = set(zip(time_ids.tolist(), entity_ids.tolist()))
    # Find missing pairs.
    missing_pairs: list[tuple[int, int]] = []
    for t in all_months:
        for e in existing_entities:
            if (int(t), int(e)) not in existing_pairs:
                missing_pairs.append((int(t), int(e)))

    if not missing_pairs:
        return values, time_ids, entity_ids

    # Append zero-filled rows for the missing pairs.
    missing_time = np.array([p[0] for p in missing_pairs], dtype=np.int64)
    missing_entity = np.array([p[1] for p in missing_pairs], dtype=np.int64)
    missing_values = np.zeros(
        (len(missing_pairs), values.shape[1]), dtype=values.dtype
    )

    values = np.concatenate([values, missing_values], axis=0)
    time_ids = np.concatenate([time_ids, missing_time], axis=0)
    entity_ids = np.concatenate([entity_ids, missing_entity], axis=0)

    # Sort by (entity, time) for deterministic ordering.
    order = np.lexsort((time_ids, entity_ids))
    return values[order], time_ids[order], entity_ids[order]
