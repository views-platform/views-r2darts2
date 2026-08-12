"""MarkovModel — a Darts-compatible Markov state fatality forecasting model.

Wraps the original pandas-based MarkovModel into a Darts
:class:`GlobalForecastingModel` subclass so it can be used through the
standard :class:`DartsForecaster` + :class:`ModelCatalog` pipeline.

The model:
    1. Computes Markov states (PEACE/DESC/ESC/WAR) from fatality counts.
    2. Fits per-state RandomForestClassifier models to predict state transitions.
    3. Fits per-state RandomForestRegressor models to predict fatalities.
    4. At predict time: predicts state probabilities → conditional fatalities →
       weighted average.

Pandas-free: all data operations use numpy + the ViewsDataset xarray/zarr
infrastructure. No pandas DataFrames are constructed.

Google Python Style.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

logger = logging.getLogger(__name__)


class MarkovState(str, Enum):
    """Markov states based on fatality count transitions."""

    PEACE = "peace"
    DESC = "desc"
    ESC = "esc"
    WAR = "war"


class MarkovModel(GlobalForecastingModel):
    """Darts-compatible Markov state fatality forecasting model.

    A non-torch model that uses sklearn RandomForest classifiers and regressors
    to predict fatalities based on Markov state transitions. Fits into the
    standard DartsForecaster pipeline via the ModelCatalog.

    The model supports the same ``fit(series, past_covariates, ...)`` /
    ``predict(n, series, ...)`` interface as Darts torch models, but
    internally works with numpy arrays extracted from the TimeSeries.
    """

    def __init__(
        self,
        input_chunk_length: int = 12,
        output_chunk_length: int = 36,
        *,
        markov_threshold: int = 0,
        markov_method: str = "direct",
        regression_method: str = "single",
        random_state: int = 42,
        n_jobs: int = -1,
        rf_class_params: dict[str, Any] | None = None,
        rf_reg_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MarkovModel.

        Args:
            input_chunk_length: Number of time steps in the input window.
                Used for Darts compatibility — the Markov model uses only
                the last time step of the input window.
            output_chunk_length: Number of time steps to forecast per call.
            markov_threshold: Threshold for Markov state computation.
            markov_method: ``"direct"`` or ``"transition"``.
            regression_method: ``"single"`` or ``"multi"``.
            random_state: Random seed.
            n_jobs: Number of parallel jobs for sklearn.
            rf_class_params: Extra params for RandomForestClassifier.
            rf_reg_params: Extra params for RandomForestRegressor.
        """
        super().__init__(**kwargs)
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self._markov_threshold = markov_threshold
        self._markov_method = markov_method
        self._regression_method = regression_method
        self._random_state = random_state
        self._n_jobs = n_jobs

        # Default RF params (matching Ranger defaults from R).
        self._rf_class_params = {"n_estimators": 500}
        self._rf_reg_params = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
        }
        if rf_class_params:
            self._rf_class_params.update(rf_class_params)
        if rf_reg_params:
            self._rf_reg_params.update(rf_reg_params)

        # Fitted state.
        self._state_models: dict[int, dict[MarkovState, RandomForestClassifier]] = {}
        self._fatality_models: dict[int, dict[MarkovState, RandomForestRegressor]] = {}
        self._state_models_by_comp: dict[
            str, dict[int, dict[MarkovState, RandomForestClassifier]]
        ] = {}
        self._fatality_models_by_comp: dict[
            str, dict[int, dict[MarkovState, RandomForestRegressor]]
        ] = {}
        self._markov_features: list[str] = []
        self._fatalities_features: list[str] = []
        self._target_names: list[str] = []
        self._target_name: str = ""
        self._train_start: int = 0
        self._train_end: int = 0
        self._markov_states = list(MarkovState)
        self._is_fitted = False
        self._steps: list[int] = []

    # ------------------------------------------------------------------ #
    # Darts GlobalForecastingModel interface
    # ------------------------------------------------------------------ #

    @property
    def supports_multivariate(self) -> bool:
        """Whether the model supports multivariate targets."""
        return True

    @property
    def supports_past_covariates(self) -> bool:
        """Whether the model supports past covariates."""
        return True

    @property
    def supports_future_covariates(self) -> bool:
        """Whether the model supports future covariates."""
        return False

    @property
    def supports_static_covariates(self) -> bool:
        """Whether the model supports static covariates."""
        return False

    @property
    def supports_probabilistic_prediction(self) -> bool:
        """Whether the model supports probabilistic prediction."""
        return False

    @property
    def input_chunk_length(self) -> int:
        """Input chunk length for Darts compatibility."""
        return self._input_chunk_length

    @property
    def output_chunk_length(self) -> int:
        """Output chunk length for Darts compatibility."""
        return self._output_chunk_length

    @property
    def min_input_chunk_length(self) -> int:
        """Minimum input chunk length."""
        return self._input_chunk_length

    @property
    def min_train_samples(self) -> int:
        """Minimum number of training samples required by the model."""
        return 1

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        """Target input/output window lengths used by Darts sanity checks."""
        return self._input_chunk_length, self._output_chunk_length

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int,
    ]:
        """Return lag limits in Darts' expected 7-tuple format."""
        return (
            -self._input_chunk_length,
            self._output_chunk_length - 1,
            -self._input_chunk_length,
            -1,
            None,
            None,
            0,
        )

    @property
    def _model_encoder_settings(
        self,
    ) -> tuple[
        int | None,
        int | None,
        bool,
        bool,
        list[int] | None,
        list[int] | None,
    ]:
        """Expose encoder metadata consumed by Darts' base model utilities."""
        return (
            self._input_chunk_length,
            self._output_chunk_length,
            True,
            False,
            None,
            None,
        )

    def fit(
        self,
        series: TimeSeries | Sequence[TimeSeries],
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        val_series: TimeSeries | Sequence[TimeSeries] | None = None,
        val_past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        val_future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        **kwargs: Any,
    ) -> "MarkovModel":
        """Fit the Markov model on the given series.

        Extracts numpy arrays from the Darts TimeSeries, computes Markov
        states, and fits per-state RandomForest classifiers and regressors.

        Args:
            series: Target TimeSeries (or list of per-entity TimeSeries).
            past_covariates: Feature TimeSeries (or list).
            **kwargs: Ignored (Darts compatibility).

        Returns:
            self.
        """
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            verbose=kwargs.get("verbose"),
        )

        series_list = series if isinstance(series, (list, tuple)) else [series]
        cov_list = (
            past_covariates
            if past_covariates is not None and isinstance(past_covariates, (list, tuple))
            else [past_covariates] * len(series_list)
            if past_covariates is not None
            else [None] * len(series_list)
        )

        # Extract target names from the first series's components.
        target_names = list(series_list[0].components)
        if len(target_names) == 0:
            raise ValueError("MarkovModel requires at least one target component.")
        self._target_names = target_names
        self._target_name = target_names[0]

        # Extract feature names from past covariates (if any).
        if cov_list[0] is not None:
            self._markov_features = list(cov_list[0].components)
            self._fatalities_features = list(cov_list[0].components)
        else:
            self._markov_features = [self._target_name]
            self._fatalities_features = [self._target_name]

        # Determine training time range.
        time_index = series_list[0].time_index
        self._train_start = int(time_index.min())
        self._train_end = int(time_index.max())

        # Determine steps.
        self._steps = list(range(1, self._output_chunk_length + 1))

        # Collect training data from all entities.
        all_target_vals: list[np.ndarray] = []
        all_feature_vals: list[np.ndarray] = []
        all_time_ids: list[np.ndarray] = []
        all_entity_ids: list[np.ndarray] = []

        for idx, (ts, cov) in enumerate(zip(series_list, cov_list)):
            # Extract target values as (T, C).
            target_arr = ts.all_values(copy=False)
            if target_arr.ndim == 3:
                target_arr = target_arr[:, :, 0]
            elif target_arr.ndim == 1:
                target_arr = target_arr.reshape(-1, 1)
            else:
                target_arr = np.asarray(target_arr)
            target_arr = np.asarray(target_arr, dtype=np.float32)
            if target_arr.ndim != 2:
                raise ValueError(
                    f"Unexpected target array shape {target_arr.shape}; expected 2D (T, C)."
                )
            if target_arr.shape[1] != len(self._target_names):
                raise ValueError(
                    "Target component mismatch between series; "
                    f"expected {len(self._target_names)} got {target_arr.shape[1]}."
                )
            times = np.asarray(ts.time_index.values, dtype=np.int64)

            # Extract entity id from static covariates.
            entity_id = idx  # fallback
            if ts.static_covariates is not None:
                try:
                    # Try to get entity_id from the first column.
                    cols = list(ts.static_covariates.columns)
                    if cols:
                        entity_id = int(ts.static_covariates.iloc[0, 0])
                except Exception:
                    pass

            # Extract feature values.
            if cov is not None:
                feat_arr = cov.all_values(copy=False)
                if feat_arr.ndim == 3:
                    feat_arr = feat_arr[:, :, 0]  # (T, F, 1) → (T, F)
                elif feat_arr.ndim == 1:
                    feat_arr = feat_arr.reshape(-1, 1)
                else:
                    feat_arr = feat_arr  # (T, F)
            else:
                # No covariates: use all target components as features.
                feat_arr = target_arr

            all_target_vals.append(target_arr)
            all_feature_vals.append(feat_arr)
            all_time_ids.append(times)
            all_entity_ids.append(np.full(len(times), entity_id, dtype=np.int64))

        # Concatenate all entities.
        target_flat_all = np.concatenate(all_target_vals, axis=0)
        feature_flat = np.concatenate(all_feature_vals, axis=0)
        time_flat = np.concatenate(all_time_ids)
        entity_flat = np.concatenate(all_entity_ids)

        # Fit state models and fatality models.
        markov_steps = self._steps if self._markov_method == "direct" else [1]
        regression_steps = self._steps if self._regression_method == "multi" else [1]

        logger.info(
            "MarkovModel.fit: %d entities, %d rows, %d targets, %d steps, method=%s/%s",
            len(series_list), target_flat_all.shape[0], len(self._target_names), len(self._steps),
            self._markov_method, self._regression_method,
        )

        self._state_models_by_comp = {}
        self._fatality_models_by_comp = {}
        for comp_idx, comp_name in enumerate(self._target_names):
            target_flat = target_flat_all[:, comp_idx]
            target_flat = np.log1p(np.maximum(target_flat, 0))

            markov_states = self._compute_markov_states_batch(
                target_flat, time_flat, entity_flat
            )

            self._state_models_by_comp[comp_name] = {}
            self._fatality_models_by_comp[comp_name] = {}

            for step in markov_steps:
                self._fit_state_model(
                    step,
                    feature_flat,
                    target_flat,
                    markov_states,
                    time_flat,
                    entity_flat,
                    component=comp_name,
                )

            for step in regression_steps:
                self._fit_fatality_model(
                    step,
                    feature_flat,
                    target_flat,
                    markov_states,
                    time_flat,
                    entity_flat,
                    component=comp_name,
                )

        # Backward-compatible aliases for single-target diagnostics.
        if len(self._target_names) == 1:
            only = self._target_names[0]
            self._state_models = self._state_models_by_comp[only]
            self._fatality_models = self._fatality_models_by_comp[only]

        self._is_fitted = True
        logger.info("MarkovModel.fit complete.")
        return self

    def predict(
        self,
        n: int,
        series: TimeSeries | Sequence[TimeSeries] | None = None,
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        num_samples: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> TimeSeries | Sequence[TimeSeries]:
        """Predict fatalities for the next ``n`` time steps.

        Args:
            n: Number of time steps to forecast.
            series: Input target TimeSeries (or list per entity).
            past_covariates: Feature TimeSeries (or list).
            num_samples: Must be 1 (deterministic model).
            **kwargs: Ignored.

        Returns:
            A TimeSeries (or list) containing the predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if num_samples != 1:
            raise ValueError("MarkovModel only supports num_samples=1.")

        series_list = series if isinstance(series, (list, tuple)) else [series]
        cov_list = (
            past_covariates
            if past_covariates is not None and isinstance(past_covariates, (list, tuple))
            else [past_covariates] * len(series_list)
            if past_covariates is not None
            else [None] * len(series_list)
        )

        results: list[TimeSeries] = []
        for idx, (ts, cov) in enumerate(zip(series_list, cov_list)):
            pred_ts = self._predict_single(ts, cov, n, idx)
            results.append(pred_ts)

        if isinstance(series, TimeSeries):
            return results[0]
        return results

    def _predict_single(
        self, ts: TimeSeries, cov: TimeSeries | None, n: int, entity_idx: int,
    ) -> TimeSeries:
        """Predict for a single entity."""
        import pandas as pd

        # Extract the last time step's target and features.
        target_arr = ts.all_values(copy=False)
        if target_arr.ndim == 3:
            target_last = target_arr[:, :, 0]
        elif target_arr.ndim == 1:
            target_last = target_arr.reshape(-1, 1)
        else:
            target_last = target_arr
        target_last = np.asarray(target_last, dtype=np.float32)

        # Extract features for the last time step.
        if cov is not None:
            feat_arr = cov.all_values(copy=False)
            if feat_arr.ndim == 3:
                feat_last = feat_arr[-1, :, 0]  # (F,)
            elif feat_arr.ndim == 1:
                feat_last = feat_arr
            else:
                feat_last = feat_arr[-1, :]
        else:
            feat_last = target_last[-1, :]

        # Compute current Markov state.
        # Need t-1 value — get from the second-to-last time step.
        prev_idx = -2 if target_last.shape[0] >= 2 else -1

        # Get time index for predictions.
        last_time = int(ts.time_index[-1])
        pred_times = pd.RangeIndex(
            start=last_time + 1, stop=last_time + 1 + n, step=1
        )

        # Predict for each step.
        pred_mat = np.zeros((n, len(self._target_names)), dtype=np.float32)
        feat_input = feat_last.reshape(1, -1)
        for comp_idx, comp_name in enumerate(self._target_names):
            target_log = np.log1p(max(float(target_last[-1, comp_idx]), 0.0))
            target_t_min_1_log = np.log1p(max(float(target_last[prev_idx, comp_idx]), 0.0))
            current_state = self._compute_markov_state(
                target_log, target_t_min_1_log, self._markov_threshold
            )
            for step_idx in range(n):
                step = step_idx + 1
                pred = self._predict_step(
                    step, feat_input, current_state, component=comp_name
                )
                pred_mat[step_idx, comp_idx] = pred

        # Inverse log transform.
        pred_mat = np.expm1(pred_mat)
        pred_mat = np.maximum(pred_mat, 0.0)

        # Build output TimeSeries.
        components = pd.Index([f"pred_{name}" for name in self._target_names])
        static_cov = ts.static_covariates

        return TimeSeries(
            times=pred_times,
            values=pred_mat.astype(np.float32),
            components=components,
            static_covariates=static_cov,
            copy=False,
        )

    def _predict_step(
        self,
        step: int,
        features: np.ndarray,
        current_state: MarkovState,
        component: str,
    ) -> float:
        """Predict fatalities for a single step.

        1. Get state transition probabilities from the state model.
        2. Get conditional fatality predictions from the fatality model.
        3. Compute weighted average.
        """
        # Get the state model for this step.
        model_step = step if self._markov_method == "direct" else 1
        comp_state_models = self._state_models_by_comp.get(component, {})
        if model_step not in comp_state_models:
            return 0.0

        state_models = comp_state_models[model_step]
        if current_state not in state_models:
            return 0.0

        # Predict state probabilities.
        clf = state_models[current_state]
        probs = clf.predict_proba(features)[0]  # (n_classes,)
        classes = clf.classes_

        # Map probabilities to states.
        prob_map: dict[MarkovState, float] = {}
        for cls, prob in zip(classes, probs):
            if isinstance(cls, MarkovState):
                state = cls
            elif isinstance(cls, str):
                try:
                    state = MarkovState(cls)
                except ValueError:
                    continue
            else:
                continue
            prob_map[state] = prob

        # Get fatality predictions for ESC and WAR states.
        reg_step = step if self._regression_method == "multi" else 1
        comp_fatality_models = self._fatality_models_by_comp.get(component, {})
        if reg_step not in comp_fatality_models:
            return 0.0

        fatality_models = comp_fatality_models[reg_step]
        esc_fatalities = 0.0
        war_fatalities = 0.0

        if MarkovState.ESC in fatality_models:
            esc_fatalities = float(
                fatality_models[MarkovState.ESC].predict(features)[0]
            )
        if MarkovState.WAR in fatality_models:
            war_fatalities = float(
                fatality_models[MarkovState.WAR].predict(features)[0]
            )

        # Weighted fatalities.
        p_esc = prob_map.get(MarkovState.ESC, 0.0)
        p_war = prob_map.get(MarkovState.WAR, 0.0)

        weighted = p_esc * esc_fatalities + p_war * war_fatalities
        return float(weighted)

    # ------------------------------------------------------------------ #
    # Internal fitting helpers
    # ------------------------------------------------------------------ #

    def _compute_markov_states_batch(
        self,
        target: np.ndarray,
        time_ids: np.ndarray,
        entity_ids: np.ndarray,
    ) -> np.ndarray:
        """Compute Markov states for all rows (vectorized).

        Args:
            target: (N,) log1p-transformed target values.
            time_ids: (N,) time ids.
            entity_ids: (N,) entity ids.

        Returns:
            (N,) array of MarkovState values.
        """
        threshold = np.log1p(max(self._markov_threshold, 0))

        # Sort by (entity, time) to get correct shift.
        sort_idx = np.lexsort((time_ids, entity_ids))
        sorted_target = target[sort_idx]
        sorted_entity = entity_ids[sort_idx]

        # Shift by 1 within each entity.
        target_t_min_1 = np.empty_like(sorted_target)
        target_t_min_1[0] = np.nan
        target_t_min_1[1:] = sorted_target[:-1]
        # Mask where entity changes.
        entity_changed = sorted_entity[1:] != sorted_entity[:-1]
        target_t_min_1[1:][entity_changed] = np.nan

        # Compute states.
        is_peace = (sorted_target <= threshold) & (target_t_min_1 <= threshold)
        is_desc = (sorted_target <= threshold) & (target_t_min_1 > threshold)
        is_esc = (sorted_target > threshold) & (target_t_min_1 <= threshold)
        is_war = (sorted_target > threshold) & (target_t_min_1 > threshold)

        states = np.empty(len(sorted_target), dtype=object)
        states[is_peace] = MarkovState.PEACE
        states[is_desc] = MarkovState.DESC
        states[is_esc] = MarkovState.ESC
        states[is_war] = MarkovState.WAR
        # NaN where entity changed (first row per entity).
        states[np.isnan(target_t_min_1)] = MarkovState.PEACE  # default

        # Unsort.
        unsort_idx = np.argsort(sort_idx)
        return states[unsort_idx]

    def _fit_state_model(
        self,
        step: int,
        features: np.ndarray,
        target: np.ndarray,
        markov_states: np.ndarray,
        time_ids: np.ndarray,
        entity_ids: np.ndarray,
        component: str,
    ) -> None:
        """Fit per-state RandomForestClassifier models for a given step."""
        # Create shifted target state (state at t+step).
        sort_idx = np.lexsort((time_ids, entity_ids))
        sorted_states = markov_states[sort_idx]
        sorted_time = time_ids[sort_idx]
        sorted_entity = entity_ids[sort_idx]

        # Shift states by -step within each entity.
        target_states = np.empty(len(sorted_states), dtype=object)
        target_states[:] = None
        for i in range(len(sorted_states) - step):
            if sorted_entity[i + step] == sorted_entity[i]:
                target_states[i] = sorted_states[i + step]

        # Filter to training period.
        target_time = sorted_time + step
        train_mask = (target_time >= self._train_start) & (target_time <= self._train_end)
        train_mask &= np.array([s is not None for s in target_states])

        comp_state_models = self._state_models_by_comp.setdefault(component, {})
        comp_state_models[step] = {}
        for state in self._markov_states:
            state_mask = sorted_states == state
            combined_mask = train_mask & state_mask
            if combined_mask.sum() == 0:
                continue

            X_train = features[sort_idx][combined_mask]
            y_train = np.array([target_states[i] for i in np.where(combined_mask)[0]])

            if len(np.unique(y_train)) < 2:
                # Sparse/conflict data often yields deterministic transitions
                # for a given origin state and horizon. Keep that signal.
                clf = DummyClassifier(strategy="most_frequent")
                clf.fit(X_train, y_train)
                comp_state_models[step][state] = clf
                continue

            clf = RandomForestClassifier(
                random_state=self._random_state,
                n_jobs=self._n_jobs,
                **self._rf_class_params,
            )
            clf.fit(X_train, y_train)
            comp_state_models[step][state] = clf

        logger.info(
            "MarkovModel[%s]: fitted state model for step %d (%d states)",
            component, step, len(comp_state_models[step]),
        )

    def _fit_fatality_model(
        self,
        step: int,
        features: np.ndarray,
        target: np.ndarray,
        markov_states: np.ndarray,
        time_ids: np.ndarray,
        entity_ids: np.ndarray,
        component: str,
    ) -> None:
        """Fit per-state RandomForestRegressor models for a given step."""
        sort_idx = np.lexsort((time_ids, entity_ids))
        sorted_states = markov_states[sort_idx]
        sorted_target = target[sort_idx]
        sorted_time = time_ids[sort_idx]
        sorted_entity = entity_ids[sort_idx]

        # Shift target by -step within each entity.
        fatalities_target = np.full(len(sorted_target), np.nan)
        for i in range(len(sorted_target) - step):
            if sorted_entity[i + step] == sorted_entity[i]:
                fatalities_target[i] = sorted_target[i + step]

        # Shift states by -step to get target state.
        target_states = np.empty(len(sorted_states), dtype=object)
        target_states[:] = None
        for i in range(len(sorted_states) - step):
            if sorted_entity[i + step] == sorted_entity[i]:
                target_states[i] = sorted_states[i + step]

        target_time = sorted_time + step
        train_mask = (target_time >= self._train_start) & (target_time <= self._train_end)
        train_mask &= ~np.isnan(fatalities_target)
        train_mask &= np.array([s is not None for s in target_states])

        comp_fatality_models = self._fatality_models_by_comp.setdefault(component, {})
        comp_fatality_models[step] = {}
        for state in [MarkovState.ESC, MarkovState.WAR]:
            state_mask = target_states == state
            combined_mask = train_mask & state_mask
            if combined_mask.sum() == 0:
                continue

            X_train = features[sort_idx][combined_mask]
            y_train = fatalities_target[combined_mask]

            reg = RandomForestRegressor(
                random_state=self._random_state,
                n_jobs=self._n_jobs,
                **self._rf_reg_params,
            )
            reg.fit(X_train, y_train)
            comp_fatality_models[step][state] = reg

        logger.info(
            "MarkovModel[%s]: fitted fatality model for step %d (%d states)",
            component, step, len(comp_fatality_models[step]),
        )

    @staticmethod
    def _compute_markov_state(
        target_t: float, target_t_min_1: float, threshold: float = 0,
    ) -> MarkovState:
        """Compute the Markov state from current and previous target values."""
        if target_t <= threshold:
            if target_t_min_1 <= threshold:
                return MarkovState.PEACE
            return MarkovState.DESC
        else:
            if target_t_min_1 <= threshold:
                return MarkovState.ESC
            return MarkovState.WAR

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the model via pickle."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "MarkovModel":
        """Load a model from a pickle file."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self) -> dict:
        """Support pickling."""
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        """Support unpickling."""
        self.__dict__.update(state)
