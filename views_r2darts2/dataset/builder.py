"""Streaming dataset builder — scaffold a Zarr store, then fill it batch by batch.

``ViewsDataset`` ingestion assumes the caller already holds a complete source
(a DataFrame, Parquet file, PredictionFrame, ...). Prediction batches are
different: the data is *produced* incrementally and can be far larger than RAM.
The builder pre-allocates a NaN-filled Zarr skeleton (metadata only — nothing
is materialized) and scatter-writes each batch straight to disk via
:class:`~views_r2darts2.dataset.converters.GridWriter`, so peak memory is one
batch, never the grid.

The scaffold is declared by the VIEWS ``loa`` code — the builder fixes the
time/entity dimension names (``pgm`` -> ``month_id``/``priogrid_id``, ``cm``
-> ``month_id``/``country_id``, ``pgy``/``cy`` the year variants) so callers
never touch the Zarr schema::

    with ViewsDataset.builder(
        loa="pgm",
        times=np.arange(528, 540),
        entities=priogrid_ids,
        variables={"pred_ln_sb_best": "num3"},
        sample_size=32,
        targets=["pred_ln_sb_best"],
    ) as b:
        for t, ents, values in my_prediction_batches():
            b.write_batch(times=t, entities=ents,
                          columns={"pred_ln_sb_best": values})
            del values  # it is on disk now
        ds = b.build()  # a real PGMDataset, lazy

The built dataset is indistinguishable from an ingested one — every export
(``to_predictionframe``, ``save_parquet``, ``save_zarr``, ...) works.

Semantics:
- Coordinates are declared up front (sorted, unique); batches are validated
  against the scaffold and fail loud naming the offenders.
- Never-written cells stay NaN. ``build(require_complete=True)`` fails loud
  when any cell was never written (needs ``strict=True`` or
  ``track_coverage=True``, which keep a small written-mask).
- Overwrites are last-write-wins by default (idempotent re-runs);
  ``strict=True`` raises on any duplicate ``(time, entity)`` write.
- ``path=None`` writes into a self-cleaning scratch store; an absolute
  ``path`` produces a durable store that survives ``close()``.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from views_r2darts2.dataset.base import ViewsDataset, _loa_to_class
from views_r2darts2.dataset.converters import (
    GridWriter,
    build_schema_attrs,
)
from views_r2darts2.dataset.zarr_store import ZarrStore

_FLOAT = np.float32

#: loa -> (time_id, entity_id). The spatial-only LOAs (``pg``/``c``) have no
#: time axis and cannot be scaffolded — the builder fails loud on them.
_LOA_AXES: dict[str, tuple[str, str]] = {
    "pgm": ("month_id", "priogrid_id"),
    "pgy": ("year_id", "priogrid_id"),
    "cm": ("month_id", "country_id"),
    "cy": ("year_id", "country_id"),
}

#: Store attributes that user metadata must never shadow.
_RESERVED_ATTRS = frozenset(
    {
        "is_prediction", "sample_size", "targets", "features", "pred_vars",
        "text_cols", "time_id", "entity_id", "broadcast_features",
    }
)


def _sorted_unique_int64(values: Any, kind: str) -> np.ndarray:
    """Coerce scaffold coordinates to sorted unique int64, failing loud."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{kind} must be a 1-D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{kind} must contain at least one value")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"{kind} must be integer identifiers, got dtype {arr.dtype}"
        )
    ordered = np.sort(arr)
    dup = ordered[1:] == ordered[:-1]
    if dup.any():
        raise ValueError(
            f"{kind} contains duplicate values, e.g. {int(ordered[1:][dup][0])}"
        )
    return ordered.astype("int64")


def _as_int64_1d(values: Any, kind: str) -> np.ndarray:
    """Coerce batch coordinates to a 1-D int64 array (scalars lifted)."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{kind} must be 1-D, got ndim={arr.ndim}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"{kind} must be integer identifiers, got dtype {arr.dtype}"
        )
    return arr.astype("int64")


def _locate(coords: np.ndarray, values: np.ndarray, kind: str) -> np.ndarray:
    """Positions of ``values`` in the sorted scaffold ``coords``, fail loud."""
    pos = np.searchsorted(coords, values)
    clipped = np.minimum(pos, coords.size - 1)
    bad = (pos >= coords.size) | (coords[clipped] != values)
    if bad.any():
        offenders = [int(v) for v in values[bad][:5]]
        raise ValueError(
            f"{kind} value(s) {offenders} are outside the scaffold "
            f"[{int(coords[0])}, {int(coords[-1])}]"
        )
    return pos


def _normalize_specs(variables: Any) -> dict[str, str]:
    """``{name: spec}``; a bare list infers num3 for ``pred_*``, num2 else."""
    if isinstance(variables, dict):
        specs = dict(variables)
    else:
        specs = {
            str(name): ("num3" if str(name).startswith("pred_") else "num2")
            for name in variables
        }
    for name, spec in specs.items():
        if spec not in ("num2", "num3"):
            raise ValueError(
                f"variable {name!r} has spec {spec!r}; expected 'num2' or "
                "'num3' (text columns are not supported by the builder)"
            )
    return specs


class DatasetBuilder:
    """Scaffold a VIEWS dataset on disk and stream batches into it.

    Construct via :meth:`ViewsDataset.builder` (or directly). One builder
    produces exactly one dataset: after :meth:`build` the store belongs to
    the returned dataset and the builder refuses further writes.
    """

    def __init__(
        self,
        loa: str,
        times: Sequence[int] | np.ndarray,
        entities: Sequence[int] | np.ndarray,
        variables: dict[str, str] | Sequence[str],
        *,
        sample_size: int = 1,
        targets: list[str] | None = None,
        broadcast_features: bool = False,
        metadata: dict[str, Any] | None = None,
        path: str | Path | None = None,
        strict: bool = False,
        track_coverage: bool = False,
        base_dir: str | Path | None = None,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        code = (loa or "").strip().lower()
        if code in ("pg", "c"):
            raise ValueError(
                f"loa {loa!r} is spatial-only and has no time axis; the "
                f"builder needs a full loa, one of {sorted(_LOA_AXES)}"
            )
        if code not in _LOA_AXES:
            raise ValueError(
                f"unknown loa {loa!r}; expected one of {sorted(_LOA_AXES)}"
            )
        self._loa = code
        self._klass = _loa_to_class(code)
        self._time_id, self._entity_id = _LOA_AXES[code]

        self._times = _sorted_unique_int64(times, "times")
        self._entities = _sorted_unique_int64(entities, "entities")
        self._specs = _normalize_specs(variables)
        if not self._specs:
            raise ValueError("variables must declare at least one column")
        if int(sample_size) < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size}")
        self._sample_size = int(sample_size)

        if targets is not None:
            missing = sorted(set(targets) - set(self._specs))
            if missing:
                raise ValueError(
                    f"targets {missing} are not declared in variables"
                )
        if metadata:
            clash = sorted(_RESERVED_ATTRS & set(metadata))
            if clash:
                raise ValueError(
                    f"metadata keys {clash} are reserved schema attributes"
                )

        is_prediction = any(n.startswith("pred_") for n in self._specs)
        attrs = build_schema_attrs(
            self._specs,
            targets=list(targets) if targets else None,
            is_prediction=is_prediction,
            time_id=self._time_id,
            entity_id=self._entity_id,
            sample_size=self._sample_size,
            broadcast_features=broadcast_features,
            extra_attrs=metadata,
        )

        # The ZarrStore owns lifecycle. With ``path=None`` the scaffold lives
        # in its scratch dir (optionally under ``base_dir`` for a persistent
        # volume); with an absolute ``path`` the scratch dir stays empty and
        # closing it never touches the durable store.
        self._store = ZarrStore(prefix="views_builder_", base_dir=base_dir)
        self._path = (
            Path(path) if path is not None
            else self._store.path / "dataset.zarr"
        )
        self._writer = GridWriter(
            self._path,
            self._time_id,
            self._entity_id,
            self._times,
            self._entities,
            self._sample_size,
            self._specs,
            attrs,
            chunks=chunks,
        )
        self._strict = bool(strict)
        self._written: np.ndarray | None = (
            np.zeros((self._times.size, self._entities.size), dtype=bool)
            if (strict or track_coverage)
            else None
        )
        self._built = False
        self._closed = False

    # ---- properties ------------------------------------------------------
    @property
    def loa(self) -> str:
        """The LOA code this builder scaffolds (``pgm``/``cm``/...)."""
        return self._loa

    @property
    def path(self) -> Path:
        """The Zarr store path being written."""
        return self._path

    @property
    def coverage(self) -> float | None:
        """Fraction of (time, entity) cells written; None if not tracked."""
        if self._written is None:
            return None
        return float(self._written.mean())

    # ---- writing ---------------------------------------------------------
    def write_batch(
        self,
        times: Sequence[int] | np.ndarray | int,
        entities: Sequence[int] | np.ndarray | int,
        columns: dict[str, Any],
    ) -> None:
        """Scatter-write one batch of ``(time, entity)`` rows to the store.

        ``columns`` maps variable names (a subset of the scaffold's
        variables is allowed; unknown names fail loud) to arrays of
        ``(N, sample_size)`` for ``num3`` variables and ``(N,)`` for
        ``num2``. A scalar side is broadcast against the other (so
        ``times=t`` with an entity array writes one month). Peak memory is
        one batch — the caller may free its buffers after the call.
        """
        self._require_writable()
        times_b = _as_int64_1d(times, "times")
        entities_b = _as_int64_1d(entities, "entities")
        if times_b.size == 1 and entities_b.size != 1:
            times_b = np.full(entities_b.size, times_b[0], dtype="int64")
        elif entities_b.size == 1 and times_b.size != 1:
            entities_b = np.full(times_b.size, entities_b[0], dtype="int64")
        if times_b.shape != entities_b.shape:
            raise ValueError(
                f"times ({times_b.size}) and entities ({entities_b.size}) "
                "must have the same length"
            )
        if times_b.size == 0:
            return
        if not columns:
            raise ValueError("write_batch requires at least one column")
        unknown = sorted(set(columns) - set(self._specs))
        if unknown:
            raise ValueError(
                f"unknown variable(s) {unknown}; scaffold declares "
                f"{sorted(self._specs)}"
            )
        tp = _locate(self._times, times_b, "time")
        ep = _locate(self._entities, entities_b, "entity")
        cols_b = {
            name: self._validate_column(name, values, times_b.size)
            for name, values in columns.items()
        }
        self._mark(tp, ep)
        self._writer.write_batch(times_b, entities_b, cols_b)

    def write_time_slice(
        self, time: int, columns: dict[str, Any]
    ) -> None:
        """Write the full entity slice for one time step (region write).

        ``columns`` maps variable names to ``(E, sample_size)`` arrays for
        ``num3`` variables and ``(E,)`` for ``num2`` — the fastest pattern
        for month-by-month producers.
        """
        self._require_writable()
        (ti,) = _locate(self._times, np.asarray([time], dtype="int64"), "time")
        if not columns:
            raise ValueError("write_time_slice requires at least one column")
        unknown = sorted(set(columns) - set(self._specs))
        if unknown:
            raise ValueError(
                f"unknown variable(s) {unknown}; scaffold declares "
                f"{sorted(self._specs)}"
            )
        e = self._entities.size
        for name, values in columns.items():
            spec = self._specs[name]
            arr = np.asarray(values)
            if spec == "num3":
                if arr.size != e * self._sample_size:
                    raise ValueError(
                        f"column {name!r}: expected {e} x {self._sample_size} "
                        f"values for the full entity slice, got {arr.size}"
                    )
                block = arr.reshape(e, self._sample_size).astype(_FLOAT)
                self._writer.group[name][int(ti), :, :] = block
            else:
                if arr.size != e:
                    raise ValueError(
                        f"column {name!r}: expected {e} values for the full "
                        f"entity slice, got {arr.size}"
                    )
                self._writer.group[name][int(ti), :] = (
                    arr.reshape(e).astype(_FLOAT)
                )
        if self._written is not None:
            if self._strict and self._written[int(ti), :].any():
                raise ValueError(
                    f"time step {int(time)} was already written (strict=True)"
                )
            self._written[int(ti), :] = True

    # ---- finalization ------------------------------------------------------
    def build(self, require_complete: bool = False) -> ViewsDataset:
        """Finalize and return the dataset (the LOA subclass for ``loa``).

        Store ownership transfers to the returned dataset; the builder is
        spent. ``require_complete=True`` fails loud if any (time, entity)
        cell was never written (requires coverage tracking).
        """
        self._require_writable()
        if require_complete:
            if self._written is None:
                raise ValueError(
                    "require_complete=True needs coverage tracking; construct "
                    "the builder with strict=True or track_coverage=True"
                )
            missing = int((~self._written).sum())
            if missing:
                raise ValueError(
                    f"{missing} of {self._written.size} (time, entity) cells "
                    "were never written"
                )
        self._built = True
        return self._klass._adopt_store(self._store, self._path)

    def close(self) -> None:
        """Clean up the scratch store. Idempotent; a no-op after
        :meth:`build` (ownership has transferred to the dataset)."""
        if self._closed or self._built:
            return
        self._closed = True
        self._store.close()

    def __enter__(self) -> "DatasetBuilder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "built" if self._built else (
            "closed" if self._closed else "open"
        )
        return (
            f"DatasetBuilder(loa={self._loa!r}, "
            f"time_steps={self._times.size}, entities={self._entities.size}, "
            f"variables={sorted(self._specs)}, state={state})"
        )

    # ---- internals ---------------------------------------------------------
    def _require_writable(self) -> None:
        if self._closed:
            raise RuntimeError("builder is closed")
        if self._built:
            raise RuntimeError(
                "build() has already been called; the store now belongs to "
                "the returned dataset"
            )

    def _validate_column(
        self, name: str, values: Any, n_rows: int
    ) -> np.ndarray:
        spec = self._specs[name]
        arr = np.asarray(values)
        if spec == "num3":
            if arr.size != n_rows * self._sample_size:
                raise ValueError(
                    f"column {name!r}: expected {n_rows} x "
                    f"{self._sample_size} values, got {arr.size}"
                )
            return arr.reshape(n_rows, self._sample_size).astype(_FLOAT)
        if arr.size != n_rows:
            raise ValueError(
                f"column {name!r}: expected {n_rows} values, got {arr.size}"
            )
        return arr.reshape(n_rows).astype(_FLOAT)

    def _mark(self, tp: np.ndarray, ep: np.ndarray) -> None:
        if self._written is None:
            return
        if self._strict:
            already = self._written[tp, ep]
            if already.any():
                raise ValueError(
                    f"{int(already.sum())} (time, entity) cell(s) in this "
                    "batch were already written (strict=True)"
                )
        self._written[tp, ep] = True
