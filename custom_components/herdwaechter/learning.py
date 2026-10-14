"""Continuous learning: label data set, power levels, label rules, quality.

Pure Python + numpy, no Home Assistant imports.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any
import zlib

import numpy as np
import numpy.typing as npt

from .model import brier_score, precision_recall

FloatArray = npt.NDArray[np.float64]

MAX_DATASET_SIZE = 5000
HOLDOUT_MODULO = 5  # 1 in 5 label groups (~20 %) is held out
MIN_HOLDOUT_PER_CLASS = 5
MIN_GROUPS_FOR_HOLDOUT = 3
MAX_MODEL_WEIGHT = 0.9

POSITIVE_WEIGHT = 1.0
STRONG_NEGATIVE_WEIGHT = 3.0
ANSWER_WEIGHT = 3.0
WEAK_NEGATIVE_WEIGHT = 0.2
RELATIVE_LEVEL_SUPPORT = 0.15
# One quiet window out of this many becomes a weak negative label.
WEAK_NEGATIVE_EVERY = 40


class LabelSource(IntEnum):
    """Where a label came from."""

    SWITCH = 1
    EXTERNAL = 2
    FALSE_ALARM = 3
    ANSWER_YES = 4
    ANSWER_NO = 5
    WEAK_NEGATIVE = 6
    SERVICE = 7


STRONG_SOURCES = frozenset(
    {
        LabelSource.SWITCH,
        LabelSource.EXTERNAL,
        LabelSource.FALSE_ALARM,
        LabelSource.ANSWER_YES,
        LabelSource.ANSWER_NO,
        LabelSource.SERVICE,
    }
)


def _encode(array: npt.NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _decode(text: str, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    return np.frombuffer(base64.b64decode(text.encode("ascii")), dtype=dtype).copy()


def is_holdout_group(group: int) -> bool:
    """Deterministically assign ~20 % of label groups to the holdout set."""
    return zlib.crc32(str(int(group)).encode("ascii")) % HOLDOUT_MODULO == 0


# ------------------------------------------------------------------- data set


class LabelDataset:
    """Labelled feature vectors (no raw data), bounded in size.

    Every entry has a feature vector, a label (0/1), a weight, the timestamp
    of the window end, the label source and a group id. All windows that
    belong to the same labelling action share a group id; the holdout split
    is done per group so that near-identical overlapping windows never end up
    on both sides.
    """

    def __init__(self, n_features: int, max_size: int = MAX_DATASET_SIZE) -> None:
        """Create an empty data set."""
        self.n_features = n_features
        self.max_size = max_size
        self._x = np.zeros((max_size + 1, n_features), dtype=np.float32)
        self._y = np.zeros(max_size + 1, dtype=np.int8)
        self._w = np.zeros(max_size + 1, dtype=np.float32)
        self._ts = np.zeros(max_size + 1, dtype=np.float64)
        self._src = np.zeros(max_size + 1, dtype=np.int8)
        self._group = np.zeros(max_size + 1, dtype=np.int64)
        self._n = 0

    def __len__(self) -> int:
        """Return the number of entries."""
        return self._n

    @property
    def x(self) -> FloatArray:
        """Return the feature matrix (float64 copy)."""
        return self._x[: self._n].astype(np.float64)

    @property
    def y(self) -> FloatArray:
        """Return the labels as float."""
        return self._y[: self._n].astype(np.float64)

    @property
    def weight(self) -> FloatArray:
        """Return the sample weights."""
        return self._w[: self._n].astype(np.float64)

    @property
    def ts(self) -> FloatArray:
        """Return the timestamps."""
        return self._ts[: self._n]

    @property
    def source(self) -> npt.NDArray[np.int8]:
        """Return the label sources."""
        return self._src[: self._n]

    @property
    def group(self) -> npt.NDArray[np.int64]:
        """Return the group ids."""
        return self._group[: self._n]

    def add(
        self,
        features: FloatArray,
        label: int,
        weight: float,
        ts: float,
        source: LabelSource,
        group: int,
    ) -> None:
        """Add or replace (same timestamp) a labelled window."""
        existing = np.flatnonzero(self._ts[: self._n] == ts)
        if existing.size:
            i = int(existing[0])
            # a weak label never overrides a strong one
            if (
                source == LabelSource.WEAK_NEGATIVE
                and LabelSource(int(self._src[i])) in STRONG_SOURCES
            ):
                return
        else:
            i = self._n
            self._n += 1
        self._x[i] = features
        self._y[i] = 1 if label else 0
        self._w[i] = weight
        self._ts[i] = ts
        self._src[i] = int(source)
        self._group[i] = group
        if self._n > self.max_size:
            self.thin()

    def _keep(self, keep: npt.NDArray[np.bool_]) -> None:
        idx = np.flatnonzero(keep)
        n = idx.size
        for arr in (self._x, self._y, self._w, self._ts, self._src, self._group):
            arr[:n] = arr[idx]
        self._n = n

    def thin(self) -> None:
        """Drop every other entry in the older half (weak negatives first).

        This allows drift (recent data dominates) without forgetting older
        situations entirely.
        """
        while self._n > self.max_size:
            n = self._n
            order = np.argsort(self._ts[:n], kind="stable")
            older = order[: n // 2]
            weak = older[self._src[older] == int(LabelSource.WEAK_NEGATIVE)]
            candidates = weak if weak.size >= 2 else older
            drop = candidates[::2]
            keep = np.ones(n, dtype=bool)
            keep[drop[: max(1, n - self.max_size + n // 20)]] = False
            self._keep(keep)

    def clear(self) -> None:
        """Remove all entries."""
        self._n = 0

    def holdout_mask(self) -> npt.NDArray[np.bool_]:
        """Return True for entries that belong to the holdout set.

        Groups are assigned by a hash of their id. A label category (positive,
        strong negative, weak negative) with fewer than
        ``MIN_GROUPS_FOR_HOLDOUT`` groups is kept completely in the training
        set: a single false alarm, for example, cannot measure generalisation
        but is exactly what the model has to learn from.
        """
        n = self._n
        mask = np.zeros(n, dtype=bool)
        if n == 0:
            return mask
        groups = self._group[:n]
        y = self._y[:n]
        weak = self._src[:n] == int(LabelSource.WEAK_NEGATIVE)
        categories = ((y == 1), (y == 0) & ~weak, (y == 0) & weak)
        for sel in categories:
            unique = np.unique(groups[sel])
            if unique.size < MIN_GROUPS_FOR_HOLDOUT:
                continue
            held = {int(g) for g in unique if is_holdout_group(int(g))}
            if held:
                mask |= sel & np.isin(groups, list(held))
        return mask

    def counts(self) -> dict[str, int]:
        """Return label statistics."""
        y = self._y[: self._n]
        groups = self._group[: self._n]
        return {
            "positive": int(np.count_nonzero(y == 1)),
            "negative": int(np.count_nonzero(y == 0)),
            "positive_groups": int(np.unique(groups[y == 1]).size),
            "negative_groups": int(np.unique(groups[y == 0]).size),
            "strong_negative": int(
                np.count_nonzero(
                    (y == 0) & (self._src[: self._n] != int(LabelSource.WEAK_NEGATIVE))
                )
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise compactly (base64 encoded little endian arrays).

        The feature matrix is stored as zlib-compressed float16, which keeps
        the storage file small (relevant for SD cards) at a precision that is
        far below the noise of the features.
        """
        n = self._n
        x16 = np.clip(self._x[:n], -60000.0, 60000.0).astype("<f2")
        return {
            "n": n,
            "n_features": self.n_features,
            "x_format": "f2z",
            "x": base64.b64encode(zlib.compress(x16.tobytes(), 6)).decode("ascii"),
            "y": _encode(self._y[:n].astype("<i1")),
            "w": _encode(self._w[:n].astype("<f4")),
            "ts": _encode(self._ts[:n].astype("<f8")),
            "src": _encode(self._src[:n].astype("<i1")),
            "group": _encode(self._group[:n].astype("<i8")),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], n_features: int, max_size: int = MAX_DATASET_SIZE
    ) -> LabelDataset:
        """Restore; an incompatible payload yields an empty data set."""
        obj = cls(n_features, max_size)
        try:
            n = int(data.get("n", 0))
            if n == 0 or int(data.get("n_features", -1)) != n_features:
                return obj
            if data.get("x_format") == "f2z":
                raw = zlib.decompress(base64.b64decode(data["x"].encode("ascii")))
                x = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            else:
                x = _decode(data["x"], "<f4")
            x = x.reshape(n, n_features)
            y = _decode(data["y"], "<i1")
            w = _decode(data["w"], "<f4")
            ts = _decode(data["ts"], "<f8")
            src = _decode(data["src"], "<i1")
            group = _decode(data["group"], "<i8")
        except (KeyError, ValueError, TypeError, zlib.error):
            return obj
        n = min(n, max_size)
        obj._x[:n] = x[-n:]
        obj._y[:n] = y[-n:]
        obj._w[:n] = w[-n:]
        obj._ts[:n] = ts[-n:]
        obj._src[:n] = src[-n:]
        obj._group[:n] = group[-n:]
        obj._n = n
        return obj


# --------------------------------------------------------------- power levels


@dataclass(slots=True)
class _Cluster:
    center: float
    support: float
    negative: float = 0.0


class PowerLevelClusterer:
    """Incremental 1D clustering of step heights.

    A height joins the nearest cluster if it is within ``tolerance`` of its
    centre, otherwise it starts a new cluster. Clusters that come closer than
    the tolerance are merged. The effective memory is limited so that the
    centres can follow slow drift. Negative evidence (false alarms) is
    tracked separately; a level is only reported when it has enough positive
    support and clearly more positive than negative evidence.
    """

    def __init__(
        self,
        tolerance: float = 0.12,
        min_support: float = 10.0,
        max_clusters: int = 12,
        memory: float = 400.0,
    ) -> None:
        """Create an empty clusterer."""
        self.tolerance = tolerance
        self.min_support = min_support
        self.max_clusters = max_clusters
        self.memory = memory
        self._clusters: list[_Cluster] = []

    def _nearest(self, height: float) -> int:
        best = -1
        best_err = self.tolerance
        for i, c in enumerate(self._clusters):
            err = abs(height - c.center) / c.center
            if err <= best_err:
                best = i
                best_err = err
        return best

    def add(self, height: float, weight: float = 1.0) -> None:
        """Add positive evidence for a step height (W, absolute)."""
        height = abs(height)
        if height <= 0 or weight <= 0:
            return
        i = self._nearest(height)
        if i < 0:
            self._clusters.append(_Cluster(height, weight))
            if len(self._clusters) > self.max_clusters:
                weakest = min(
                    range(len(self._clusters)),
                    key=lambda k: (
                        self._clusters[k].support - self._clusters[k].negative
                    ),
                )
                del self._clusters[weakest]
        else:
            c = self._clusters[i]
            eff = min(c.support, self.memory)
            c.center += weight / (eff + weight) * (height - c.center)
            c.support = min(c.support + weight, 10 * self.memory)
        self._merge()

    def add_negative(self, height: float, weight: float = 1.0) -> None:
        """Add negative evidence (a false alarm showed this height)."""
        i = self._nearest(abs(height))
        if i >= 0:
            self._clusters[i].negative += weight

    def _merge(self) -> None:
        self._clusters.sort(key=lambda c: c.center)
        merged: list[_Cluster] = []
        for c in self._clusters:
            if merged and (c.center - merged[-1].center) / merged[-1].center < (
                self.tolerance
            ):
                m = merged[-1]
                total = m.support + c.support
                m.center = (m.center * m.support + c.center * c.support) / total
                m.support = total
                m.negative += c.negative
            else:
                merged.append(c)
        self._clusters = merged

    def levels(self) -> FloatArray:
        """Return the confirmed power levels in W (sorted).

        A level needs an absolute minimum support, at least 15 % of the
        support of the strongest level (rejects occasional sums of zones and
        stray appliances) and clearly more positive than negative evidence.
        """
        if not self._clusters:
            return np.empty(0)
        top = max(c.support for c in self._clusters)
        needed = max(self.min_support, RELATIVE_LEVEL_SUPPORT * top)
        return np.array(
            [
                c.center
                for c in self._clusters
                if c.support >= needed and c.support >= 2.0 * c.negative
            ],
            dtype=np.float64,
        )

    def clear(self) -> None:
        """Forget all clusters."""
        self._clusters = []

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage."""
        return {
            "clusters": [
                [round(c.center, 1), round(c.support, 3), round(c.negative, 3)]
                for c in self._clusters
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PowerLevelClusterer:
        """Restore from storage."""
        obj = cls()
        for item in data.get("clusters", []):
            try:
                center, support, negative = (float(v) for v in item)
            except (TypeError, ValueError):
                continue
            if center > 0:
                obj._clusters.append(_Cluster(center, support, negative))
        obj._merge()
        return obj


# --------------------------------------------------------------- label rules


@dataclass(frozen=True, slots=True)
class LabelDecision:
    """An automatic label for the current window."""

    label: int
    weight: float
    source: LabelSource
    group: int


class LabelPolicy:
    """Positive-unlabelled rules for automatic labels.

    * label switch on: windows are positive (only if they show any edge
      activity, so a forgotten switch does not teach that silence is cooking)
    * label switched off: the next evaluations are neutral (run-on)
    * windows without any edge activity: sampled weak negatives
    * activity without a label: never labelled automatically
    """

    def __init__(
        self,
        neutral_after_label: int = 2,
        weak_negative_every: int = WEAK_NEGATIVE_EVERY,
    ) -> None:
        """Initialise counters."""
        self.neutral_after_label = neutral_after_label
        self.weak_negative_every = weak_negative_every
        self._neutral_left = 0
        self._label_was_active = False
        self._quiet_counter = 0

    def decide(
        self,
        now: float,
        label_active: bool,
        label_group: int,
        label_source: LabelSource,
        n_edges: int,
        output_on: bool,
    ) -> LabelDecision | None:
        """Return the automatic label for this evaluation, if any."""
        if label_active:
            self._label_was_active = True
            self._neutral_left = 0
            if n_edges > 0:
                return LabelDecision(1, POSITIVE_WEIGHT, label_source, label_group)
            return None
        if self._label_was_active:
            self._label_was_active = False
            self._neutral_left = self.neutral_after_label
        if self._neutral_left > 0:
            self._neutral_left -= 1
            return None
        if n_edges == 0 and not output_on:
            self._quiet_counter += 1
            if self._quiet_counter >= self.weak_negative_every:
                self._quiet_counter = 0
                return LabelDecision(
                    0,
                    WEAK_NEGATIVE_WEIGHT,
                    LabelSource.WEAK_NEGATIVE,
                    int(now // 3600),
                )
        return None


# -------------------------------------------------------------------- quality


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Holdout evaluation of the personal model."""

    known: bool = False
    n_positive: int = 0
    n_negative: int = 0
    precision: float | None = None
    recall: float | None = None
    brier: float | None = None
    brier_prior: float | None = None
    best_mix: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage / diagnostics."""
        return {
            "known": self.known,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "precision": self.precision,
            "recall": self.recall,
            "brier": self.brier,
            "brier_prior": self.brier_prior,
            "best_mix": self.best_mix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityReport:
        """Restore from storage."""

        def opt(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            known=bool(data.get("known", False)),
            n_positive=int(data.get("n_positive", 0)),
            n_negative=int(data.get("n_negative", 0)),
            precision=opt("precision"),
            recall=opt("recall"),
            brier=opt("brier"),
            brier_prior=opt("brier_prior"),
            best_mix=opt("best_mix"),
        )


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def evaluate_holdout(
    p_model: FloatArray, p_prior: FloatArray, y: FloatArray, weight: FloatArray
) -> QualityReport:
    """Compute precision, recall and Brier score on the holdout set."""
    n_pos = int(np.count_nonzero(y > 0.5))
    n_neg = int(np.count_nonzero(y <= 0.5))
    if p_model.shape[0] == 0:
        return QualityReport(n_positive=n_pos, n_negative=n_neg)
    precision, recall = precision_recall(p_model, y)
    known = n_pos >= MIN_HOLDOUT_PER_CLASS and n_neg >= MIN_HOLDOUT_PER_CLASS
    # balance both classes so that easy weak negatives do not dominate
    w = weight.copy()
    for cls in (0.0, 1.0):
        sel = (y > 0.5) == bool(cls)
        total = float(w[sel].sum())
        if total > 0:
            w[sel] /= total
    # the mixture weight that minimises the holdout Brier score
    grid = np.linspace(0.0, 1.0, 21)
    scores = [brier_score(m * p_model + (1 - m) * p_prior, y, w) for m in grid]
    best = float(grid[int(np.argmin(scores))])
    return QualityReport(
        known=known,
        n_positive=n_pos,
        n_negative=n_neg,
        precision=_finite_or_none(precision),
        recall=_finite_or_none(recall),
        brier=_finite_or_none(brier_score(p_model, y, w)),
        brier_prior=_finite_or_none(brier_score(p_prior, y, w)),
        best_mix=best,
    )


def maturity(positive_groups: int, negative_groups: int) -> float:
    """Return 0..1 depending on how many independent label groups exist."""
    return (1.0 - math.exp(-positive_groups / 3.0)) * (
        1.0 - math.exp(-negative_groups / 6.0)
    )


def mixing_weight(
    positive_groups: int, negative_groups: int, quality: QualityReport
) -> float:
    """Return the weight ``w`` of the personal model in ``p = w*p_model + ...``.

    ``w`` grows with the number of independent positive and negative label
    groups (maturity) and with the measured holdout quality: once the
    quality is known, ``w`` is the mixture weight that minimises the holdout
    Brier score of the blended probability. A model that does not beat the
    prior therefore gets little or no weight. While the quality is unknown,
    ``w`` is capped at half of its maximum.
    """
    base = MAX_MODEL_WEIGHT * maturity(positive_groups, negative_groups)
    if base <= 0:
        return 0.0
    if not quality.known or quality.best_mix is None:
        return 0.5 * base
    return base * float(np.clip(quality.best_mix, 0.0, 1.0))
