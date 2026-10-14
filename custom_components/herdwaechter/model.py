"""Online logistic regression in plain numpy.

* running feature standardisation (Welford)
* L2 regularisation (intercept not penalised)
* per-sample weights (used for label strength and class balancing)
* cheap mini-batch gradient updates for every new label
* exact, damped Newton/IRLS refits over the stored label data set
* an optional per-row offset (the prior logit), so that the model learns a
  correction of the prior: without data it predicts the prior, and the
  regularisation pulls towards the prior instead of towards 50 %
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-9
# The L2 penalty per unit of sample weight is l2 / REGULARISATION_REFERENCE.
REGULARISATION_REFERENCE = 100.0


def _sigmoid(z: FloatArray) -> FloatArray:
    out: FloatArray = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return out


class RunningStandardizer:
    """Welford running mean / variance per feature."""

    def __init__(self, n_features: int) -> None:
        """Start with no observations."""
        self.n_features = n_features
        self.count = 0
        self.mean = np.zeros(n_features)
        self.m2 = np.zeros(n_features)

    def update(self, matrix: FloatArray) -> None:
        """Add observations (rows of ``matrix``)."""
        for row in np.atleast_2d(matrix):
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    def refit(self, matrix: FloatArray) -> None:
        """Recompute the statistics exactly from ``matrix``."""
        x = np.atleast_2d(matrix)
        self.count = int(x.shape[0])
        if self.count == 0:
            self.mean = np.zeros(self.n_features)
            self.m2 = np.zeros(self.n_features)
            return
        self.mean = x.mean(axis=0)
        self.m2 = ((x - self.mean) ** 2).sum(axis=0)

    @property
    def std(self) -> FloatArray:
        """Return the standard deviation (1 for constant or unseen features)."""
        if self.count < 2:
            return np.ones(self.n_features)
        std: FloatArray = np.sqrt(self.m2 / (self.count - 1))
        return np.where(std > 1e-6, std, 1.0)

    def transform(self, matrix: FloatArray) -> FloatArray:
        """Standardise ``matrix``."""
        out: FloatArray = (np.atleast_2d(matrix) - self.mean) / self.std
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage."""
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], n_features: int) -> RunningStandardizer:
        """Restore from storage (ignores data of a different dimension)."""
        obj = cls(n_features)
        mean = np.asarray(data.get("mean", []), dtype=np.float64)
        m2 = np.asarray(data.get("m2", []), dtype=np.float64)
        if mean.shape == (n_features,) and m2.shape == (n_features,):
            obj.count = int(data.get("count", 0))
            obj.mean = mean
            obj.m2 = m2
        return obj


class OnlineLogisticRegression:
    """Weighted, L2 regularised logistic regression."""

    def __init__(
        self, n_features: int, l2: float = 1.0, learning_rate: float = 0.05
    ) -> None:
        """Create an untrained model (predicts 0.5)."""
        self.n_features = n_features
        self.l2 = l2
        self.learning_rate = learning_rate
        self.coef = np.zeros(n_features)
        self.intercept = 0.0
        self.scaler = RunningStandardizer(n_features)
        self.n_updates = 0

    @property
    def is_trained(self) -> bool:
        """Return True once the model has seen at least one update."""
        return self.n_updates > 0

    def decision_function(
        self, matrix: FloatArray, offset: FloatArray | None = None
    ) -> FloatArray:
        """Return the logit for each row."""
        z: FloatArray = self.scaler.transform(matrix) @ self.coef + self.intercept
        if offset is not None:
            z = z + offset
        return z

    def predict_proba(
        self, matrix: FloatArray, offset: FloatArray | None = None
    ) -> FloatArray:
        """Return P(y = 1) for each row."""
        return _sigmoid(self.decision_function(matrix, offset))

    def partial_fit(
        self,
        matrix: FloatArray,
        y: FloatArray,
        weight: FloatArray,
        steps: int = 3,
        new_rows: int | None = None,
        offset: FloatArray | None = None,
    ) -> None:
        """Run a few gradient steps on a mini batch.

        Only the first ``new_rows`` rows (default: all) are new observations
        and update the running standardisation; the remaining rows are replayed
        samples from the stored data set. The L2 penalty uses the same
        per-sample strength as :meth:`fit`.
        """
        x = np.atleast_2d(matrix)
        if x.shape[0] == 0:
            return
        self.scaler.update(x if new_rows is None else x[:new_rows])
        xs = self.scaler.transform(x)
        w = np.asarray(weight, dtype=np.float64)
        total = float(w.sum())
        if total <= 0:
            return
        base = np.zeros(x.shape[0]) if offset is None else np.asarray(offset)
        for _ in range(steps):
            p = _sigmoid(xs @ self.coef + self.intercept + base)
            err = w * (p - y)
            grad = xs.T @ err / total + self.l2 * self.coef / REGULARISATION_REFERENCE
            self.coef -= self.learning_rate * grad
            self.intercept -= self.learning_rate * float(err.sum()) / total
        self.n_updates += 1

    def _penalty(self, total_weight: float, size: int) -> FloatArray:
        """L2 penalty relative to the data set weight (intercept unpenalised).

        The penalty grows with the total sample weight so that the
        regularisation strength per sample stays constant as data accumulates.
        """
        penalty = np.full(
            size, self.l2 * max(1.0, total_weight) / REGULARISATION_REFERENCE
        )
        penalty[-1] = 1e-6
        return penalty

    @staticmethod
    def _objective(
        xs: FloatArray,
        y: FloatArray,
        w: FloatArray,
        beta: FloatArray,
        penalty: FloatArray,
        offset: FloatArray,
    ) -> float:
        z = xs @ beta + offset
        loss = float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))
        return loss + 0.5 * float(np.sum(penalty * beta * beta))

    def fit(
        self,
        matrix: FloatArray,
        y: FloatArray,
        weight: FloatArray,
        max_iter: int = 30,
        tol: float = 1e-6,
        offset: FloatArray | None = None,
    ) -> None:
        """Fit exactly with damped Newton/IRLS on the whole (weighted) data set.

        A backtracking line search guarantees that every step decreases the
        regularised objective, so (nearly) separable data cannot make the
        iteration diverge.
        """
        x = np.atleast_2d(matrix)
        n = x.shape[0]
        if n == 0:
            return
        self.scaler.refit(x)
        xs = np.column_stack([self.scaler.transform(x), np.ones(n)])
        w = np.asarray(weight, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        base = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
        beta = np.zeros(xs.shape[1])
        penalty = self._penalty(float(w.sum()), beta.shape[0])
        current = self._objective(xs, y, w, beta, penalty, base)
        for _ in range(max_iter):
            p = _sigmoid(xs @ beta + base)
            grad = xs.T @ (w * (p - y)) + penalty * beta
            s = w * p * (1.0 - p)
            hess = (xs.T * s) @ xs + np.diag(penalty)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            decrease = float(grad @ step)
            t = 1.0
            while True:
                candidate = beta - t * step
                value = self._objective(xs, y, w, candidate, penalty, base)
                if value <= current - 1e-4 * t * decrease or t < 1e-4:
                    break
                t *= 0.5
            if value > current:
                break
            beta = candidate
            improvement = current - value
            current = value
            if float(np.max(np.abs(t * step))) < tol or improvement < tol:
                break
        self.coef = beta[:-1].copy()
        self.intercept = float(beta[-1])
        self.n_updates += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage."""
        return {
            "n_features": self.n_features,
            "l2": self.l2,
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "n_updates": self.n_updates,
            "scaler": self.scaler.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], n_features: int
    ) -> OnlineLogisticRegression:
        """Restore from storage; returns an untrained model on dimension mismatch."""
        obj = cls(n_features, l2=float(data.get("l2", 1.0)))
        coef = np.asarray(data.get("coef", []), dtype=np.float64)
        if coef.shape != (n_features,) or int(data.get("n_features", -1)) != n_features:
            return obj
        obj.coef = coef
        obj.intercept = float(data.get("intercept", 0.0))
        obj.n_updates = int(data.get("n_updates", 0))
        obj.scaler = RunningStandardizer.from_dict(data.get("scaler", {}), n_features)
        return obj


def brier_score(
    p: FloatArray, y: FloatArray, weight: FloatArray | None = None
) -> float:
    """Return the (weighted) Brier score."""
    if p.shape[0] == 0:
        return float("nan")
    w = np.ones_like(p) if weight is None else weight
    return float(np.sum(w * (p - y) ** 2) / max(float(np.sum(w)), _EPS))


def precision_recall(
    p: FloatArray, y: FloatArray, threshold: float = 0.5
) -> tuple[float, float]:
    """Return (precision, recall) at ``threshold``; NaN when undefined."""
    pred = p >= threshold
    pos = y > 0.5
    tp = float(np.count_nonzero(pred & pos))
    fp = float(np.count_nonzero(pred & ~pos))
    fn = float(np.count_nonzero(~pred & pos))
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    return precision, recall
