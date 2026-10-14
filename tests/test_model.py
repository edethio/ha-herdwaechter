"""Tests for the online logistic regression."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from custom_components.herdwaechter.model import (
    OnlineLogisticRegression,
    RunningStandardizer,
    brier_score,
    precision_recall,
)


def _data(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)) * [1.0, 5.0, 100.0, 0.1] + [0, 3, 1000, 0]
    logit = 2.0 * x[:, 0] - 0.5 * (x[:, 1] - 3) + 0.01 * (x[:, 2] - 1000)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return x, y


def test_standardizer_welford_matches_batch() -> None:
    x, _ = _data(500)
    online = RunningStandardizer(4)
    online.update(x[:200])
    online.update(x[200:])
    batch = RunningStandardizer(4)
    batch.refit(x)
    np.testing.assert_allclose(online.mean, batch.mean)
    np.testing.assert_allclose(online.std, batch.std)
    np.testing.assert_allclose(online.std, x.std(axis=0, ddof=1))
    assert np.all(RunningStandardizer(3).std == 1.0)
    empty = RunningStandardizer(2)
    empty.refit(np.empty((0, 2)))
    assert empty.count == 0
    restored = RunningStandardizer.from_dict(online.to_dict(), 4)
    np.testing.assert_allclose(restored.mean, online.mean)
    assert RunningStandardizer.from_dict(online.to_dict(), 3).count == 0


def test_fit_converges() -> None:
    x, y = _data()
    model = OnlineLogisticRegression(4)
    assert not model.is_trained
    np.testing.assert_allclose(model.predict_proba(x[:3]), 0.5)
    model.fit(x, y, np.ones_like(y))
    assert model.is_trained
    p = model.predict_proba(x)
    accuracy = float(np.mean((p >= 0.5) == (y > 0.5)))
    assert accuracy > 0.8
    assert brier_score(p, y) < 0.15
    # coefficients point the right way (standardised space)
    assert model.coef[0] > 0
    assert model.coef[1] < 0


def test_fit_is_stable_on_separable_data() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 3))
    y = (x[:, 0] > 0).astype(float)
    model = OnlineLogisticRegression(3)
    model.fit(x, y, np.ones(800) * 5.0)
    assert np.all(np.isfinite(model.coef))
    assert abs(model.intercept) < 10
    assert np.max(np.abs(model.coef)) < 50
    assert float(np.mean((model.predict_proba(x) >= 0.5) == (y > 0.5))) > 0.98
    model.fit(np.empty((0, 3)), np.empty(0), np.empty(0))  # no-op


def test_class_weights_shift_the_decision() -> None:
    x, y = _data()
    w_pos = np.where(y > 0.5, 10.0, 1.0)
    neutral = OnlineLogisticRegression(4)
    neutral.fit(x, y, np.ones_like(y))
    favour = OnlineLogisticRegression(4)
    favour.fit(x, y, w_pos)
    assert favour.predict_proba(x).mean() > neutral.predict_proba(x).mean()


def test_partial_fit_learns_online() -> None:
    x, y = _data()
    model = OnlineLogisticRegression(4, learning_rate=0.2)
    for i in range(0, 1500, 15):
        model.partial_fit(x[i : i + 15], y[i : i + 15], np.ones(15), new_rows=15)
    p = model.predict_proba(x)
    assert float(np.mean((p >= 0.5) == (y > 0.5))) > 0.75
    assert model.scaler.count == 1500
    model.partial_fit(np.empty((0, 4)), np.empty(0), np.empty(0))
    model.partial_fit(x[:2], y[:2], np.zeros(2))


def test_save_and_load_roundtrip() -> None:
    x, y = _data()
    model = OnlineLogisticRegression(4)
    model.fit(x, y, np.ones_like(y))
    data = json.loads(json.dumps(model.to_dict()))
    restored = OnlineLogisticRegression.from_dict(data, 4)
    np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x))
    assert restored.n_updates == model.n_updates
    mismatch = OnlineLogisticRegression.from_dict(data, 5)
    assert not mismatch.is_trained


def test_metrics() -> None:
    p = np.array([0.9, 0.8, 0.2, 0.6])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier_score(p, y) == pytest.approx((0.01 + 0.04 + 0.04 + 0.36) / 4)
    assert math.isnan(brier_score(np.empty(0), np.empty(0)))
    precision, recall = precision_recall(p, y)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)
    precision, recall = precision_recall(np.zeros(2), np.zeros(2))
    assert math.isnan(precision)
    assert math.isnan(recall)


def test_offset_model_starts_at_the_prior() -> None:
    x, y = _data()
    prior_logit = 2.0 * x[:, 0]
    model = OnlineLogisticRegression(4)
    np.testing.assert_allclose(
        model.predict_proba(x[:5], prior_logit[:5]), 1 / (1 + np.exp(-prior_logit[:5]))
    )
    model.fit(x, y, np.ones_like(y), offset=prior_logit)
    # the offset already explains the first feature: little correction needed
    assert abs(model.coef[0]) < 1.0
    p = model.predict_proba(x, prior_logit)
    assert float(np.mean((p >= 0.5) == (y > 0.5))) > 0.8
    model.partial_fit(x[:10], y[:10], np.ones(10), offset=prior_logit[:10])
    assert np.all(np.isfinite(model.coef))
