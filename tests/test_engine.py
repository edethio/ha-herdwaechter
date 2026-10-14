"""Tests for the detection engine: hysteresis, confidence, mixing, labels."""

from __future__ import annotations

import json

import numpy as np
import pytest

from custom_components.herdwaechter.const import (
    QUESTION_TIMEOUT_SECONDS,
    STATE_CALIBRATING,
    STATE_INSUFFICIENT_DATA,
    STATE_PRIOR_ONLY,
)
from custom_components.herdwaechter.engine import (
    EngineConfig,
    EvaluationInput,
    EvaluationResult,
    HerdwaechterEngine,
    Hysteresis,
    compute_confidence,
    edge_parameters,
    effective_weight,
)
from custom_components.herdwaechter.learning import LabelSource
from custom_components.herdwaechter.sampling import WindowData

from .helpers import cycling

T0 = 1_800_000_000.0
CFG = EngineConfig()
N = CFG.buffer_seconds


def _window(x: np.ndarray, end_ts: float) -> WindowData:
    n = x.shape[0]
    return WindowData(
        end_ts=end_ts,
        grid=x.astype(float),
        gen=np.zeros(n),
        sub=np.zeros(n),
        aux_missing=np.zeros(n, dtype=bool),
    )


def _cooking(seed: int = 0) -> np.ndarray:
    return cycling(
        N, power=1800, period=40, duty=0.5, start=N - 700, heat_up=90, seed=seed
    )


def _quiet(seed: int = 0) -> np.ndarray:
    return 200 + np.random.default_rng(seed).normal(0, 8, N)


def _eval(
    engine: HerdwaechterEngine,
    x: np.ndarray,
    now: float,
    label: bool = False,
    group: int = 0,
    interval: float | None = 3.0,
) -> EvaluationResult:
    return engine.evaluate(
        EvaluationInput(
            now=now,
            window=_window(x, now),
            local_hour=(now / 3600) % 24,
            local_day=int(now // 86400),
            update_interval_s=interval,
            label_active=label,
            label_group=group,
            label_source=LabelSource.SWITCH,
        )
    )


def test_edge_parameters() -> None:
    assert edge_parameters(None, False) == (4, 5, 9)
    assert edge_parameters(1.0, False) == (3, 3, 5)
    span, smooth, period_smooth = edge_parameters(10.0, True)
    assert span == 11
    assert smooth == 5
    assert period_smooth == 15


def test_hysteresis_on_off_and_timeout() -> None:
    h = Hysteresis(0.75, 0.35, consecutive=2, no_cycle_timeout_s=720)
    assert not h.update(0.8, 0, None)
    assert h.update(0.8, 30, None)
    assert h.update(0.5, 60, 60)
    assert h.update(None, 90, None)  # not evaluable: no change
    assert h.update(0.3, 120, 120)
    assert not h.update(0.3, 150, None)
    # a single low value does not switch off
    h.update(0.9, 200, None)
    assert h.update(0.9, 230, None)
    assert h.update(0.2, 260, 260)
    assert h.update(0.8, 290, 290)
    # 12 min without a matching cycling edge switches off
    assert h.update(0.8, 290 + 700, None)
    assert not h.update(0.8, 290 + 730, None)


def test_confidence_combines_decision_maturity_and_data_quality() -> None:
    assert compute_confidence(0.5, 0.9, 1.0, 3.0, True) == 0.0
    prior_only = compute_confidence(0.95, 0.0, 1.0, 3.0, True)
    mature = compute_confidence(0.95, 0.9, 1.0, 3.0, True)
    assert 0 < prior_only < mature <= 1.0
    assert compute_confidence(0.95, 0.9, 0.85, 3.0, True) < mature
    assert compute_confidence(0.95, 0.9, 1.0, 8.0, True) < mature
    slow = compute_confidence(0.95, 0.9, 1.0, 12.0, True)
    bad = compute_confidence(0.95, 0.9, 1.0, 20.0, True)
    assert bad < slow < mature
    assert bad < 0.2 * mature
    assert compute_confidence(0.95, 0.9, 1.0, None, False) < mature


def test_detects_cooking_and_ignores_quiet() -> None:
    engine = HerdwaechterEngine(CFG)
    assert engine.last_result.learning_state == STATE_CALIBRATING
    r1 = _eval(engine, _cooking(), T0)
    assert r1.evaluated
    assert r1.probability is not None
    assert r1.probability >= 0.75
    assert not r1.is_on  # needs two consecutive evaluations
    r2 = _eval(engine, _cooking(1), T0 + 30)
    assert r2.is_on
    assert engine.is_on
    assert r2.detected_period_s == pytest.approx(40, abs=3)
    assert r2.last_cycle is not None
    assert r2.last_cycle["power_w"] == pytest.approx(1800, rel=0.05)
    assert not any(k.startswith("_") for k in r2.last_cycle)
    assert r2.learning_state == STATE_CALIBRATING
    assert 0 < r2.confidence < 1
    for i in range(2, 4):
        r = _eval(engine, _quiet(i), T0 + 30 * i)
    assert r.probability is not None
    assert r.probability < 0.1
    assert not r.is_on


def test_gaps_make_window_not_evaluable() -> None:
    engine = HerdwaechterEngine(CFG)
    x = _cooking()
    x[-400:] = np.nan
    r = _eval(engine, x, T0)
    assert not r.evaluated
    assert r.learning_state == STATE_INSUFFICIENT_DATA
    assert r.confidence == 0.0
    r = _eval(engine, _quiet(), T0 + 30, interval=20.0)
    assert r.learning_state == STATE_INSUFFICIENT_DATA


def _calibrated_engine(**kwargs: object) -> HerdwaechterEngine:
    engine = HerdwaechterEngine(EngineConfig(**kwargs))  # type: ignore[arg-type]
    engine.calibrator.noise_sigma = 20.0
    engine.calibrator.noise_windows = 1000
    return engine


def _train(engine: HerdwaechterEngine, groups: int = 12) -> float:
    """Feed labelled cooking sessions and quiet hours; return the end time."""
    now = T0
    for g in range(groups):
        for k in range(6):
            _eval(engine, _cooking(g * 10 + k), now, label=True, group=g + 1)
            now += 30
        for k in range(3):  # neutral run-on after the label
            _eval(engine, _quiet(k), now)
            now += 30
        for k in range(200):  # quiet: weak negatives
            _eval(engine, _quiet(g * 100 + k), now)
            now += 120
    return now


def test_learning_from_labels_and_pu_rules() -> None:
    engine = _calibrated_engine()
    r = _eval(engine, _quiet(), T0 - 30)
    assert r.learning_state == STATE_PRIOR_ONLY
    now = _train(engine)
    counts = engine.dataset.counts()
    assert counts["positive"] == 12 * 6
    assert counts["negative"] > 0
    assert counts["strong_negative"] == 0
    assert engine.model.is_trained
    engine.retrain(now)
    assert engine.quality.known
    assert engine.model_weight > 0
    levels = engine.clusterer.levels()
    assert levels.shape[0] == 1
    assert levels[0] == pytest.approx(1800, rel=0.05)
    # activity without a label never becomes a negative example
    before = engine.dataset.counts()["negative"]
    for k in range(5):
        _eval(engine, _cooking(500 + k), now + 30 * k)
    assert engine.dataset.counts()["negative"] == before


def test_mixing_of_prior_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _calibrated_engine()
    _train(engine)
    engine.model_weight = 0.6
    feats = engine._history[-1].features.astype(float)
    p_prior, p_model, p = engine._probability(feats)
    assert p_model is not None
    w = effective_weight(0.6, p_prior, p_model)
    assert p == pytest.approx(w * p_model + (1 - w) * p_prior)
    engine.model_weight = 0.0
    assert engine._probability(feats)[1] is None
    # PU-safe gate: lowering is always allowed, raising only with prior support
    assert effective_weight(0.6, 0.9, 0.1) == pytest.approx(0.6)
    assert effective_weight(0.6, 0.1, 0.9) == 0.0
    assert effective_weight(0.6, 0.9, 0.95) == pytest.approx(0.6)
    assert 0.0 < effective_weight(0.6, 0.8, 0.9) < 0.6


def test_false_alarm_labels_event_and_suppresses() -> None:
    engine = _calibrated_engine()
    assert engine.mark_false_alarm(T0) == 0  # nothing detected yet
    engine._suppress_until = None
    now = T0 + 60
    for k in range(4):
        r = _eval(engine, _cooking(k), now)
        now += 30
    assert r.is_on
    count = engine.mark_false_alarm(now)
    assert count >= 2
    assert not engine.is_on
    assert engine.dataset.counts()["strong_negative"] == count
    # the same activity does not switch the output on again
    for k in range(4):
        r = _eval(engine, _cooking(10 + k), now + 30 * (k + 1))
    assert not r.is_on
    # paused learning: no labels
    engine.learning_paused = True
    assert engine.add_label(T0, now, False, now) == 0


def test_questions_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _calibrated_engine(questions_enabled=True, max_questions_per_day=1)
    monkeypatch.setattr(engine, "_probability", lambda f: (0.6, None, 0.6))
    r = _eval(engine, _cooking(), T0)
    assert r.question is not None
    q = r.question
    assert 0.4 <= q.probability <= 0.8
    assert q.expires == q.created + QUESTION_TIMEOUT_SECONDS
    # only one question per activity / day
    assert _eval(engine, _cooking(1), T0 + 30).question is None
    assert engine.answer_question("unknown", True, T0 + 40) is False
    assert engine.answer_question(q.question_id, True, T0 + 60) is True
    assert engine.dataset.counts()["positive"] >= 1
    assert engine.pending_questions == []
    # expiry
    monkeypatch.setattr(engine, "_probability", lambda f: (0.5, None, 0.5))
    engine._asked_activity = None
    engine._question_day = None
    r = _eval(engine, _cooking(2), T0 + 90)
    assert r.question is not None
    _eval(engine, _cooking(3), T0 + 90 + QUESTION_TIMEOUT_SECONDS + 1)
    assert engine.pending_questions == []
    # answering "no" creates strong negatives
    engine._asked_activity = None
    engine._question_day = None
    r = _eval(engine, _cooking(4), T0 + 4000)
    assert r.question is not None
    assert engine.answer_question(r.question.question_id, False, T0 + 4010)
    assert engine.dataset.counts()["strong_negative"] >= 1


def test_questions_disabled_or_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _calibrated_engine(questions_enabled=False)
    monkeypatch.setattr(engine, "_probability", lambda f: (0.6, None, 0.6))
    assert _eval(engine, _cooking(), T0).question is None
    engine = _calibrated_engine(questions_enabled=True)
    monkeypatch.setattr(engine, "_probability", lambda f: (0.95, None, 0.95))
    assert _eval(engine, _cooking(), T0).question is None
    monkeypatch.setattr(engine, "_probability", lambda f: (0.6, None, 0.6))
    assert _eval(engine, _cooking(), T0 + 30, label=True).question is None


def test_add_label_and_reset() -> None:
    engine = _calibrated_engine()
    now = T0
    for k in range(5):
        _eval(engine, _cooking(k), now)
        now += 30
    assert engine.add_label(now, T0, True, now) == 0
    assert engine.add_label(T0, now, True, now) == 5
    assert engine.add_label(T0, now, False, now) == 5
    assert engine.dataset.counts()["negative"] == 5
    engine.calibrator.lag_s = 2.0
    engine.reset_model()
    assert len(engine.dataset) == 0
    assert engine.model_weight == 0.0
    assert not engine.model.is_trained
    assert engine.calibrator.lag_s == 2.0  # calibration is kept


def test_persistence_roundtrip() -> None:
    engine = _calibrated_engine()
    _train(engine, groups=6)
    engine.calibrator.lag_s = 3.2
    engine.learning_paused = True
    data = json.loads(json.dumps(engine.to_dict()))
    assert not engine.dirty
    assert not engine.calibration_dirty
    restored = HerdwaechterEngine(EngineConfig())
    restored.load_dict(data)
    assert len(restored.dataset) == len(engine.dataset)
    assert restored.calibrator.lag_s == pytest.approx(3.2)
    assert restored.learning_paused
    assert restored.model_weight == pytest.approx(engine.model_weight)
    feats = engine._history[-1].features.astype(float)
    x = engine.profile.model_inputs(feats[None, :], engine.clusterer.levels())
    offset = engine.profile.prior_offsets(feats[None, :])
    np.testing.assert_allclose(
        restored.model.predict_proba(x, offset), engine.model.predict_proba(x, offset)
    )
    np.testing.assert_allclose(
        restored.clusterer.levels(), engine.clusterer.levels(), rtol=1e-4
    )
    other = HerdwaechterEngine(EngineConfig())
    other.load_dict({**data, "profile": "induction"})
    assert len(other.dataset) == 0
    untrained = HerdwaechterEngine(EngineConfig())
    untrained.load_dict({"model_weight": 0.7})
    assert untrained.model_weight == 0.0


def test_diagnostics_and_export_are_anonymous() -> None:
    engine = _calibrated_engine()
    _train(engine, groups=2)
    diag = engine.diagnostics()
    assert {"calibration", "model", "label_counts", "quality", "recent_windows"} <= set(
        diag
    )
    assert len(diag["recent_windows"]) == 10
    export = engine.export_training_data()
    assert export["samples"]
    sample = export["samples"][0]
    assert set(sample) == {"label", "weight", "source", "holdout", "features"}
    assert len(sample["features"]) == len(export["feature_names"])
    assert "ts" not in json.dumps(export)


def _run_timeline(signal: np.ndarray, start_eval: int) -> list[EvaluationResult]:
    """Evaluate a long synthetic signal every 30 s with sliding windows."""
    engine = _calibrated_engine()
    results = []
    for t in range(start_eval, signal.shape[0], 30):
        window = signal[t - N : t]
        results.append(_eval(engine, window, T0 + t))
    return results


def test_cycling_without_heatup_is_not_a_cooktop() -> None:
    total = 4000
    base = _quiet(5)
    quiet = np.r_[base, base, base, base][:total]
    # microwave programme: cycles right away (no continuous phase), 15 s period
    microwave = quiet.copy()
    t = 2400
    while t < 2400 + 420:
        microwave[t : t + 7] += 1350
        t += 15
    results = _run_timeline(microwave, N + 30)
    assert not any(r.is_on for r in results)
    # the same power with a continuous heat-up first is detected
    cooking = quiet.copy()
    cooking[2400:2520] += 1350
    t = 2520
    while t < total:
        t += 20
        cooking[t : t + 20] += 1350
        t += 20
    results = _run_timeline(cooking, N + 30)
    assert any(r.is_on for r in results)
