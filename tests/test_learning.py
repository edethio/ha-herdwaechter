"""Tests for the label data set, power levels, PU rules and model mixing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from custom_components.herdwaechter.learning import (
    MAX_MODEL_WEIGHT,
    WEAK_NEGATIVE_WEIGHT,
    LabelDataset,
    LabelPolicy,
    LabelSource,
    PowerLevelClusterer,
    QualityReport,
    evaluate_holdout,
    is_holdout_group,
    maturity,
    mixing_weight,
)

F = 4


def _vec(v: float) -> np.ndarray:
    return np.full(F, v)


def test_dataset_add_replace_and_roundtrip() -> None:
    ds = LabelDataset(F, max_size=100)
    ds.add(_vec(1), 1, 1.0, 10.0, LabelSource.SWITCH, 1)
    ds.add(_vec(2), 0, 0.2, 20.0, LabelSource.WEAK_NEGATIVE, 2)
    assert len(ds) == 2
    # same timestamp: strong label replaces
    ds.add(_vec(3), 0, 3.0, 20.0, LabelSource.FALSE_ALARM, 3)
    assert len(ds) == 2
    assert ds.source[1] == LabelSource.FALSE_ALARM
    # weak label never overrides a strong one
    ds.add(_vec(4), 0, 0.2, 10.0, LabelSource.WEAK_NEGATIVE, 4)
    assert ds.y[0] == 1
    counts = ds.counts()
    assert counts["positive"] == 1
    assert counts["negative"] == 1
    assert counts["strong_negative"] == 1
    restored = LabelDataset.from_dict(json.loads(json.dumps(ds.to_dict())), F, 100)
    np.testing.assert_allclose(restored.x, ds.x)
    np.testing.assert_array_equal(restored.y, ds.y)
    np.testing.assert_array_equal(restored.group, ds.group)
    assert len(LabelDataset.from_dict(ds.to_dict(), F + 1)) == 0
    assert len(LabelDataset.from_dict({"n": 2, "n_features": F}, F)) == 0
    ds.clear()
    assert len(ds) == 0


def test_dataset_thinning_keeps_recent_and_strong() -> None:
    ds = LabelDataset(F, max_size=200)
    for i in range(400):
        src = LabelSource.WEAK_NEGATIVE if i % 2 else LabelSource.SWITCH
        ds.add(_vec(i), 0 if i % 2 else 1, 1.0, float(i), src, i)
    assert len(ds) <= 200
    ts = ds.ts
    # the newest entries are all still present
    assert set(range(350, 400)) <= set(ts.astype(int).tolist())
    # older history is thinned, not dropped entirely
    assert np.min(ts) < 100
    # weak negatives were thinned first
    assert ds.counts()["positive"] > ds.counts()["negative"]


def test_holdout_is_deterministic_and_about_twenty_percent() -> None:
    flags = [is_holdout_group(g) for g in range(5000)]
    assert 0.15 < float(np.mean(flags)) < 0.25
    assert flags == [is_holdout_group(g) for g in range(5000)]
    ds = LabelDataset(F)
    for g in range(50):
        for k in range(3):
            ds.add(_vec(g), 1, 1.0, float(g * 10 + k), LabelSource.SWITCH, g)
    mask = ds.holdout_mask()
    # all windows of one group are on the same side of the split
    for g in range(50):
        assert len(set(mask[ds.group == g].tolist())) == 1
    assert LabelDataset(F).holdout_mask().shape[0] == 0


def test_clusterer_learns_power_levels() -> None:
    rng = np.random.default_rng(0)
    cl = PowerLevelClusterer()
    for level in (1180.0, 1820.0, 2300.0):
        for _ in range(60):
            cl.add(level * rng.uniform(0.97, 1.03))
    cl.add(3000.0)  # a stray sum of two zones
    levels = cl.levels()
    assert levels.shape[0] == 3
    np.testing.assert_allclose(levels, [1180, 1820, 2300], rtol=0.02)
    # negative evidence suppresses a level
    for _ in range(40):
        cl.add_negative(1180.0)
    assert cl.levels().shape[0] == 2
    restored = PowerLevelClusterer.from_dict(json.loads(json.dumps(cl.to_dict())))
    np.testing.assert_allclose(restored.levels(), cl.levels(), rtol=1e-3)
    assert (
        PowerLevelClusterer.from_dict({"clusters": [["x", 1, 2], [0, 1, 1]]})
        .levels()
        .shape[0]
        == 0
    )
    cl.add(-5.0, 0.0)
    cl.clear()
    assert cl.levels().shape[0] == 0


def test_clusterer_limits_number_of_clusters() -> None:
    cl = PowerLevelClusterer(max_clusters=3)
    for h in (1000.0, 1500.0, 2200.0, 3300.0, 5000.0):
        cl.add(h, 20.0)
    assert len(cl.to_dict()["clusters"]) == 3


def test_pu_label_policy() -> None:
    policy = LabelPolicy(neutral_after_label=2, weak_negative_every=3)
    # label on with activity -> positive; without activity -> nothing
    d = policy.decide(0.0, True, 7, LabelSource.SWITCH, n_edges=4, output_on=False)
    assert d is not None
    assert (d.label, d.source, d.group) == (1, LabelSource.SWITCH, 7)
    assert policy.decide(1.0, True, 7, LabelSource.SWITCH, 0, False) is None
    # label switched off -> the next two evaluations are neutral
    assert policy.decide(2.0, False, 0, LabelSource.SWITCH, 0, False) is None
    assert policy.decide(3.0, False, 0, LabelSource.SWITCH, 0, False) is None
    # activity without label is never labelled automatically
    for _ in range(10):
        assert policy.decide(4.0, False, 0, LabelSource.SWITCH, 5, False) is None
    # quiet windows become sampled weak negatives
    decisions = [
        policy.decide(3600.0 * 5, False, 0, LabelSource.SWITCH, 0, False)
        for _ in range(6)
    ]
    weak = [d for d in decisions if d is not None]
    assert len(weak) == 2
    assert all(d.label == 0 and d.weight == WEAK_NEGATIVE_WEIGHT for d in weak)
    assert weak[0].group == 5
    # no weak negatives while the output is on
    assert all(
        policy.decide(0.0, False, 0, LabelSource.SWITCH, 0, True) is None
        for _ in range(6)
    )


def test_quality_and_mixing_weight() -> None:
    y = np.r_[np.ones(6), np.zeros(6)]
    good = evaluate_holdout(
        np.r_[np.full(6, 0.9), np.full(6, 0.1)],
        np.full(12, 0.5),
        y,
        np.ones(12),
    )
    assert good.known
    assert good.precision == 1.0
    assert good.recall == 1.0
    assert good.brier == pytest.approx(0.01)
    assert good.brier_prior == pytest.approx(0.25)
    few = evaluate_holdout(
        np.array([0.9]), np.array([0.5]), np.array([1.0]), np.ones(1)
    )
    assert not few.known
    assert not evaluate_holdout(
        np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    ).known
    restored = QualityReport.from_dict(good.to_dict())
    assert restored == good

    assert maturity(0, 10) == 0.0
    assert mixing_weight(0, 10, good) == 0.0
    w_few = mixing_weight(2, 10, good)
    w_many = mixing_weight(20, 50, good)
    assert 0 < w_few < w_many <= MAX_MODEL_WEIGHT
    # unknown quality caps the weight at half
    assert mixing_weight(20, 50, QualityReport()) == pytest.approx(
        0.5 * MAX_MODEL_WEIGHT * maturity(20, 50)
    )
    bad = evaluate_holdout(
        np.r_[np.full(6, 0.4), np.full(6, 0.6)], np.full(12, 0.5), y, np.ones(12)
    )
    assert mixing_weight(20, 50, bad) < 0.5 * w_many


def test_rare_label_categories_stay_in_training() -> None:
    ds = LabelDataset(F)
    for g in range(40):
        ds.add(_vec(g), 1, 1.0, float(g), LabelSource.SWITCH, g)
    # a single false alarm group must never be held out
    for k in range(10):
        ds.add(_vec(-1), 0, 3.0, 1000.0 + k, LabelSource.FALSE_ALARM, 4)
    mask = ds.holdout_mask()
    assert not np.any(mask[ds.source == LabelSource.FALSE_ALARM])
    assert np.any(mask[ds.source == LabelSource.SWITCH])
