#!/usr/bin/env python3
"""Run the household simulation and print an evaluation report.

Usage examples::

    python scripts/simulate.py                      # 30 days, reference scenario
    python scripts/simulate.py --days 10 --seed 3   # another household
    python scripts/simulate.py --grid-interval 10 --no-zero-export
    python scripts/simulate.py --plot sim.png       # needs matplotlib (optional)

The simulation uses the real detection engine of the integration (no Home
Assistant needed). Only numpy is required; matplotlib is optional and only
used for ``--plot``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.sim import (  # noqa: E402
    HouseholdConfig,
    RunResult,
    SensorConfig,
    UserConfig,
    build_scenario,
    run_simulation,
)
from tests.sim.household import DAY  # noqa: E402

REQUIREMENTS = {
    "precision": (">=", 0.90),
    "recall": (">=", 0.90),
    "median_latency_min": ("<=", 4.0),
    "false_alarms_per_day": ("<=", 0.2),
}


def _check(metrics: dict[str, float]) -> list[str]:
    lines = []
    for key, (op, limit) in REQUIREMENTS.items():
        value = metrics[key]
        ok = value >= limit if op == ">=" else value <= limit
        verdict = "PASS" if ok else "FAIL"
        lines.append(f"  {key:<22} {value:7.3f}  (required {op} {limit})  {verdict}")
    return lines


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"sessions {int(metrics['sessions']):3d}  "
        f"detected {int(metrics['detected']):3d}  "
        f"precision {metrics['precision']:.3f}  recall {metrics['recall']:.3f}  "
        f"latency {metrics['median_latency_min']:5.2f} min  "
        f"false alarms/day {metrics['false_alarms_per_day']:.3f}"
    )


def report(result: RunResult, baseline: RunResult | None, elapsed: float) -> str:
    """Return the text report."""
    house = result.house
    days = house.config.days
    engine = result.engine
    lines: list[str] = []
    lines.append("Herdwächter simulation report")
    lines.append("=" * 29)
    lines.append(f"days simulated        {days}")
    lines.append(
        f"cooktop zones (truth) {', '.join(f'{v:.0f} W' for v in house.hob_levels)}"
    )
    learned = ", ".join(f"{v:.0f} W" for v in engine.clusterer.levels()) or "-"
    lines.append(f"learned power levels  {learned}")
    lag = engine.calibrator.lag_s
    lines.append(f"estimated sensor lag  {'-' if lag is None else f'{lag:.1f} s'}")
    lines.append(f"edge threshold        {engine.calibrator.edge_threshold:.0f} W")
    lines.append(f"model weight          {engine.model_weight:.2f}")
    q = engine.quality
    if q.known:
        lines.append(
            f"holdout quality       precision {q.precision or 0:.3f}  "
            f"recall {q.recall or 0:.3f}  "
            f"Brier {q.brier or 0:.4f} (prior {q.brier_prior or 0:.4f})"
        )
    counts = engine.dataset.counts()
    lines.append(
        f"labels                {counts['positive']} positive / "
        f"{counts['negative']} negative "
        f"({counts['strong_negative']} strong)"
    )
    lines.append(
        f"virtual user          labelled {result.labels_pressed} sessions, "
        f"pressed false alarm {result.false_alarm_presses} times"
    )
    lines.append(f"appliance events      {house.events}")
    lines.append(f"run time              {elapsed:.0f} s")
    lines.append("")
    lines.append("Weekly (with learning)")
    for start in range(0, days, 7):
        m = result.metrics(start, min(start + 7, days)).as_dict()
        lines.append(f"  days {start:2d}-{min(start + 7, days) - 1:2d}: {_fmt(m)}")
    if baseline is not None:
        lines.append("")
        lines.append("Prior knowledge only (learning paused)")
        lines.append(f"  day 0      : {_fmt(baseline.metrics(0, 1).as_dict())}")
        lines.append(f"  all days   : {_fmt(baseline.metrics().as_dict())}")
    final_from = max(0, days - 7)
    final = result.metrics(final_from, days).as_dict()
    lines.append("")
    lines.append(
        f"Trained state (days {final_from}-{days - 1}) against the requirements"
    )
    lines.extend(_check(final))
    lines.append("")
    lines.append(f"All {days} days with learning: {_fmt(result.metrics().as_dict())}")
    false_alarms = [e for e in result.episodes if not e.true_positive]
    if false_alarms:
        lines.append("")
        lines.append("False alarms (appliances running at that time)")
        for e in false_alarms:
            active = house.active_at(e.start - 600, e.end)
            lines.append(
                f"  day {e.start / DAY:5.2f}  {e.start % DAY / 3600:5.2f} h  "
                f"{(e.end - e.start) / 60:5.1f} min  {', '.join(active) or '-'}"
            )
    return "\n".join(lines)


def plot(result: RunResult, path: Path) -> bool:
    """Plot the probability and the output around the busiest day."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    import numpy as np

    house = result.house
    day = max(
        range(house.config.days),
        key=lambda d: sum(
            1 for s in house.sessions if d * DAY <= s.start < (d + 1) * DAY
        ),
    )
    lo, hi = day * DAY, (day + 1) * DAY
    sel = (result.eval_ts >= lo) & (result.eval_ts < hi)
    hours = (result.eval_ts[sel] - lo) / 3600
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    seconds = np.arange(lo, hi)
    ax1.plot((seconds - lo) / 3600, house.total[lo:hi], lw=0.5, label="household load")
    for s in house.sessions:
        if lo <= s.start < hi:
            ax1.axvspan(
                (s.start - lo) / 3600,
                (s.end - lo) / 3600,
                alpha=0.2,
                color="tab:orange",
            )
    ax1.set_ylabel("W")
    ax1.legend(loc="upper left")
    ax2.plot(hours, result.probability[sel], label="probability")
    ax2.plot(hours, result.confidence[sel], label="confidence", alpha=0.6)
    ax2.fill_between(
        hours, 0, result.is_on[sel].astype(float), alpha=0.2, label="cooktop on"
    )
    ax2.set_xlabel(f"hour of day {day}")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return True


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the simulation and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-interval", type=float, default=3.0)
    parser.add_argument("--no-zero-export", action="store_true")
    parser.add_argument("--no-subtract", action="store_true")
    parser.add_argument("--label-share", type=float, default=0.5)
    parser.add_argument("--press-share", type=float, default=0.5)
    parser.add_argument(
        "--skip-baseline", action="store_true", help="skip the prior-only run"
    )
    parser.add_argument("--plot", type=Path, help="write a PNG plot (needs matplotlib)")
    parser.add_argument("--output", type=Path, help="also write the report to a file")
    args = parser.parse_args(argv)

    start = time.time()
    house, events = build_scenario(
        HouseholdConfig(days=args.days, seed=args.seed),
        SensorConfig(
            seed=args.seed + 4,
            grid_interval_s=args.grid_interval,
            zero_export=not args.no_zero_export,
            meter_subtract=not args.no_subtract,
        ),
    )
    user = UserConfig(
        label_share=args.label_share, false_alarm_press_share=args.press_share
    )

    def progress(fraction: float) -> None:
        print(
            f"\r  simulating ... {100 * fraction:5.1f} %",
            end="",
            file=sys.stderr,
            flush=True,
        )

    result = run_simulation(house, events, user, learning=True, progress=progress)
    baseline = None
    if not args.skip_baseline:
        baseline = run_simulation(
            house, events, user, learning=False, progress=progress
        )
    print(file=sys.stderr)
    text = report(result, baseline, time.time() - start)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.plot is not None:
        if plot(result, args.plot):
            print(f"\nplot written to {args.plot}")
        else:
            print("\nmatplotlib is not installed, no plot written")
    summary: dict[str, Any] = result.metrics(max(0, args.days - 7), args.days).as_dict()
    return (
        0
        if all(
            (summary[k] >= v) if op == ">=" else (summary[k] <= v)
            for k, (op, v) in REQUIREMENTS.items()
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
