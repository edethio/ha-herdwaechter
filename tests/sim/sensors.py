"""Zero-export inverter and sensor reporting model (no Home Assistant).

Turns a true household load into the state change events Home Assistant
would see: a net grid power sensor, an inverter output sensor and optional
power meters of individual appliances, each with its own update cadence,
latency, change-only reporting and occasional ``unavailable`` periods.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from .household import DAY, Household, pv_potential

FloatArray = npt.NDArray[np.float64]

ROLE_GRID = 0
ROLE_GEN = 1
ROLE_SUB = 2


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Parameters of the metering setup."""

    seed: int = 11
    grid_interval_s: float = 3.0
    zero_export: bool = True
    pv_peak_w: float = 900.0
    inverter_limit_w: float = 800.0
    battery_wh: float = 2000.0
    battery_min_soc: float = 0.1
    inverter_ramp_w_s: float = 200.0
    control_delay_s: int = 2
    control_gain: float = 0.6
    gen_interval_s: float = 5.0
    gen_latency_s: float = 2.0
    meter_subtract: bool = True
    sub_interval_s: float = 10.0
    grid_outages_per_day: float = 0.7
    gen_unavailable_at_night: bool = True


@dataclass(frozen=True, slots=True)
class SensorEvents:
    """Merged, time ordered state change events."""

    ts: FloatArray
    role: npt.NDArray[np.int8]
    index: npt.NDArray[np.int8]
    value: FloatArray  # NaN = unavailable
    n_generation: int
    n_subtract: int
    grid_true: FloatArray
    inverter_out: FloatArray

    @property
    def count(self) -> int:
        """Return the number of events."""
        return int(self.ts.shape[0])


def simulate_inverter(
    load: FloatArray, pv: FloatArray, cfg: SensorConfig
) -> tuple[FloatArray, FloatArray]:
    """Simulate a zero-export inverter with battery; return (grid, inverter_out).

    The controller looks at the grid power with a delay, moves its output
    towards zero grid import with a limited ramp rate and is capped by the
    inverter limit and the available PV + battery power.
    """
    n = load.shape[0]
    out = np.zeros(n)
    grid = np.empty(n)
    if not cfg.zero_export:
        grid[:] = load
        return grid, out
    cap_wh = cfg.battery_wh
    soc_wh = 0.5 * cap_wh
    min_wh = cfg.battery_min_soc * cap_wh
    limit = cfg.inverter_limit_w
    ramp = cfg.inverter_ramp_w_s
    gain = cfg.control_gain
    delay = cfg.control_delay_s
    current = 0.0
    load_l = load.tolist()
    pv_l = pv.tolist()
    grid_l = [0.0] * n
    out_l = [0.0] * n
    for t in range(n):
        seen = grid_l[t - delay] if t >= delay else 0.0
        pv_t = pv_l[t]
        avail = pv_t + (limit if soc_wh > min_wh else 0.0)
        avail = min(avail, limit)
        desired = current + gain * (seen - 15.0)
        desired = max(desired, 0.0)
        desired = min(desired, avail)
        step = desired - current
        if step > ramp:
            step = ramp
        elif step < -ramp:
            step = -ramp
        current += step
        current = max(current, 0.0)
        out_l[t] = current
        grid_l[t] = load_l[t] - current
        # battery bookkeeping (Wh per second)
        if current > pv_t:
            soc_wh -= (current - pv_t) / 3600.0
        else:
            soc_wh = min(cap_wh, soc_wh + 0.95 * (pv_t - current) / 3600.0)
    grid[:] = grid_l
    out[:] = out_l
    return grid, out


def _change_only(ts: FloatArray, values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Drop reports that repeat the previous value (NaN aware)."""
    if ts.shape[0] == 0:
        return ts, values
    prev = np.concatenate(([np.inf], values[:-1]))
    same = (values == prev) | (np.isnan(values) & np.isnan(prev))
    keep = ~same
    return ts[keep], values[keep]


def periodic_reports(
    signal: FloatArray,
    interval: float,
    latency: float,
    rng: np.random.Generator,
    unavailable: npt.NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Sample ``signal`` every ``interval`` s and report only changes."""
    n = signal.shape[0]
    phase = rng.uniform(0, interval)
    ts = np.arange(phase + latency, n - 1, interval)
    ts = ts + rng.uniform(0.0, 0.3, ts.shape[0])
    idx = np.clip(np.floor(ts - latency).astype(np.int64), 0, n - 1)
    values = np.round(signal[idx])
    if unavailable is not None:
        values = np.where(unavailable[idx], np.nan, values)
    return _change_only(ts, values)


def _outage_mask(
    n: int, per_day: float, rng: np.random.Generator
) -> npt.NDArray[np.bool_]:
    mask = np.zeros(n, dtype=bool)
    count = int(rng.poisson(per_day * n / DAY))
    for _ in range(count):
        start = int(rng.uniform(0, n))
        dur = int(rng.choice([rng.uniform(15, 60), rng.uniform(60, 480)]))
        mask[start : start + dur] = True
    return mask


def generate_events(house: Household, cfg: SensorConfig) -> SensorEvents:
    """Produce the sensor events Home Assistant would receive."""
    rng = np.random.default_rng(cfg.seed)
    n = house.seconds
    pv = pv_potential(n, cfg.seed, cfg.pv_peak_w) if cfg.zero_export else np.zeros(n)
    grid_true, inv_out = simulate_inverter(house.total, pv, cfg)

    parts_ts: list[FloatArray] = []
    parts_role: list[npt.NDArray[np.int8]] = []
    parts_idx: list[npt.NDArray[np.int8]] = []
    parts_val: list[FloatArray] = []

    def push(ts: FloatArray, vals: FloatArray, role: int, index: int) -> None:
        parts_ts.append(ts)
        parts_val.append(vals)
        parts_role.append(np.full(ts.shape[0], role, dtype=np.int8))
        parts_idx.append(np.full(ts.shape[0], index, dtype=np.int8))

    grid_outage = _outage_mask(n, cfg.grid_outages_per_day, rng)
    ts, vals = periodic_reports(grid_true, cfg.grid_interval_s, 0.5, rng, grid_outage)
    push(ts, vals, ROLE_GRID, 0)

    n_gen = 0
    if cfg.zero_export:
        n_gen = 1
        unavailable = _outage_mask(n, 0.3, rng)
        if cfg.gen_unavailable_at_night:
            hour = (np.arange(n) % DAY) / 3600.0
            night_days = rng.random(math.ceil(n / DAY)) < 0.5
            night = np.repeat(night_days, DAY)[:n] & ((hour >= 22) | (hour < 6))
            unavailable |= night & (inv_out <= 0.5)
        ts, vals = periodic_reports(
            inv_out, cfg.gen_interval_s, cfg.gen_latency_s, rng, unavailable
        )
        push(ts, vals, ROLE_GEN, 0)

    n_sub = 0
    if cfg.meter_subtract:
        n_sub = 1
        ts, vals = periodic_reports(house.metered, cfg.sub_interval_s, 1.0, rng)
        push(ts, vals, ROLE_SUB, 0)

    all_ts = np.concatenate(parts_ts)
    order = np.argsort(all_ts, kind="stable")
    return SensorEvents(
        ts=all_ts[order],
        role=np.concatenate(parts_role)[order],
        index=np.concatenate(parts_idx)[order],
        value=np.concatenate(parts_val)[order],
        n_generation=n_gen,
        n_subtract=n_sub,
        grid_true=grid_true,
        inverter_out=inv_out,
    )
