"""Synthetic household load generator (1 s resolution, no Home Assistant).

Produces the true household consumption plus ground truth for cooking
sessions. Everything is random but reproducible from a seed. All values are
synthetic; nothing is derived from a real household.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DAY = 86400


def moving_average(x: FloatArray, width: int) -> FloatArray:
    """Centred moving average via cumulative sums (O(n))."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    half = width // 2
    idx = np.arange(x.shape[0])
    lo = np.clip(idx - half, 0, x.shape[0])
    hi = np.clip(idx + half + 1, 0, x.shape[0])
    out: FloatArray = (c[hi] - c[lo]) / (hi - lo)
    return out


@dataclass(frozen=True, slots=True)
class HouseholdConfig:
    """Parameters of the simulated household."""

    days: int = 30
    seed: int = 7
    # power of the cooktop zones (W); chosen randomly per seed if empty
    hob_levels: tuple[float, ...] = ()
    microwave_w: float = 0.0  # chosen randomly per seed if 0
    kettle_w: float = 0.0
    cooking_probability: tuple[float, float, float] = (0.15, 0.45, 0.85)
    disturbers: bool = True
    big_loads: bool = True


@dataclass(frozen=True, slots=True)
class Session:
    """Ground truth of one cooking session (seconds since simulation start)."""

    start: float
    end: float
    zones: int
    heat_up_s: float = 0.0
    period_s: float = 0.0
    duty: float = 0.0


@dataclass(slots=True)
class Household:
    """Generated loads."""

    config: HouseholdConfig
    total: FloatArray
    metered: FloatArray  # loads of devices with their own power meter
    sessions: list[Session]
    hob_levels: tuple[float, ...]
    events: dict[str, int] = field(default_factory=dict)
    activities: list[tuple[str, float, float]] = field(default_factory=list)

    def active_at(self, start: float, end: float) -> list[str]:
        """Return the appliances that were running during [start, end]."""
        return sorted(
            {name for name, a, b in self.activities if b >= start and a <= end}
        )

    @property
    def seconds(self) -> int:
        """Return the simulated duration."""
        return int(self.total.shape[0])


class _Builder:
    def __init__(self, cfg: HouseholdConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = cfg.days * DAY
        self.total = np.zeros(self.n)
        self.metered = np.zeros(self.n)
        self.sessions: list[Session] = []
        self.events: dict[str, int] = {}
        self.activities: list[tuple[str, float, float]] = []

    # --------------------------------------------------------------- helpers

    def add(
        self, start: float, duration: float, power: float, metered: bool = False
    ) -> None:
        a = max(0, int(start))
        b = min(self.n, int(start + duration))
        if b <= a:
            return
        self.total[a:b] += power
        if metered:
            self.metered[a:b] += power

    def count(self, name: str, start: float = 0.0, end: float = 0.0) -> None:
        self.events[name] = self.events.get(name, 0) + 1
        if end > start:
            self.activities.append((name, start, end))

    def daytime(self, day: int, lo_h: float, hi_h: float) -> float:
        return day * DAY + self.rng.uniform(lo_h, hi_h) * 3600

    # ----------------------------------------------------------- components

    def base_load(self) -> None:
        rng = self.rng
        t = np.arange(self.n)
        hour = (t % DAY) / 3600.0
        # daily pattern: low at night, higher in the evening
        pattern = (
            140
            + 60 * np.exp(-((hour - 20.0) ** 2) / 8.0)
            + 30 * np.exp(-((hour - 7.5) ** 2) / 2.0)
        )
        walk = np.cumsum(rng.normal(0, 0.4, self.n))
        walk -= moving_average(walk, 3601)
        self.total += pattern + np.clip(walk, -60, 60) + rng.normal(0, 6, self.n)
        # fridge compressor
        t0 = 0.0
        while t0 < self.n:
            on = rng.uniform(600, 900)
            off = rng.uniform(1300, 1900)
            self.add(t0, on, rng.uniform(85, 110))
            self.add(t0, 2, 250)  # inrush
            t0 += on + off
        # small random appliances (TV, lights, computer): below edge threshold
        n_small = int(self.cfg.days * 12)
        for _ in range(n_small):
            day = int(rng.integers(0, self.cfg.days))
            start = self.daytime(day, 6.0, 23.5)
            self.add(start, rng.uniform(300, 7200), rng.uniform(20, 250))

    def zone_profile(
        self, start: float, end: float, power: float
    ) -> tuple[float, float, float]:
        """One cooktop zone: continuous heat-up, then regulator cycling."""
        rng = self.rng
        # continuous heat-up phase: median 2 min, 0.5-8 min
        heat_up = float(np.clip(rng.lognormal(math.log(120.0), 0.6), 30.0, 480.0))
        power *= rng.uniform(0.985, 1.015)
        self.add(start, min(heat_up, end - start), power)
        t = start + heat_up
        period = rng.uniform(20, 70)
        duty = rng.uniform(0.25, 0.75)
        change_at = t + rng.uniform(0.3, 0.8) * max(0.0, end - t)
        changed = rng.random() < 0.4
        while t < end:
            if changed and t >= change_at:
                duty = rng.uniform(0.2, 0.7)
                changed = False
            p = period * rng.uniform(0.92, 1.08)
            on = max(3.0, p * duty * rng.uniform(0.92, 1.08))
            on = min(on, end - t)
            # the element draws slightly less when hot
            self.add(t, on, power * rng.uniform(0.99, 1.01))
            t += p
        return heat_up, period, duty

    def cooking(self) -> None:
        rng = self.rng
        levels = self.cfg.hob_levels or (
            float(rng.choice([1000, 1100, 1200])),
            float(rng.choice([1700, 1800])),
            float(rng.choice([2100, 2200, 2300])),
        )
        self.levels = levels
        slots = ((7.0, 8.5, 10, 15), (11.75, 13.0, 15, 40), (17.5, 19.5, 20, 60))
        for day in range(self.cfg.days):
            for prob, (lo, hi, dmin, dmax) in zip(
                self.cfg.cooking_probability, slots, strict=True
            ):
                if rng.random() >= prob:
                    continue
                start = self.daytime(day, lo, hi)
                duration = rng.uniform(dmin, dmax) * 60
                n_zones = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
                zone_powers = rng.choice(levels, size=n_zones, replace=False)
                end = start
                heat_up = period = duty = 0.0
                for i, zp in enumerate(zone_powers):
                    z_start = (
                        start if i == 0 else start + rng.uniform(0, 0.4) * duration
                    )
                    z_end = start + duration * (
                        1.0 if i == 0 else rng.uniform(0.6, 1.0)
                    )
                    if z_end - z_start < 300:
                        continue
                    h, per, dut = self.zone_profile(z_start, z_end, float(zp))
                    if i == 0:
                        heat_up, period, duty = h, per, dut
                    end = max(end, z_end)
                self.sessions.append(
                    Session(start, end, n_zones, heat_up, period, duty)
                )
                self.count("cooking", start, end)

    def disturbers(self) -> None:
        rng = self.rng
        cfg = self.cfg
        kettle = cfg.kettle_w or float(rng.uniform(2000, 2200))
        micro = cfg.microwave_w or self.distinct_power(1150.0, 1600.0)
        for day in range(cfg.days):
            # kettle: single short block
            for _ in range(int(rng.integers(1, 4))):
                start = self.daytime(day, 6.5, 22.0)
                dur = rng.uniform(120, 240)
                self.add(start, dur, kettle)
                self.count("kettle", start, start + dur)
            # hair dryer: block, sometimes switching level
            if rng.random() < 0.4:
                start = self.daytime(day, 6.5, 8.0)
                dur = rng.uniform(180, 480)
                self.add(start, dur, 1000)
                if rng.random() < 0.5:
                    self.add(start + dur * 0.3, dur * 0.4, 800)
                self.count("hair_dryer", start, start + dur)
            # toaster
            if rng.random() < 0.3:
                start = self.daytime(day, 6.5, 9.0)
                dur = rng.uniform(120, 180)
                self.add(start, dur, 850)
                self.count("toaster", start, start + dur)
            # microwave: mostly continuous, sometimes cycling at reduced power
            for _ in range(int(rng.poisson(0.9))):
                start = self.daytime(day, 7.0, 21.0)
                if rng.random() < 0.65:
                    dur = rng.uniform(60, 240)
                    self.add(start, dur, micro)
                    self.count("microwave", start, start + dur)
                else:
                    end = start + rng.uniform(180, 480)
                    period = rng.uniform(12, 22)
                    duty = rng.uniform(0.3, 0.6)
                    self.add(start, end - start, 100)  # fan, lamp, turntable
                    t = start
                    while t < end:
                        self.add(t, period * duty, micro - 100)
                        t += period
                    self.count("microwave_cycling", start, end)
            # washing machine
            if rng.random() < 0.4:
                start = self.daytime(day, 8.0, 18.0)
                self.count("washer", start, start + self.washer(start, metered=True))
            # dish washer
            if rng.random() < 0.6:
                start = self.daytime(day, 13.0, 21.0)
                self.count(
                    "dishwasher", start, start + self.dishwasher(start, metered=True)
                )

    def distinct_power(self, lo: float, hi: float, margin: float = 0.12) -> float:
        """Draw a power in [lo, hi] that differs from every cooktop zone.

        The microwave magnetron step (power minus the 100 W fan) is kept at
        least ``margin`` away from all zone powers ("different step heights").
        """
        for _ in range(200):
            power = float(self.rng.uniform(lo, hi))
            step = power - 100.0
            if all(abs(step - z) > margin * z for z in self.levels):
                return power
        return hi

    def washer(self, start: float, metered: bool) -> float:
        rng = self.rng
        t = start + rng.uniform(120, 300)
        heat = rng.uniform(720, 1200)
        # heater with a few thermostat interruptions at the end
        self.add(t, heat, 2000, metered)
        for _ in range(int(rng.integers(0, 3))):
            gap_at = t + heat + rng.uniform(60, 600)
            self.add(gap_at, rng.uniform(20, 60), 2000, metered)
        # drum motor: tumbling pattern (small load)
        total = rng.uniform(5400, 9000)
        tt = start
        while tt < start + total:
            self.add(tt, 12, rng.uniform(150, 300), metered)
            tt += 15
        self.add(start + total - 600, 480, rng.uniform(300, 500), metered)
        return float(total)

    def dishwasher(self, start: float, metered: bool) -> float:
        rng = self.rng
        total = rng.uniform(5400, 8000)
        self.add(start, total, 70, metered)
        for phase in (rng.uniform(300, 900), rng.uniform(3600, 5000)):
            self.add(start + phase, rng.uniform(600, 900), 2050, metered)
        return float(total)

    def big_loads(self) -> None:
        rng = self.rng
        for day in range(self.cfg.days):
            # instantaneous water heater: large, modulating blocks
            for _ in range(int(rng.poisson(1.5))):
                start = self.daytime(day, 6.0, 23.0)
                dur = rng.uniform(60, 420)
                level = rng.uniform(6000, 12000)
                t = start
                while t < start + dur:
                    step = rng.uniform(5, 20)
                    self.add(t, step, level * rng.uniform(0.85, 1.15))
                    t += step
                self.count("water_heater", start, start + dur)
            # heat pump: big but smooth (soft start ramp)
            for _ in range(int(rng.integers(1, 4))):
                start = self.daytime(day, 0.0, 22.0)
                dur = rng.uniform(1800, 5400)
                level = rng.uniform(1500, 2500)
                ramp = np.linspace(0, level, 90)
                a = int(start)
                b = min(self.n, a + 90)
                self.total[a:b] += ramp[: b - a]
                self.add(start + 90, dur, level)
                c = int(start + 90 + dur)
                d = min(self.n, c + 60)
                if d > c:
                    self.total[c:d] += np.linspace(level, 0, 60)[: d - c]
                self.count("heat_pump", start, start + dur + 150)
            # wallbox: rare long charging sessions with a ramp
            if rng.random() < 0.15:
                start = self.daytime(day, 17.0, 22.0)
                a = int(start)
                b = min(self.n, a + 20)
                self.total[a:b] += np.linspace(0, 11000, 20)[: b - a]
                dur = rng.uniform(3600, 10800)
                self.add(start + 20, dur, 11000)
                self.count("wallbox", start, start + dur)


def generate_household(cfg: HouseholdConfig) -> Household:
    """Generate a household according to ``cfg``."""
    builder = _Builder(cfg)
    builder.base_load()
    builder.cooking()
    if cfg.disturbers:
        builder.disturbers()
    if cfg.big_loads:
        builder.big_loads()
    total = np.maximum(builder.total, 0.0)
    return Household(
        config=cfg,
        total=total,
        metered=builder.metered,
        sessions=sorted(builder.sessions, key=lambda s: s.start),
        hob_levels=builder.levels,
        events=builder.events,
        activities=builder.activities,
    )


def pv_potential(n_seconds: int, seed: int, peak_w: float = 900.0) -> FloatArray:
    """Return the available PV power per second (sunny and cloudy days)."""
    rng = np.random.default_rng(seed + 1000)
    t = np.arange(n_seconds)
    hour = (t % DAY) / 3600.0
    sun = np.clip(np.sin(math.pi * (hour - 6.5) / 13.0), 0.0, None)
    days = math.ceil(n_seconds / DAY)
    day_factor = np.repeat(rng.uniform(0.2, 1.0, days), DAY)[:n_seconds]
    clouds = np.cumsum(rng.normal(0, 0.01, n_seconds))
    clouds = 0.75 + 0.25 * np.tanh(clouds - moving_average(clouds, 1801))
    out: FloatArray = peak_w * sun * day_factor * clouds
    return out
