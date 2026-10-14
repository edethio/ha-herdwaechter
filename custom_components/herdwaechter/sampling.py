"""Resampling of asynchronous sensor updates onto a fixed 1 s raster.

Pure Python + numpy, no Home Assistant imports.

Three channels are kept in fixed-size ring buffers:

* ``grid``: net grid power (import positive). Gaps are stored as NaN.
* ``gen``: sum of all generation sensors (inverter AC output, battery
  discharge, ...). Unknown contributions count as 0 W after a short hold.
* ``sub``: sum of all devices with their own power meter that are subtracted.

Values are held (sample-and-hold) until the next update. ``unavailable`` /
``unknown`` on the grid sensor produces gaps, never zeros. Optional sensors
are held for ``aux_hold_s`` seconds and then count as 0 W, because many
inverters report ``unavailable`` while idle at night; the number of affected
seconds is reported as a data quality metric.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import math
import statistics

import numpy as np
import numpy.typing as npt

from .const import AUX_HOLD_SECONDS, MAX_GRID_HOLD_SECONDS, RING_BUFFER_SECONDS

FloatArray = npt.NDArray[np.float64]

ROLE_GRID = "grid"
ROLE_GEN = "gen"
ROLE_SUB = "sub"

_INTERVAL_HISTORY = 64
_MAX_COUNTED_INTERVAL = 120.0


@dataclass(slots=True)
class _SensorState:
    """Last known value of a single sensor."""

    value: float | None = None
    last_valid: float | None = None
    updated: float | None = None
    unavailable_since: float | None = None


@dataclass(frozen=True, slots=True)
class WindowData:
    """A contiguous slice of the raster, oldest sample first."""

    end_ts: float
    grid: FloatArray
    gen: FloatArray
    sub: FloatArray
    aux_missing: npt.NDArray[np.bool_]

    @property
    def length(self) -> int:
        """Return the number of 1 s samples."""
        return int(self.grid.shape[0])


class RingBuffer:
    """Fixed-size float ring buffer on the 1 s raster."""

    def __init__(self, capacity: int, fill: float = math.nan) -> None:
        """Allocate the buffer."""
        self._data = np.full(capacity, fill, dtype=np.float64)
        self._pos = 0
        self._count = 0
        self.capacity = capacity

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return self._count

    def push_constant(self, value: float, count: int) -> None:
        """Append ``count`` copies of ``value``."""
        if count <= 0:
            return
        if count >= self.capacity:
            self._data[:] = value
            self._pos = 0
            self._count = self.capacity
            return
        end = self._pos + count
        if end <= self.capacity:
            self._data[self._pos : end] = value
        else:
            first = self.capacity - self._pos
            self._data[self._pos :] = value
            self._data[: count - first] = value
        self._pos = end % self.capacity
        self._count = min(self.capacity, self._count + count)

    def last(self, count: int) -> FloatArray:
        """Return the newest ``count`` samples in chronological order.

        Missing history (buffer not yet full) is padded with NaN at the start.
        """
        count = min(count, self.capacity)
        out = np.full(count, math.nan, dtype=np.float64)
        avail = min(count, self._count)
        if avail == 0:
            return out
        start = (self._pos - avail) % self.capacity
        if start + avail <= self.capacity:
            out[count - avail :] = self._data[start : start + avail]
        else:
            first = self.capacity - start
            out[count - avail : count - avail + first] = self._data[start:]
            out[count - avail + first :] = self._data[: avail - first]
        return out


class Resampler:
    """Collect sensor updates and expose them on a 1 s sample-and-hold raster."""

    def __init__(
        self,
        n_generation: int,
        n_subtract: int,
        capacity_s: int = RING_BUFFER_SECONDS,
        max_grid_hold_s: float = MAX_GRID_HOLD_SECONDS,
        aux_hold_s: float = AUX_HOLD_SECONDS,
    ) -> None:
        """Initialise empty buffers."""
        self.capacity_s = capacity_s
        self._max_grid_hold = max_grid_hold_s
        self._aux_hold = aux_hold_s
        self._grid = _SensorState()
        self._gen = [_SensorState() for _ in range(n_generation)]
        self._sub = [_SensorState() for _ in range(n_subtract)]
        self._buf_grid = RingBuffer(capacity_s)
        self._buf_gen = RingBuffer(capacity_s, 0.0)
        self._buf_sub = RingBuffer(capacity_s, 0.0)
        self._buf_missing = RingBuffer(capacity_s, 0.0)
        self._next_second: int | None = None
        self._grid_update_times: deque[float] = deque(maxlen=_INTERVAL_HISTORY)
        self._gen_update_times: list[deque[float]] = [
            deque(maxlen=_INTERVAL_HISTORY) for _ in range(n_generation)
        ]
        self._sub_update_times: list[deque[float]] = [
            deque(maxlen=_INTERVAL_HISTORY) for _ in range(n_subtract)
        ]

    # ------------------------------------------------------------------ input

    def update(self, role: str, index: int, ts: float, value: float | None) -> None:
        """Register a new value (``None`` = unavailable/unknown) of a sensor."""
        self._write_until(math.ceil(ts))
        state = self._state(role, index)
        if value is None or not math.isfinite(value):
            if state.unavailable_since is None:
                state.unavailable_since = ts
            state.value = None
        else:
            state.value = float(value)
            state.last_valid = state.value
            state.unavailable_since = None
        state.updated = ts
        if value is not None:
            self._record_update(role, index, ts)

    def report(self, role: str, index: int, ts: float) -> None:
        """Register an update that did not change the value (state reported)."""
        self._write_until(math.ceil(ts))
        state = self._state(role, index)
        if state.value is None:
            return
        state.updated = ts
        self._record_update(role, index, ts)

    def _record_update(self, role: str, index: int, ts: float) -> None:
        if role == ROLE_GRID:
            self._grid_update_times.append(ts)
        elif role == ROLE_GEN:
            self._gen_update_times[index].append(ts)
        else:
            self._sub_update_times[index].append(ts)

    def _state(self, role: str, index: int) -> _SensorState:
        if role == ROLE_GRID:
            return self._grid
        if role == ROLE_GEN:
            return self._gen[index]
        if role == ROLE_SUB:
            return self._sub[index]
        msg = f"unknown role {role}"
        raise ValueError(msg)

    # ----------------------------------------------------------------- raster

    def _held_grid(self, second: int) -> float:
        state = self._grid
        if state.value is None or state.updated is None:
            return math.nan
        if second - state.updated > self._max_grid_hold:
            return math.nan
        return state.value

    def _write_until(self, second_exclusive: int) -> None:
        """Write held values for all raster seconds before ``second_exclusive``."""
        if self._next_second is None:
            self._next_second = second_exclusive
            return
        count = second_exclusive - self._next_second
        if count <= 0:
            return
        start = self._next_second
        # Values can only change at update events, so a single held value is
        # valid for the whole span except for hold timeouts, which we split on.
        while count > 0:
            chunk = min(count, self.capacity_s)
            chunk = self._split_for_timeouts(start, chunk)
            grid = self._held_grid(start)
            gen, gen_missing = self._held_aux(self._gen, start)
            sub, sub_missing = self._held_aux(self._sub, start)
            self._buf_grid.push_constant(grid, chunk)
            self._buf_gen.push_constant(gen, chunk)
            self._buf_sub.push_constant(sub, chunk)
            self._buf_missing.push_constant(
                1.0 if (gen_missing or sub_missing) else 0.0, chunk
            )
            start += chunk
            count -= chunk
        self._next_second = second_exclusive

    def _held_aux(self, states: list[_SensorState], second: int) -> tuple[float, bool]:
        """Sum auxiliary sensors at ``second``.

        An unavailable sensor keeps its last valid value for ``aux_hold_s``
        seconds, afterwards it contributes 0 W and the second is flagged.
        """
        total = 0.0
        missing = False
        for state in states:
            if state.value is not None:
                total += state.value
            elif state.unavailable_since is not None:
                if (
                    state.last_valid is not None
                    and second - state.unavailable_since < self._aux_hold
                ):
                    total += state.last_valid
                else:
                    missing = True
        return total, missing

    def _split_for_timeouts(self, start: int, chunk: int) -> int:
        """Shorten ``chunk`` so that it does not cross a hold timeout boundary."""
        # first raster second at which a held value changes its meaning
        boundaries: list[int] = []
        grid = self._grid
        if grid.value is not None and grid.updated is not None:
            # stale once second - updated > max_hold
            boundaries.append(math.floor(grid.updated + self._max_grid_hold) + 1)
        boundaries.extend(
            # held while second - unavailable_since < aux_hold
            math.ceil(state.unavailable_since + self._aux_hold)
            for state in (*self._gen, *self._sub)
            if state.value is None and state.unavailable_since is not None
        )
        limit = chunk
        for boundary in boundaries:
            if start < boundary < start + limit:
                limit = boundary - start
        return max(1, limit)

    def snapshot(self, now: float, seconds: int) -> WindowData:
        """Advance the raster to ``now`` and return the newest ``seconds`` samples."""
        self._write_until(math.floor(now) + 1)
        return WindowData(
            end_ts=float(math.floor(now)),
            grid=self._buf_grid.last(seconds),
            gen=self._buf_gen.last(seconds),
            sub=self._buf_sub.last(seconds),
            aux_missing=self._buf_missing.last(seconds) > 0.5,
        )

    # ---------------------------------------------------------------- quality

    @staticmethod
    def _median_interval(times: deque[float]) -> float | None:
        if len(times) < 6:
            return None
        values = list(times)
        diffs = [
            b - a
            for a, b in itertools.pairwise(values)
            if 0.05 < b - a <= _MAX_COUNTED_INTERVAL
        ]
        if len(diffs) < 5:
            return None
        return float(statistics.median(diffs))

    def effective_update_interval(self) -> float | None:
        """Estimate the grid sensor update interval (median of recent intervals)."""
        return self._median_interval(self._grid_update_times)

    def generation_update_interval(self) -> float | None:
        """Estimate the slowest update interval of the auxiliary sensors.

        Covers generation sensors and subtracted power meters: their late
        reports cause the transients that edge detection has to ignore.
        """
        values = [
            v
            for v in (
                self._median_interval(t)
                for t in (*self._gen_update_times, *self._sub_update_times)
            )
            if v is not None
        ]
        return max(values) if values else None

    def seconds_since_grid_update(self, now: float) -> float | None:
        """Return the age of the newest grid value."""
        if self._grid.updated is None:
            return None
        return now - self._grid.updated
