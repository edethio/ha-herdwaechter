"""Edge detection, edge pairing, periodicity and the feature vector.

All functions are pure, vectorised with numpy and independent of Home
Assistant. Times are expressed as sample indices on the 1 s raster.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FEATURE_NAMES: tuple[str, ...] = (
    "n_on",
    "n_off",
    "height_median",
    "height_cv",
    "level_match_frac",
    "interval_cv",
    "period_s",
    "period_strength",
    "on_median_s",
    "off_median_s",
    "duty_cycle",
    "n_cycles",
    "baseline_rise",
    "longest_block_s",
    "tod_sin",
    "tod_cos",
    "height_max",
    "activity_minutes",
    "n_equal_cycles",
    "zone_height",
    "heatup_cycles",
    "peak_above_base",
    "levels_known",
    "zone_heatup_seen",
)
N_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

PAIR_TOLERANCE = 0.20
GROUP_TOLERANCE = 0.15
LEVEL_TOLERANCE = 0.10
LEVEL_TOLERANCE_MIN_W = 80.0
MAX_ON_DURATION_S = 900
PERIOD_MIN_S = 8
PERIOD_MAX_S = 120
NO_INTERVAL_CV = 2.0
HEATUP_MIN_S = 45.0


@dataclass(frozen=True, slots=True)
class Edges:
    """Detected edges: sample index and signed height in W."""

    idx: FloatArray
    height: FloatArray

    @property
    def count(self) -> int:
        """Return the number of edges."""
        return int(self.idx.shape[0])

    @classmethod
    def empty(cls) -> Edges:
        """Return an empty edge set."""
        return cls(np.empty(0), np.empty(0))


@dataclass(frozen=True, slots=True)
class Cycles:
    """Paired on/off edges: on index, off index, mean height."""

    on: FloatArray
    off: FloatArray
    height: FloatArray

    @property
    def count(self) -> int:
        """Return the number of cycles."""
        return int(self.on.shape[0])

    @property
    def on_duration(self) -> FloatArray:
        """Return the on durations in s."""
        return self.off - self.on


@dataclass(frozen=True, slots=True)
class WindowAnalysis:
    """Result of analysing one window."""

    features: FloatArray
    edges: Edges
    cycles: Cycles
    open_on: FloatArray
    period_s: float
    period_strength: float


@contextmanager
def _quiet_nan_warnings() -> Iterator[None]:
    """Silence numpy's 'all-NaN slice' warnings; NaN results are handled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


# --------------------------------------------------------------------- signal


def reconstruct_residual(
    grid: FloatArray, gen: FloatArray, sub: FloatArray, lag: int
) -> FloatArray:
    """Return ``grid[t] + gen[t + lag] - sub[t]``.

    Samples for which the shifted generation value is not available are NaN.
    """
    n = grid.shape[0]
    shifted = np.full(n, math.nan)
    if lag >= 0:
        shifted[: n - lag] = gen[lag:]
    else:
        shifted[-lag:] = gen[: n + lag]
    residual: FloatArray = grid + shifted - sub
    return residual


def rolling_median(x: FloatArray, width: int) -> FloatArray:
    """Centred rolling median; NaN where the window touches a gap or the border."""
    if width <= 1 or x.shape[0] < width:
        return x.copy()
    half = width // 2
    out = np.full_like(x, math.nan)
    view = sliding_window_view(x, width)
    med = np.sort(view, axis=1)[:, half] if width % 2 else np.median(view, axis=1)
    med[np.isnan(view).any(axis=1)] = math.nan
    out[half : x.shape[0] - (width - 1 - half)] = med
    return out


def refine_heights(xs: FloatArray, edges: Edges, settle: int) -> Edges:
    """Re-measure edge heights on the settled plateaus around each edge.

    The height found by :func:`detect_edges` is the largest step within the
    span. With zero export the inverter may already regulate within that
    span, which distorts the height by up to a few hundred watts. Here the
    height is measured as the median of the settled part after the edge
    (second half, up to ``settle`` seconds plus margin, never across the next
    edge) minus the median before the edge.
    """
    if edges.count == 0:
        return edges
    n = xs.shape[0]
    pos = np.round(edges.idx).astype(np.int64)
    out = edges.height.copy()
    for k in range(edges.count):
        t = int(pos[k])
        prev_t = int(pos[k - 1]) if k > 0 else -(10**9)
        next_t = int(pos[k + 1]) if k + 1 < edges.count else 10**9
        b0, b1 = max(prev_t + 2, t - 10, 0), t - 1
        a0, a1 = t + 1, min(next_t - 1, t + settle + 6, n)
        if b1 - b0 < 2 or a1 - a0 < 2:
            continue
        before_seg = xs[b0:b1]
        after_seg = xs[a0:a1]
        after_seg = after_seg[after_seg.shape[0] // 2 :]
        before_seg = before_seg[np.isfinite(before_seg)]
        after_seg = after_seg[np.isfinite(after_seg)]
        if before_seg.shape[0] == 0 or after_seg.shape[0] == 0:
            continue
        height = float(np.median(after_seg) - np.median(before_seg))
        if height * out[k] > 0 and abs(height) >= 0.5 * abs(out[k]):
            out[k] = height
    return Edges(edges.idx, out)


def detect_edges(
    x: FloatArray,
    threshold: float,
    span: int = 3,
    smooth: int = 3,
    settle: int | None = None,
) -> Edges:
    """Detect steps larger than ``threshold`` that complete within ``span`` samples.

    ``x`` is smoothed with a rolling median of ``smooth`` samples first, so
    that short transients (for example while an inverter regulates) do not
    show up as edges. Consecutive samples exceeding the threshold with the
    same sign form one edge; its height is the largest step in that run.
    """
    xs = rolling_median(x, smooth)
    n = xs.shape[0]
    if n <= span:
        return Edges.empty()
    d = xs[span:] - xs[:-span]
    with np.errstate(invalid="ignore"):
        up = d >= threshold
        down = d <= -threshold
    sign = up.astype(np.int8) - down.astype(np.int8)
    active = np.flatnonzero(sign)
    if active.size == 0:
        return Edges.empty()
    # split active samples into runs: new run on index gap > 2 or sign change
    breaks = np.flatnonzero((np.diff(active) > 3) | (np.diff(sign[active]) != 0))
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [active.size]))
    idx_out = np.empty(starts.size)
    h_out = np.empty(starts.size)
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        run = active[s:e]
        vals = d[run]
        k = int(np.argmax(np.abs(vals)))
        h_out[i] = vals[k]
        idx_out[i] = 0.5 * (run[0] + run[-1]) + 0.5 * span + 0.5
    edges = Edges(idx_out, h_out)
    if settle is not None:
        edges = refine_heights(xs, edges, settle)
    return edges


def remove_generation_artifacts(
    edges: Edges,
    generation: FloatArray,
    span: int,
    smooth: int,
    max_gap_s: float,
    tolerance: float = 0.3,
) -> Edges:
    """Drop residual edges that are caused by a late auxiliary report.

    ``generation`` is the non-grid part of the residual, i.e. the shifted
    generation minus the subtracted power meters. With zero export, the
    inverter follows every load step within seconds; the grid sensor shows
    that regulation at once, the (slower) generation sensor only with its
    next report. Likewise a separately metered appliance appears in the grid
    power immediately but in its own meter only with the next report. Until
    then the residual load shows a dip or spike that ends exactly when the
    late report arrives. A residual edge that coincides with an equally large
    change of the auxiliary signal (same sign) therefore cannot be a load
    step; it is removed together with its opposite partner edge that opened
    the transient.
    """
    if (
        edges.count == 0
        or generation.shape[0] == 0
        or not np.any(np.isfinite(generation))
    ):
        return edges
    gs = rolling_median(np.nan_to_num(generation, nan=0.0), smooth)
    dg = np.zeros_like(gs)
    dg[span:] = gs[span:] - gs[:-span]
    dg = np.nan_to_num(dg)
    keep = np.ones(edges.count, dtype=bool)
    idx = edges.idx
    h = edges.height
    n = dg.shape[0]
    for i in range(edges.count):
        lo = max(0, int(idx[i]) - 2)
        hi = min(n, int(idx[i]) + span + 3)
        if hi <= lo:
            continue
        seg = dg[lo:hi]
        j = int(np.argmax(seg * np.sign(h[i])))
        change = float(seg[j])
        if change * h[i] <= 0 or abs(change - h[i]) > tolerance * abs(h[i]):
            continue
        keep[i] = False
        # partner: opposite edge of similar height shortly before or after
        best = -1
        best_dt = max_gap_s + 1.0
        for k in range(edges.count):
            if k == i or not keep[k]:
                continue
            dt = abs(idx[k] - idx[i])
            if (
                dt <= max_gap_s
                and h[k] * h[i] < 0
                and (abs(abs(h[k]) - abs(h[i])) <= tolerance * abs(h[i]))
                and dt < best_dt
            ):
                best = k
                best_dt = dt
        if best >= 0:
            keep[best] = False
    if np.all(keep):
        return edges
    return Edges(idx[keep], h[keep])


# -------------------------------------------------------------------- pairing


def pair_edges(
    edges: Edges,
    tolerance: float = PAIR_TOLERANCE,
    max_on_s: float = MAX_ON_DURATION_S,
) -> tuple[Cycles, FloatArray]:
    """Pair rising edges with later falling edges of similar height (±tolerance).

    Returns the cycles and the indices of rising edges that are still open.
    """
    open_t: list[float] = []
    open_h: list[float] = []
    on: list[float] = []
    off: list[float] = []
    height: list[float] = []
    for t, h in zip(edges.idx.tolist(), edges.height.tolist(), strict=True):
        if h > 0:
            open_t.append(t)
            open_h.append(h)
            continue
        mag = -h
        best = -1
        best_err = tolerance + 1e-9
        # newest first, strict comparison: the most recent candidate wins ties
        for j in range(len(open_t) - 1, -1, -1):
            if t - open_t[j] > max_on_s:
                continue
            err = abs(mag - open_h[j]) / max(mag, open_h[j])
            if err < best_err:
                best = j
                best_err = err
        if best >= 0:
            on.append(open_t[best])
            off.append(t)
            height.append(0.5 * (open_h[best] + mag))
            del open_t[best]
            del open_h[best]
    order = np.argsort(on) if on else np.empty(0, dtype=np.int64)
    cycles = Cycles(
        np.asarray(on)[order] if on else np.empty(0),
        np.asarray(off)[order] if off else np.empty(0),
        np.asarray(height)[order] if height else np.empty(0),
    )
    return cycles, np.asarray(open_t, dtype=np.float64)


def group_by_height(values: FloatArray, tolerance: float = GROUP_TOLERANCE) -> IntArray:
    """Assign group ids to positive ``values``.

    A group starts at its smallest value and spans ``1 + tolerance`` of it,
    i.e. roughly ±tolerance/2 around its centre, so that typical cooktop
    zones (for example 1.8 kW and 2.3 kW) end up in different groups.
    """
    n = values.shape[0]
    groups = np.zeros(n, dtype=np.int64)
    if n == 0:
        return groups
    order = np.argsort(values)
    gid = 0
    anchor = values[order[0]]
    for i in order:
        if values[i] > anchor * (1 + tolerance):
            gid += 1
            anchor = values[i]
        groups[i] = gid
    return groups


def largest_equal_group(heights: FloatArray, tolerance: float = GROUP_TOLERANCE) -> int:
    """Return the size of the largest set of heights within ±tolerance of its median."""
    n = heights.shape[0]
    if n == 0:
        return 0
    srt = np.sort(heights)
    lo = srt * (1 - tolerance)
    hi = srt * (1 + tolerance)
    counts = np.searchsorted(srt, hi, side="right") - np.searchsorted(
        srt, lo, side="left"
    )
    return int(np.max(counts))


def robust_cv(values: FloatArray) -> float:
    """Return 1.4826 * MAD / median (robust coefficient of variation)."""
    if values.shape[0] < 2:
        return NO_INTERVAL_CV
    med = float(np.median(values))
    if med <= 0:
        return NO_INTERVAL_CV
    return 1.4826 * float(np.median(np.abs(values - med))) / med


def interval_cv(edges: Edges) -> float:
    """Return the robust CV of intervals between same-kind edges of one group.

    Rising and falling edges are treated separately and their intervals are
    pooled. The robust (median based) variant ignores a single long interval
    such as the heat-up phase before a cooktop starts cycling.
    """
    intervals: list[FloatArray] = []
    for direction in (1.0, -1.0):
        t = np.sort(edges.idx[np.sign(edges.height) == direction])
        if t.shape[0] >= 2:
            intervals.append(np.diff(t))
    if not intervals:
        return NO_INTERVAL_CV
    pooled = np.concatenate(intervals)
    if pooled.shape[0] < 3:
        return NO_INTERVAL_CV
    return robust_cv(pooled)


def off_durations(cycles: Cycles, tolerance: float = PAIR_TOLERANCE) -> FloatArray:
    """Return the off time between consecutive cycles of the same height group."""
    if cycles.count < 2:
        return np.empty(0)
    groups = group_by_height(cycles.height, tolerance / 2)
    out: list[FloatArray] = []
    for g in np.unique(groups):
        sel = groups == g
        on = cycles.on[sel]
        off = cycles.off[sel]
        if on.shape[0] < 2:
            continue
        gap = on[1:] - off[:-1]
        out.append(gap[(gap > 0) & (gap < MAX_ON_DURATION_S)])
    return np.concatenate(out) if out else np.empty(0)


def still_running(current: FloatArray, open_on: FloatArray, edges: Edges) -> FloatArray:
    """Return the open rising edges whose load is still on at the window end."""
    if open_on.shape[0] == 0:
        return open_on
    n = current.shape[0]
    with np.errstate(invalid="ignore"), _quiet_nan_warnings():
        now_level = float(np.nanmedian(current[max(0, n - 10) :]))
    if not math.isfinite(now_level):
        return np.empty(0)
    keep: list[float] = []
    lookup = dict(zip(edges.idx.tolist(), edges.height.tolist(), strict=True))
    for t in open_on.tolist():
        start = int(t)
        before_seg = current[max(0, start - 12) : max(0, start - 2)]
        if before_seg.shape[0] == 0 or not np.any(np.isfinite(before_seg)):
            continue
        with _quiet_nan_warnings():
            before = float(np.nanmedian(before_seg))
        if now_level - before >= 0.6 * lookup.get(t, 0.0):
            keep.append(t)
    return np.asarray(keep, dtype=np.float64)


# ---------------------------------------------------------------- periodicity


def dominant_period(
    edges: Edges,
    length: int,
    min_period: int = PERIOD_MIN_S,
    max_period: int = PERIOD_MAX_S,
    smooth: int = 7,
) -> tuple[float, float]:
    """Return ``(period_s, strength)`` from the autocorrelation of an edge train.

    The signed edge train (+h for rising, -h for falling edges) is smoothed
    with a triangular kernel to tolerate jitter. The autocorrelation is
    corrected for the shrinking overlap at larger lags, so that a handful of
    regular cycles already yields a strength close to 1 while a single block
    yields none.
    """
    if edges.count < 4 or length <= min_period + 1:
        return 0.0, 0.0
    first = float(np.min(edges.idx))
    extent = float(np.max(edges.idx)) - first
    if extent < 2 * min_period:
        return 0.0, 0.0
    size = int(extent) + smooth + 2
    train = np.zeros(size)
    pos = np.clip(
        np.round(edges.idx - first).astype(np.int64) + smooth // 2 + 1, 0, size - 1
    )
    np.add.at(train, pos, edges.height / 1000.0)
    kernel = np.bartlett(smooth + 2)[1:-1]
    train = np.convolve(train, kernel, mode="same")
    full = np.correlate(train, train, mode="full")
    acf = full[size - 1 :]
    if acf[0] <= 0:
        return 0.0, 0.0
    max_lag = min(max_period, int(extent / 2))
    if max_lag < min_period:
        return 0.0, 0.0
    lags = np.arange(min_period, max_lag + 1)
    overlap = np.maximum(extent - lags, 1.0) / max(extent, 1.0)
    seg = np.minimum(acf[min_period : max_lag + 1] / acf[0] / overlap, 1.0)
    peak = float(np.max(seg))
    if peak <= 0:
        return 0.0, 0.0
    # choose the shortest lag that is an interior local maximum close to the
    # global one; a maximum at the edge of the search range is no period
    for i in range(1, seg.shape[0] - 1):
        if seg[i] >= 0.85 * peak and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            return float(lags[i]), float(seg[i])
    return 0.0, 0.0


@dataclass(frozen=True, slots=True)
class ZoneStats:
    """Statistics of the dominant group of equal-height edges (one zone)."""

    height: float = 0.0
    n_edges: int = 0
    n_cycles: int = 0
    period_s: float = 0.0
    strength: float = 0.0
    interval_cv: float = NO_INTERVAL_CV
    on_median_s: float = 0.0
    off_median_s: float = 0.0
    heatup_cycles: int = 0


def heatup_restarts(
    edges: Edges, cycles: Cycles, tolerance: float = GROUP_TOLERANCE
) -> int:
    """Count re-starts at the same power after a continuous heat-up block.

    A radiant zone typically runs continuously first and is then switched
    on and off by its energy regulator at exactly the same power. A kettle,
    toaster or hair dryer does not restart by itself. Returns the number of
    rising edges of the same height (±tolerance) after the longest block of
    at least ``HEATUP_MIN_S`` seconds.
    """
    if cycles.count == 0:
        return 0
    durations = cycles.on_duration
    i = int(np.argmax(durations))
    if durations[i] < HEATUP_MIN_S:
        return 0
    height = float(cycles.height[i])
    off = float(cycles.off[i])
    after = (edges.idx > off) & (edges.height > 0)
    same = np.abs(edges.height - height) <= tolerance * height
    return int(np.count_nonzero(after & same))


def zone_stats(edges: Edges, cycles: Cycles, length: int, smooth: int = 7) -> ZoneStats:
    """Analyse every group of equal-height edges and return the dominant one.

    Several cooktop zones (and other appliances) superimpose linearly, so
    edges of one zone keep their height. Grouping by height separates them;
    the group with the most regular cycling wins.
    """
    if edges.count == 0:
        return ZoneStats()
    mags = np.abs(edges.height)
    groups = group_by_height(mags)
    best = ZoneStats()
    best_score = -1.0
    for g in np.unique(groups):
        sel = groups == g
        g_edges = Edges(edges.idx[sel], edges.height[sel])
        lo = float(np.min(mags[sel])) * (1 - PAIR_TOLERANCE / 2)
        hi = float(np.max(mags[sel])) * (1 + PAIR_TOLERANCE / 2)
        csel = (cycles.height >= lo) & (cycles.height <= hi)
        g_cycles = Cycles(cycles.on[csel], cycles.off[csel], cycles.height[csel])
        period, strength = dominant_period(g_edges, length, smooth=smooth)
        icv = interval_cv(g_edges)
        on_d = g_cycles.on_duration
        off_d = off_durations(g_cycles)
        stats = ZoneStats(
            height=float(np.median(mags[sel])),
            n_edges=g_edges.count,
            n_cycles=g_cycles.count,
            period_s=period,
            strength=strength,
            interval_cv=icv,
            on_median_s=float(np.median(on_d)) if on_d.shape[0] else 0.0,
            off_median_s=float(np.median(off_d)) if off_d.shape[0] else 0.0,
            heatup_cycles=heatup_restarts(g_edges, g_cycles),
        )
        score = g_cycles.count * (0.5 + strength) + 0.1 * g_edges.count
        if score > best_score:
            best = stats
            best_score = score
    return best


# ------------------------------------------------------------------- features


def level_distance(heights: FloatArray, levels: FloatArray) -> FloatArray:
    """Return the relative distance of each height to its nearest level.

    1.0 means "no level known" or "height unknown"; values are capped at 1.
    """
    h = np.abs(np.asarray(heights, dtype=np.float64))
    out = np.ones(h.shape[0])
    if levels.shape[0] == 0 or h.shape[0] == 0:
        return out
    rel = np.min(np.abs(h[:, None] - levels[None, :]) / levels[None, :], axis=1)
    return np.where(h > 0, np.minimum(rel, 1.0), 1.0)


def match_levels(heights: FloatArray, levels: FloatArray) -> npt.NDArray[np.bool_]:
    """Return which absolute heights match one of the learned power levels."""
    if levels.shape[0] == 0 or heights.shape[0] == 0:
        return np.zeros(heights.shape[0], dtype=bool)
    mags = np.abs(heights)[:, None]
    tol = np.maximum(LEVEL_TOLERANCE * levels, LEVEL_TOLERANCE_MIN_W)[None, :]
    matched: npt.NDArray[np.bool_] = np.any(
        np.abs(mags - levels[None, :]) <= tol, axis=1
    )
    return matched


def time_of_day(local_hour: float) -> tuple[float, float]:
    """Encode the local hour as (sin, cos)."""
    angle = 2 * math.pi * (local_hour % 24.0) / 24.0
    return math.sin(angle), math.cos(angle)


def _cv(values: FloatArray) -> float:
    if values.shape[0] < 2:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values)) / mean if mean > 0 else 0.0


def analyse_window(
    residual: FloatArray,
    window_s: int,
    threshold: float,
    span: int,
    smooth: int,
    levels: FloatArray,
    local_hour: float,
    activity_minutes: float = 0.0,
    period_smooth: int = 7,
    generation: FloatArray | None = None,
    generation_interval: float | None = None,
) -> WindowAnalysis:
    """Compute the feature vector for the newest ``window_s`` samples.

    ``residual`` may be longer than ``window_s``; the samples before the
    analysis window serve as the reference for the base load rise.
    """
    n = residual.shape[0]
    window_s = min(window_s, n)
    current = residual[n - window_s :]
    previous = residual[: n - window_s]

    settle = int(min(20.0, (generation_interval or 3.0) + 2.0))
    edges = detect_edges(current, threshold, span, smooth, settle)
    if generation is not None:
        gap = 2.0 * (generation_interval or 5.0) + 6.0
        edges = remove_generation_artifacts(
            edges, generation[n - window_s :], span, smooth, gap
        )
    cycles, open_on = pair_edges(edges)
    feats = np.zeros(N_FEATURES)
    fi = FEATURE_INDEX

    ups = edges.height[edges.height > 0]
    downs = edges.height[edges.height < 0]
    feats[fi["n_on"]] = ups.shape[0]
    feats[fi["n_off"]] = downs.shape[0]
    mags = np.abs(edges.height)
    if mags.shape[0]:
        feats[fi["height_median"]] = float(np.median(mags))
        feats[fi["height_cv"]] = _cv(mags)
        feats[fi["height_max"]] = float(np.max(mags))
        feats[fi["level_match_frac"]] = float(np.mean(match_levels(mags, levels)))
    zone = zone_stats(edges, cycles, window_s, period_smooth)
    period, strength = zone.period_s, zone.strength
    feats[fi["interval_cv"]] = zone.interval_cv
    feats[fi["period_s"]] = period
    feats[fi["period_strength"]] = strength
    feats[fi["zone_height"]] = zone.height
    feats[fi["levels_known"]] = levels.shape[0]
    # set by the engine from the activity episode; 1 = "assume yes"
    feats[fi["zone_heatup_seen"]] = 1.0
    feats[fi["heatup_cycles"]] = zone.heatup_cycles
    on_med = zone.on_median_s
    off_med = zone.off_median_s
    feats[fi["on_median_s"]] = on_med
    feats[fi["off_median_s"]] = off_med
    feats[fi["duty_cycle"]] = (
        on_med / (on_med + off_med) if on_med > 0 and off_med > 0 else 0.0
    )
    feats[fi["n_cycles"]] = cycles.count
    feats[fi["n_equal_cycles"]] = largest_equal_group(cycles.height)
    on_d = cycles.on_duration

    finite_now = current[np.isfinite(current)]
    if finite_now.shape[0]:
        k10 = int(0.1 * (finite_now.shape[0] - 1))
        feats[fi["peak_above_base"]] = float(np.max(finite_now)) - float(
            np.partition(finite_now, k10)[k10]
        )
    with np.errstate(invalid="ignore"), _quiet_nan_warnings():
        if np.any(np.isfinite(current)) and np.any(np.isfinite(previous)):
            prev_ok = previous[np.isfinite(previous)]
            k = int(0.1 * (prev_ok.shape[0] - 1))
            base_prev = float(np.partition(prev_ok, k)[k])
            feats[fi["baseline_rise"]] = float(np.nanmean(current)) - base_prev

    longest = float(np.max(on_d)) if on_d.shape[0] else 0.0
    running = still_running(current, open_on, edges)
    if running.shape[0]:
        longest = max(longest, float(window_s - np.min(running)))
    feats[fi["longest_block_s"]] = longest

    tod_sin, tod_cos = time_of_day(local_hour)
    feats[fi["tod_sin"]] = tod_sin
    feats[fi["tod_cos"]] = tod_cos
    feats[fi["activity_minutes"]] = activity_minutes

    return WindowAnalysis(
        features=feats,
        edges=edges,
        cycles=cycles,
        open_on=open_on,
        period_s=period,
        period_strength=strength,
    )
