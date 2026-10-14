# Architecture

This document describes how Herdwächter works and why it was built this way.

## Overview

```
 HA state events (grid, generation, subtracted meters, external label)
        │  W / kW / MW → W, sign inversion, unavailable → gap
        ▼
 Resampler ── 1 s sample-and-hold raster, 30 min ring buffers (RAM only),
        │     effective update interval (median of real update gaps)
        ▼
 HerdwaechterEngine.evaluate()         (executor thread, every 30 s + extra)
   1. residual = grid[t] + generation[t + lag] − subtracted[t]
   2. calibration: noise floor (MAD) → edge threshold; lag (cross correlation)
   3. edges: rolling median + span difference, settled-plateau heights,
      removal of transients caused by late auxiliary reports
   4. pairing (±20 %), grouping by height (≈ one cooking zone each),
      per zone: autocorrelation period, robust interval CV, duty cycle,
      heat-up/restart pattern
   5. feature vector (24 values)
   6. p_prior  = profile rules (profiles/ceran.py)
      p_model  = online logistic regression (personal model)
      p        = w·p_model + (1 − w)·p_prior   (PU-safe weight, see below)
   7. hysteresis (2 consecutive evaluations, 12 min no-cycle timeout)
   8. confidence, learning state, labels (PU rules), power levels,
      retraining, active questions
        ▼
 EvaluationResult → entities (binary_sensor, sensors, switches, buttons)
        ▼
 Store (.storage/herdwaechter.<entry_id>, versioned, delayed saves)
```

### Modules

| Module | Responsibility | Home Assistant? |
|---|---|---|
| `const.py` | keys, defaults, limits | no |
| `sampling.py` | 1 s raster, ring buffers, update intervals | no |
| `calibration.py` | noise floor, edge threshold, lag estimation | no |
| `features.py` | edges, pairing, periodicity, zone analysis, feature vector | no |
| `profiles/` | profile interface, registry, radiant cooktop prior | no |
| `model.py` | Welford standardiser, logistic regression, metrics | no |
| `learning.py` | label data set, holdout, power level clustering, PU rules, quality, mixing weight | no |
| `engine.py` | orchestration, hysteresis, confidence, questions, extra-evaluation scheduler, persistence | no |
| `coordinator.py` | listeners, unit conversion, timers, executor jobs, storage, notifications | yes |
| `config_flow.py` | config flow + options flow with validation | yes |
| `binary_sensor.py`, `sensor.py`, `switch.py`, `button.py`, `entity.py` | entities | yes |
| `services.py`, `diagnostics.py` | services, diagnostics | yes |

The detection core is pure Python + numpy. The simulation (`tests/sim`) drives exactly the same `Resampler` and `HerdwaechterEngine` that run inside Home Assistant; only the event source differs.

## Decisions

### Platform and compatibility

- **Minimum Home Assistant 2026.3.0.** The current stable release is 2026.10.1 (tested). 2026.3 is the first release that serves brand images from a custom integration's own `brand/` folder, which HACS requires; it also runs on Python 3.14. All APIs used (`runtime_data`, `OptionsFlowWithReload`, `AddConfigEntryEntitiesCallback`, `Store(minor_version, serialize_in_event_loop)`, `async_track_state_report_event`) are available there.
- **No requirements.** Only the standard library and numpy (shipped with Home Assistant). No scikit-learn, pandas or scipy.
- **No blocking code in the event loop.** Every numpy computation (evaluation, retraining, diagnostics, export, serialisation, loading) runs via `hass.async_add_executor_job`. Engine calls are serialised with an `asyncio.Lock`, so the engine never runs concurrently in two threads. Storage serialisation happens off the event loop (`serialize_in_event_loop=False`) on an immutable snapshot that is produced inside the executor job. Only two tiny operations stay in the event loop on purpose, because they must be ordered with the state events: writing a sensor update into the ring buffers (≈ 10 µs) and copying the analysis window (≈ 30 µs) before each evaluation.
- **Own coordinator instead of `DataUpdateCoordinator`.** Data is pushed by state events, not polled; the evaluation cadence is a mix of a fixed interval and event-driven extra evaluations. A small listener-based coordinator is simpler and keeps all timers under explicit control (`async_shutdown` cancels everything).
- Files beyond the requested list: `engine.py` (HA-free orchestration so that the simulation uses the identical code), `entity.py` (shared entity base), `services.py` (service registration), `icons.json`, `brand/`.

### Configuration

- **One config entry per monitored appliance**, profile stored in `entry.data`, everything else in `entry.options`. This makes every setting editable in the options flow (two steps: sensors, tuning) and lets `OptionsFlowWithReload` reload the entry automatically. A second entry with the same grid sensor and profile is rejected.
- Default name depends on the backend language ("Herd" in German, "Stove" otherwise).
- The external label source can only be an `input_boolean`, `switch` or `binary_sensor` (enforced by the selector), so it can never collide with a power sensor.

### Sampling

- **Sample-and-hold on a 1 s raster** with fixed-size numpy ring buffers (30 min, or 2 × window + margin if larger).
- **Grid sensor `unavailable`/`unknown` → gap (NaN), never 0 W.** A grid value older than 15 min is also treated as a gap (sensor stuck); `state_reported` events keep sensors that report identical values fresh.
- **Optional sensors `unavailable` → their last value is held for 30 s, then they count as 0 W** and the seconds are flagged. Many inverters report `unavailable` at night when they are off; treating that as a gap would blind the detector every evening.
- Windows with less than 80 % valid samples are not evaluated (`insufficient_data`).
- The **effective update interval** is the median of the gaps between real updates (state changes plus state reports). ≤ 10 s: normal; 10–15 s: confidence reduced linearly; > 15 s: `insufficient_data` and confidence × 0.15.

### Calibration

- **Noise floor:** MAD-based sigma of span differences in quiet windows (≤ 2 % of differences above 300 W). Edge threshold = 10 σ, clamped to 500–900 W, default 700 W. Updated on every 4th evaluation, EMA smoothed.
- **Lag compensation:** cross correlation of the difference signals of grid and generation within ±20 s. In zero export every inverter change is mirrored by an opposite grid change at the same instant, so the lag of the strongest negative correlation is the reporting offset. Two refinements were necessary in the simulation: grid differences larger than the inverter's range (the load steps themselves) are ignored, because the inverter reacts to them a few seconds later and that control delay would otherwise dominate; only clear interior peaks are accepted. The estimate is smoothed with an EMA; outliers (more than 3 s from the median of the last 15 raw estimates) are discarded, and if the majority of recent estimates moves, the estimate follows the median.
- While not calibrated the learning state is `calibrating`; detection runs with the defaults (700 W threshold, lag 0).

### Edges and features

- **Edges:** rolling median (3–5 s) then the difference over `span` raster steps (3, or update interval + 1 for slow sensors). Runs above the threshold form one edge. The height is re-measured on the settled plateaus around the edge, because with zero export the inverter may already regulate within the span.
- **Artefact suppression:** a residual edge that coincides with an equally large change of the non-grid part of the residual (shifted generation − subtracted meters) is caused by a late auxiliary report, not by a load step. It is removed together with its opposite partner. This makes zero export and separately metered appliances with 5–10 s reporting usable.
- **Pairing:** a falling edge closes the most recent open rising edge within ±20 % height.
- **Zones:** several zones superimpose linearly, so edges are grouped by height (one group ≈ one zone). Period (autocorrelation of the smoothed signed edge train, overlap-corrected, 8–120 s), regularity (robust CV = MAD/median of same-kind edge intervals – a single long heat-up interval does not spoil it), duty cycle and on/off durations are computed per group; the group with the strongest cycling is reported. Without grouping, a second zone or any other appliance in the window destroyed the periodicity features.
- **Heat-up pattern:** a block of ≥ 45 s followed by restarts at the same power is the hallmark of an energy regulator taking over after the heat-up phase; kettles and toasters never restart by themselves. This feature reduced the median detection latency by about one regulator period.
- **Heat-up memory per activity episode:** the engine remembers the powers that ran continuously (≥ 30 s) since the current activity started. A zone that cycles although its power never ran continuously during this activity did not heat up first – typical for microwave programmes, which cycle right away, and untypical for a radiant element that starts cold (even without a boost the first on-phase of an energy regulator is long because its bimetal is cold). A time-based rule ("no heat-up during the first 8 minutes") only shifted the false alarms to minute 8 in the simulation; the episode memory removed 9 of 10 prior-only false alarms. If the start of an activity was not observed (e.g. right after a restart) the zone gets the benefit of the doubt.
- The feature vector contains everything listed in the specification plus `height_max`, `activity_minutes` (duration of the current activity episode), `n_equal_cycles`, `zone_height`, `heatup_cycles`, `peak_above_base`, `levels_known` and `zone_heatup_seen`.

### Prior (profile "ceran")

Explainable logit rules in `profiles/ceran.py`: repeated equal-height cycles, heat-up then cycling, periodicity in 8–120 s (only interior autocorrelation peaks), regular intervals, plausible zone power 0.7–3.6 kW, plausible duty cycle; penalties for single blocks, long blocks without cycling, cycling without a preceding heat-up, very short periods (< 18 s), very large or modulating loads (> 7 kW above base load) and – once ≥ 2 power levels are known – regular cycling at a power that matches none of them; bonuses for steps that match learned power levels and for a known zone power restarting after its heat-up. The output is mapped to `[0.01, 0.86]`, so the prior alone rarely exceeds 85 %. New profiles implement `DeviceProfile` and register in `profiles/__init__.py`.

### Personal model and learning

- **Model:** logistic regression with Welford standardisation, L2 penalty proportional to the total sample weight, class weights (both classes get equal total weight), a mini-batch gradient update for every new label (new window + 15 replayed windows) and a damped Newton/IRLS refit (with backtracking line search, robust on separable data) every 25 labels or 6 h. The prior logit is used as a fixed **offset**: `logit(p_model) = prior_logit + correction(x)`. Without data the model equals the prior, and the regularisation pulls towards the prior instead of towards 50 %.
- **Power levels:** incremental 1D clustering of paired cycle heights from labelled windows (weight 1) and clearly cycling windows with high probability (weight 0.3). A level needs ≥ 10 support, ≥ 15 % of the strongest level and twice as much positive as negative evidence (false alarms add negative evidence). Whether a window's zone matches a level is **recomputed from the stored zone height with the current levels** at training and prediction time; windows stored before a level was known would otherwise teach the model that "no matching level" is fine.
- **Positive-unlabelled rules** (`LabelPolicy`): label on → positive (only windows with edge activity, so a forgotten switch does not teach "silence = cooking"); after label off the next 2 evaluations are neutral; windows without any edge activity become weak negatives (weight 0.2, one in 40 quiet evaluations); false alarm button and "no" answers → strong negatives (weight 3); active windows without a label are never labelled.
- **Data set:** feature vectors with label, weight, timestamp, source and group id only – no raw data. Max 5000 entries; when full, every other entry of the older half is dropped (weak negatives first), plus an age weight with 180 day half-life, so drift is possible without forgetting.
- **Holdout:** a deterministic CRC32 hash of the *label group* (one labelling action, one hour of weak negatives) assigns ~20 % of the groups to the holdout set. Splitting per window would leak, because neighbouring windows overlap by 95 %. A label category (positive / strong negative / weak negative) with fewer than 3 groups is not held out at all: in the simulation the only false alarm the user had marked landed in the holdout, so the model never learned from it. Quality is known once ≥ 5 positive and ≥ 5 negative holdout examples exist.
- **Mixing weight `w`:** `w = 0.9 · maturity · m*`. Maturity grows with the number of positive and negative label groups. `m*` is the mixture weight that minimises the (class-balanced) holdout Brier score of `m·p_model + (1−m)·p_prior`; if the model does not beat the prior, `m*` → 0 and the prior takes over automatically. While the quality is unknown `m* = 0.5`.
- **PU-safe mixing:** the model may always lower `p` (it learns that from real negative labels), but it may raise `p` only where the prior itself sees a cooktop signature (gate from p_prior 0.4 to 0.8). Active but unlabelled windows (kettle, hair dryer, water heater, …) are never negatives, so the model has no counter examples there and is biased towards "on". In the simulation the ungated model produced 4× more false alarms although its holdout score was excellent – the holdout contains only labelled windows.
- **False alarm button:** labels all windows of the current (or last, ≤ 60 min ago) detected event as strong negatives, adds negative evidence to their power levels, retrains, switches the output off and suppresses it for the rest of that activity (max 30 min).

### Output

- **Hysteresis:** on when `p ≥ on threshold` in 2 consecutive evaluations; off when `p < off threshold` in 2 consecutive evaluations or when no edge matching the event's zones/learned levels was seen for 12 minutes.
- **Extra evaluations:** besides the regular interval (30 s) an evaluation runs ~8 s (or auxiliary update interval + 4 s) after a grid power step ≥ 500 W, and a confirming evaluation 10 s after the probability first crosses the switch-on threshold; evaluations are at least 10 s apart. This keeps the "two consecutive evaluations" rule but removes up to 30 s of waiting.
- **Confidence** = decisiveness `|p − 0.5|·2` × maturity factor (0.5 … 1 with the model weight) × data quality (valid fraction, update interval, calibration).
- **Learning state** priority: `insufficient_data` > `calibrating` > `prior_only` (no model weight, quality unknown) > `trained` (quality known, maturity ≥ 0.6) > `learning`.

### Active questions

When enabled and `0.40 ≤ p ≤ 0.80` without a label: at most one question per activity episode and the configured number per day. The event `herdwaechter_question` is fired; with a notify service a Companion app notification with *Yes*/*No* actions (`HERDWAECHTER_YES_<id>`, `HERDWAECHTER_NO_<id>`) is sent. Answers (mobile app action event or `herdwaechter.answer_question`) label the windows around the question with weight 3. Questions expire after 30 minutes. Notification texts come from the `common` section of the translations.

### Persistence and lifecycle

- `Store` with version 1.1 and a migration hook (`HerdwaechterStore._async_migrate_func`). Content: calibration, model weights and standardisation, power levels, label data set (feature matrix as zlib-compressed float16), quality, model weight, pause flag, label switch state.
- **Write budget:** routine learning changes are persisted at most every 15 minutes, pure calibration updates at most hourly, user actions within 10 s, and the state is always saved on unload/restart. A typical state file is a few hundred kB.
- A restart restores the learned state without recalibration; only the 30 min ring buffer and the 24 h window history (used by "false alarm", answers and `add_label`) are lost. The label switch is restored with its remaining auto-off time.
- `async_unload_entry` unloads the platforms, cancels all listeners and timers (state change, state report, interval, mobile app event, label timer, extra evaluation timer) and saves. `async_remove_entry` deletes the store.

### Privacy and naming

- No network access, no telemetry. Diagnostics contain sensor *counts*, never entity IDs. The training data export contains no timestamps and no entity IDs.
- "Ceran" is used only as the internal profile key and descriptively in documentation (CERAN® is a trademark of SCHOTT AG). The icon is an own design.

## Resource usage

Measured on a desktop x86 CPU with Python 3.14 and numpy 2.3 (one entry, zero export, 2 generation + 2 subtracted sensors):

| Item | Value |
|---|---|
| Memory (ring buffers, 24 h window history, full label data set of 5000) | ≈ 3.7 MB |
| Sensor update (event → raster) | ≈ 11 µs |
| Evaluation, quiet window | ≈ 1.3 ms |
| Evaluation, cooking window | ≈ 3.8 ms |
| Full refit with 4750 labels | ≈ 50 ms |
| Stored state (full data set) | ≈ 0.4 MB |

A Raspberry Pi 4 is roughly 5–10× slower per core: about 10–40 ms per evaluation every 30 s (plus a few event-driven extra evaluations while a cooktop is active) and well below one second per refit, all in the executor. This is far below 1 % of one core. The memory footprint is dominated by the 24 h history (≈ 1.3 MB) and the label data set (≈ 0.5 MB).
