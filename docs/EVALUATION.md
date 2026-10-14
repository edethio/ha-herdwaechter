# Evaluation

All numbers below come from the synthetic household simulation in `tests/sim`. It drives the **same code** that runs in Home Assistant (state events → `Resampler` → `HerdwaechterEngine`). The data is entirely synthetic; it is not derived from any real household.

Reproduce:

```bash
python scripts/simulate.py                         # reference scenario incl. prior-only baseline
pytest -m slow -v                                  # the end-to-end test asserting the requirements
python scripts/simulate.py --grid-interval 10 --skip-baseline   # variants, see below
```

## Reference scenario

**Household (seed 7, 30 days):**

- Base load 140–250 W with daily pattern, noise, random small appliances (< 250 W), fridge cycles (~100 W, 10–15 min on).
- **Radiant cooktop** with zones of 1200 / 1800 / 2100 W (drawn per household from 1000–1200 / 1700–1800 / 2100–2300 W). Breakfast (15 % of days, 10–15 min), lunch (45 %, 15–40 min), dinner (85 %, 20–60 min); 1–3 zones (45 / 40 / 15 %), additional zones start up to 40 % into the session. Every zone first runs continuously (heat-up, log-normal with median 2 min, 0.5–8 min), then cycles with a period of 20–70 s, duty cycle 25–75 %, ±8 % jitter per cycle, power ±1.5 %, a setting change in 40 % of the sessions. 39 sessions in 30 days.
- **Disturbers:** kettle (2000–2200 W, 2–4 min, 1–3 per day), hair dryer (two levels), toaster, microwave (65 % continuous, 35 % cycling programmes with a period of 12–22 s for 3–8 min; magnetron step at least 12 % away from every zone power, as the specification describes "shorter and with different step heights"), washing machine (heater with thermostat restarts, tumbling motor, spin) and dishwasher (two heating phases) – both with their own power meter that is subtracted, instantaneous water heater (6–12 kW, modulating), heat pump (1.5–2.5 kW with soft start), 11 kW wallbox.
- **Zero export:** PV up to 900 W (sunny and cloudy days) plus a 2 kWh battery; the inverter regulates the grid import to ~15 W with 2 s control delay and 200 W/s ramp, limited to 800 W. At night the battery runs empty and load steps are fully visible at the meter.

**Sensors:** grid power every 3 s (random phase, only changes are reported, ~0.7 outages per day of 15 s – 8 min), inverter output every 5 s with 2 s latency (`unavailable` at night on half of the nights), dishwasher/washer meters every 10 s.

**Virtual user:** labels 50 % of the cooking sessions (label on 0.5–4 min after the start, off 0–3 min after the end, 10 % forget to switch off → automatic off after 120 min) and presses *False alarm* for 50 % of the false alarms (1–6 min after the onset).

**Metrics** (event level): an on-period of the binary sensor is a true positive if it overlaps a cooking session (or ends at most 10 min after it). Precision = true on-periods / all on-periods; recall = detected sessions / sessions; latency = first on-period start − session start; false alarms per day = false on-periods / days. The trained state is evaluated on the last seven days (days 23–29).

### Results

| Metric (trained state, days 23–29) | Required | Reached |
|---|---|---|
| Precision (cooking events) | ≥ 0.90 | **1.000** |
| Recall (cooking events) | ≥ 0.90 | **1.000** |
| Median detection latency | ≤ 4 min | **3.92 min** |
| False alarms per day | ≤ 0.2 | **0.000** |

Progress over time (with learning):

| Days | Sessions | Precision | Recall | Median latency | False alarms / day |
|---|---|---|---|---|---|
| 0–6 | 9 | 1.000 | 1.000 | 4.60 min | 0.000 |
| 7–13 | 11 | 0.917 | 1.000 | 4.09 min | 0.143 |
| 14–20 | 8 | 1.000 | 1.000 | 4.10 min | 0.000 |
| 21–27 | 8 | 1.000 | 1.000 | 3.92 min | 0.000 |
| 28–29 | 3 | 1.000 | 1.000 | 4.48 min | 0.000 |
| **all 30 days** | 39 | 0.976 | 1.000 | 4.28 min | 0.033 |

Prior knowledge only (learning paused, same data):

| Days | Sessions | Precision | Recall | Median latency | False alarms / day |
|---|---|---|---|---|---|
| day 0 | 2 | 1.000 | **1.000** | 4.53 min | 0.000 |
| all 30 days | 39 | 0.977 | 1.000 | 4.34 min | 0.033 |

The prior alone is already useful from day 0 (recall 1.0, requirement ≥ 0.6). Learning improves the detection over time: the latency drops from 4.6 min in the first week to 3.9 min, because the learned power levels make the "known zone restarts after its heat-up" rule available, and the false alarm of week 2 does not recur.

Learned state after 30 days:

- Learned power levels **1202 / 1798 / 2099 W** (truth 1200 / 1800 / 2100 W) plus a sum level of 3090 W (two zones switching within the same second).
- Estimated sensor lag 2.6 s (inverter sensor latency 2 s plus reporting phase), edge threshold 500 W.
- Personal model: weight 0.90; holdout precision 1.000, recall 0.963, Brier score 0.018 (prior: 0.028).
- 1571 positive and 2015 negative labels (95 strong) – the virtual user pressed *False alarm* only once.
- The only false alarm in 30 days: a cycling microwave programme while the washing machine was running (day 12).

## Variants

Same household and user, one parameter changed (trained state, days 23–29):

| Scenario | Precision | Recall | Median latency | False alarms / day | Result |
|---|---|---|---|---|---|
| Reference (3 s, zero export, subtraction) | 1.000 | 1.000 | 3.92 min | 0.000 | all met |
| Grid sensor every 1 s | 1.000 | 1.000 | 3.90 min | 0.000 | all met |
| Grid sensor every 5 s | 0.917 | 1.000 | 3.88 min | 0.143 | all met |
| Grid sensor every 10 s | 0.786 | 1.000 | 4.11 min | 0.429 | not met |
| No zero export (no PV/battery) | 1.000 | 1.000 | 3.77 min | 0.000 | all met |
| Dishwasher/washer not subtracted | 1.000 | 1.000 | 4.11 min | 0.000 | latency +0.11 min |
| Other household (seed 3) | 1.000 | 1.000 | 4.61 min | 0.000 | latency +0.61 min |
| Other household (seed 21) | 0.917 | 1.000 | 4.19 min | 0.143 | latency +0.19 min |

Observations:

- Precision, recall and the false alarm rate meet the requirements in all scenarios with update intervals up to 5 s.
- **10 s updates** are at the limit stated in the specification ("< 10 s normal"): the regulator cycles of 20–70 s are then sampled with only 2–7 points per period, cycling microwave programmes are not resolved any more and are confused with the cooktop. Use update intervals ≤ 5 s.
- The **latency** is met in the reference scenario but exceeds 4 minutes by 0.1–0.6 min for other households. See the next section.

## Latency: physical lower bound

Before its energy regulator starts cycling, a radiant zone is indistinguishable from a kettle, toaster or hair dryer: it is one continuous block of power. Detecting it during the heat-up would mean accepting a false alarm for every kettle (in the reference household the kettle draws 2000–2200 W, the large zone 2100 W). The earliest robust moment is therefore after the heat-up **plus** the first regulator cycle:

```
latency ≥ heat-up + off-time + one regulator period + confirmation (≈ 10 s) + edge settling (≈ 8 s)
```

In the reference run (all 39 sessions) the median heat-up of the first zone is 2.2 min and the median time from the end of the heat-up to the detection is 2.2 min (range 1.0–4.7 min); the median regulator period is 43 s. The time after the heat-up grows almost linearly with the regulator period – a zone with a 70 s period simply needs 70 s to show its first cycle. With heat-up phases of 2–3 minutes and periods above one minute the median latency is necessarily close to or above 4 minutes, independent of the detector. The requirement is reached when the heat-up phases are short (as in most pan cooking) and missed by up to 0.6 minutes for households that heat up longer – this is a property of the cooking pattern, not of a missing tuning step.

Measures that were implemented to reach this bound (each measured in the simulation):

| Measure | Effect on median latency |
|---|---|
| Features per zone (height groups) instead of per window | 10.4 → 5.6 min |
| Robust lag estimation (load steps excluded from the correlation) | removed outliers up to 25 min |
| Heat-up followed by restart at the same power (energy regulator signature) | −0.5 … −2 min for long periods |
| Event-driven extra evaluations (8 s after a power step, 10 s confirmation) | −0.35 min |
| Known zone power restarting after its heat-up (uses learned levels), measured together with the stricter microwave rule | −0.3 … −0.4 min |

## False alarms: what was learned along the way

- Without PU-safe mixing the personal model produced up to 23 false alarms in 30 days although its holdout score was excellent: active but unlabelled windows (kettle, water heater, washing machine artefacts) are never used as negatives, so the model was biased towards "on" for them. The model may now only lower the probability freely; raising it requires the prior to be above the switch-on threshold already.
- A cycling microwave programme is the hardest confuser: a regular on/off pattern of one fixed power. Two physical rules separate it: the cycling zone must have run continuously (heat-up) earlier in the same activity, and very short periods (< 18 s) are penalised. A time-based variant of the first rule only shifted the false alarms to its time limit; the episode memory removed 9 of 10 prior-only false alarms.
- Zero export distorts step heights while the inverter regulates. Heights are therefore re-measured on the settled plateaus, and steps that coincide with an equally large change of the inverter or subtracted meter signal are removed as reporting artefacts.

## Limitations of this evaluation

- The household, the cooktop and all appliances are synthetic. Real cooktops, meters and inverters may behave differently (e.g. averaging meters, much slower inverter polling, ovens, induction).
- The microwave's magnetron step was drawn at least 12 % away from the cooktop's zone powers, following the specification. A microwave with exactly the power of a zone, cycling right after a continuous run, would be harder to separate and would rely on false alarm labels.
- Ovens were not simulated; they also cycle heating elements and are a planned profile.
- The virtual user's behaviour (50 % labels, 50 % false alarm presses) is a simplification.
