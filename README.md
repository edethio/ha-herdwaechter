# Herdwächter – cooktop detection from your electricity meter

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz/docs/faq/custom_repositories)
[![Validate](https://github.com/edethio/ha-herdwaechter/actions/workflows/validate.yml/badge.svg)](https://github.com/edethio/ha-herdwaechter/actions/workflows/validate.yml)
[![Tests](https://github.com/edethio/ha-herdwaechter/actions/workflows/tests.yml/badge.svg)](https://github.com/edethio/ha-herdwaechter/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!WARNING]
> This integration is a convenience tool based on statistical detection. It is **not a safety device** and must not be relied upon to prevent fire or damage. Detection can be wrong or delayed.

Herdwächter ("stove guard") is a Home Assistant custom integration that tells you whether a **radiant glass-ceramic cooktop** is switched on – using nothing but the **total household power** measured at your electricity meter. It needs no extra hardware, no smart plug and no cloud. It starts with built-in prior knowledge, reports a probability *and* how confident it is, and keeps learning from you.

[Deutsche Kurzanleitung weiter unten](#kurzanleitung-deutsch).

## Contents

- [What it does – and what it does not](#what-it-does--and-what-it-does-not)
- [Requirements](#requirements)
- [Installation via HACS](#installation-via-hacs)
- [Setup](#setup)
- [Entities](#entities)
- [How learning works](#how-learning-works)
- [Labelling by voice](#labelling-by-voice)
- [Example automations](#example-automations)
- [Services and events](#services-and-events)
- [Limitations](#limitations)
- [FAQ and troubleshooting](#faq-and-troubleshooting)
- [Privacy](#privacy)
- [Development](#development)
- [Kurzanleitung (Deutsch)](#kurzanleitung-deutsch)

## What it does – and what it does not

Radiant heating elements are switched by an energy regulator: each cooking zone turns on and off with a fixed power step (typically 1.0–2.5 kW) every 10–90 seconds, for many minutes. Herdwächter looks for exactly this signature in your meter data:

- resamples all configured sensors to a 1 second grid,
- reconstructs the household load if you have PV/battery with **zero export** (the inverter hides load steps at the meter after a few seconds),
- detects power steps, pairs them into on/off cycles, measures period, regularity and the typical "heat up continuously, then cycle" pattern,
- combines an explainable **prior** (works from day one) with a **personal model** that learns your cooktop's power levels and your household,
- switches a binary sensor with hysteresis and reports probability and confidence.

What it does **not** do:

- It is not a safety device and does not switch anything off.
- It contains no presence, person or zone logic and ships no "when leaving" blueprints – it only provides sensors. Build the automations you want yourself.
- It does not detect induction or gas cooktops (see [limitations](#limitations)).
- It sends nothing anywhere: no cloud, no telemetry, no network access.

## Requirements

- Home Assistant **2026.3** or newer.
- A sensor with the **net grid power** (W or kW, positive while importing) that updates at least every 10 seconds – ideally every 1–5 seconds. Smart meter readers (IR heads, P1 dongles), inverter/battery systems and energy monitors usually provide one.
- A radiant glass-ceramic cooktop (often called "Ceran" cooktop).
- Optional: sensors for the AC power of your inverter / battery discharge (needed with zero export) and power meters of large appliances (dishwasher, washing machine) that should be subtracted.

## Installation via HACS

1. Open **HACS** in Home Assistant.
2. Open the menu (⋮, top right) → **Custom repositories**.
3. Enter `https://github.com/edethio/ha-herdwaechter`, choose type **Integration** and click **Add**.
4. Search for **Herdwächter** in HACS, open it and click **Download**. Pick the latest release.
5. **Restart Home Assistant.**
6. Go to **Settings → Devices & services → Add integration** and search for **Herdwächter**.

Manual installation: copy `custom_components/herdwaechter` into the `custom_components` folder of your Home Assistant configuration and restart.

## Setup

One integration entry monitors one cooktop. The setup dialog asks for:

| Field | Description |
|---|---|
| Name | Name of the device, e.g. "Herd". |
| Grid power sensor | Net power at the meter, import positive. W and kW are converted automatically. |
| Invert sign | Enable if your meter reports import as a negative value. |
| Generation sensors (optional, several) | Inverter AC output, battery discharge power, … They are added to the grid power. |
| Devices to subtract (optional, several) | Power meters of dishwasher, washing machine, … They are subtracted. |
| External label source (optional) | An `input_boolean`, `switch` or `binary_sensor` whose "on" state means "the cooktop is on". |

**Zero export (Nulleinspeisung):** if an inverter or battery actively regulates your grid import to 0 W, every load step is visible at the meter only for a few seconds. Add the inverter's AC power and/or the battery discharge power as **generation sensors** so that Herdwächter can reconstruct `household load = grid + generation`. Set the **polling interval of your inverter integration as short as possible** (1–5 s): the lag between grid and inverter sensor is learned and compensated automatically, but slow inverter updates still blur the signal.

All settings can be changed later (**Configure** on the integration entry). The second page contains the tuning options:

| Option | Default |
|---|---|
| Switch-on threshold | 75 % |
| Switch-off threshold | 35 % |
| Evaluation interval | 30 s |
| Analysis window | 10 min |
| Maximum label duration (label switch turns itself off) | 120 min |
| Active questions | off |
| Notification service for questions | – |
| Maximum questions per day | 3 |

## Entities

All entities belong to one device per cooktop.

| Entity | Type | Description |
|---|---|---|
| *Herd* | binary sensor (`running`) | On while the cooktop is detected. Attributes: `probability`, `confidence`, `learned_power_levels`, `last_cycle`, `detected_period_s`, `model_weight`, `learning_state`. Switches on when the probability reaches the switch-on threshold in two consecutive evaluations, off when it stays below the switch-off threshold twice or when no matching cycling step was seen for 12 minutes. |
| Probability | sensor (%) | Current probability that the cooktop is on. |
| Confidence | sensor (%) | How much the current statement can be trusted – separate from the probability. Combines how clear the decision is, how mature the model is (labels, measured quality) and the current data quality (gaps, update interval, calibration). |
| Learning state | sensor (diagnostic) | `calibrating`, `prior_only`, `learning`, `trained` or `insufficient_data`. Attributes: positive/negative labels, holdout precision/recall/Brier score, estimated sensor lag, effective update interval, edge threshold. |
| Residual load | sensor (W, diagnostic, disabled by default) | The reconstructed load the detector analyses (grid + generation − subtracted devices). |
| Label cooktop on | switch | Your label: switch it on while you cook. It turns itself off after the maximum label duration. |
| False alarm | button | Marks the current or the last detected event as "not the cooktop". |
| Pause learning | switch (configuration) | No model updates while on (e.g. while you test other appliances). |
| Reset model | button (configuration, disabled by default) | Deletes the learned model, power levels and labels. Calibration is kept. Enable it in the entity settings first. |

Learning states:

- **calibrating** – the noise floor (and, with generation sensors, the sensor lag) is still being measured. Detection already runs with defaults.
- **prior_only** – no labels yet, only the built-in prior knowledge is used.
- **learning** – labels exist, the personal model is used with limited weight.
- **trained** – enough labels and a measured holdout quality.
- **insufficient_data** – gaps or an update interval above 15 s; the probability is shown with strongly reduced confidence.

## How learning works

Herdwächter works from the first minute with its prior knowledge. With your help it gets better:

1. **Label switch:** switch *Label cooktop on* on when you start cooking and off when you are done. You don't have to label every time – every second time is plenty. Windows while the label is on become positive examples.
2. **False alarm button:** when the binary sensor is on although the cooktop is off, press *False alarm*. The event becomes a strong negative example and the output switches off.
3. **Quiet periods** without any power steps are used as weak negative examples automatically. Periods with activity but without a label are **never** used as negatives – Herdwächter cannot know whether you just forgot to label.
4. From labelled cooking sessions it learns your cooktop's **power levels** (e.g. 1200 / 1800 / 2300 W, visible as attribute `learned_power_levels`).
5. A personal logistic regression model is trained on the labelled feature vectors (no raw data is stored). 20 % of the label groups are held out to measure precision, recall and Brier score. The model's weight grows with the number of labels and the measured quality and shrinks automatically if the model does not beat the prior. The model may always lower the probability, but it may raise it only where the prior itself sees a cooktop signature – this protects against appliances that were never labelled.

Tips for good labels:

- Switch the label on within the first few minutes of cooking, and off when the last zone is off.
- If you forget to switch it off, it turns off by itself; windows without any activity are never used as positive examples.
- Press *False alarm* only when you are sure the cooktop was off.
- Optional **active questions**: when the detection is unsure (40–80 %), Herdwächter fires an event and – if configured – sends a notification with *Yes*/*No* buttons to the Companion app. At most once per event and only up to the configured number per day. Unanswered questions expire after 30 minutes.

## Labelling by voice

The label switch is a normal switch entity, so you can control it with Assist (or another voice assistant connected to Home Assistant):

1. **Settings → Voice assistants → Expose**, add the entity *Label cooktop on*.
2. Open the entity and add an alias such as **"Herd"** or "Kochen".
3. Say "Turn on Herd" when you start cooking and "Turn off Herd" when you are done (German: "Schalte Herd ein" / "Schalte Herd aus").

## Example automations

Notify when the cooktop has been on for more than 2 hours:

```yaml
automation:
  - alias: "Cooktop on for more than 2 hours"
    triggers:
      - trigger: state
        entity_id: binary_sensor.herd
        to: "on"
        for: "02:00:00"
    actions:
      - action: notify.notify
        data:
          title: "Cooktop"
          message: "The cooktop seems to be on for more than 2 hours."
```

Use an existing kitchen button as label source: create an `input_boolean.cooking`, select it as *external label source* and toggle it from any automation.

Handle questions yourself (e.g. on a dashboard or another messenger):

```yaml
automation:
  - alias: "Herdwächter question"
    triggers:
      - trigger: event
        event_type: herdwaechter_question
    actions:
      - action: persistent_notification.create
        data:
          title: "Is the cooktop on?"
          message: >
            Probability {{ (trigger.event.data.probability * 100) | round }} %.
            Question ID {{ trigger.event.data.question_id }}
```

Answer it with the service shown below.

## Services and events

| Service | Description |
|---|---|
| `herdwaechter.answer_question` | `question_id`, `answer` (true = cooktop was on). Turns the answer into a strong label. |
| `herdwaechter.add_label` | `config_entry_id`, `start`, `end`, `cooking` – label a past period (within the last 24 h) afterwards. Returns the number of labelled windows. |
| `herdwaechter.export_training_data` | `config_entry_id` – writes the anonymised feature vectors (no raw data, no timestamps, no entity IDs) as JSON into your configuration folder. Useful for analysis and for improving the prior knowledge. |

| Event | Data |
|---|---|
| `herdwaechter_question` | `entry_id`, `question_id`, `probability` |

Notification actions from the Companion app (`HERDWAECHTER_YES_<id>` / `HERDWAECHTER_NO_<id>`) are handled automatically.

## Limitations

- **Induction cooktops** modulate their power electronically and do not show fixed on/off steps – not detected (a separate profile is planned).
- **Gas cooktops** use no relevant electrical power – not detected.
- **Slow meters:** with update intervals above ~10 s the regulator cycles become hard to see; above 15 s the learning state shows `insufficient_data` and the confidence is strongly reduced. Meters that average their readings over long intervals behave like slow meters.
- **Zero export with slow inverter sensors:** the reconstruction needs the inverter power with a similar update rate as the grid power. Very slow inverter polling (≥ 30 s) degrades detection while the inverter is regulating.
- **Confusable appliances:** ovens and some microwave programs also cycle heating elements. Herdwächter learns to separate them from your false alarm labels, but the first occurrences may cause false alarms.
- **Latency:** the cooktop is recognised once its regulator starts cycling, typically 2–5 minutes after switching it on. A zone that runs continuously at full power (e.g. boiling water) is recognised only when it starts cycling.

## FAQ and troubleshooting

**The learning state stays `calibrating`.** Calibration needs some quiet periods. With generation sensors it also needs periods in which the inverter regulates (daytime or a charged battery). Detection already works during calibration.

**`insufficient_data`.** The grid sensor updates too rarely or reports `unavailable`. Check the *effective update interval* attribute of the learning state sensor and reduce the polling interval of your meter integration.

**Many false alarms right after installation.** That is expected while only the prior is used. Press *False alarm* for them and label a few cooking sessions – the personal model learns quickly.

**The cooktop is detected late.** Detection needs the regulator to cycle a few times. Check that the update interval is ≤ 5 s and – with zero export – that the inverter power is configured as a generation sensor.

**The estimated lag looks wrong.** It is only estimated while the inverter actually regulates. It is a diagnostic value; the detector already removes transients caused by late inverter reports.

**Where is the learned data stored?** In `.storage/herdwaechter.<entry_id>` in your configuration folder. Removing the integration entry deletes it.

**Diagnostics:** Settings → Devices & services → Herdwächter → ⋮ → *Download diagnostics*. The file contains configuration counts, model parameters and recent feature vectors, but no entity IDs.

## Privacy

Everything runs locally inside Home Assistant. The integration makes no network connections and sends no telemetry. Only feature vectors with labels, model parameters, power levels and calibration values are stored – never raw power data. The power history (last 30 minutes) is kept in memory only.

## Development

```bash
pip install -r requirements_test.txt
ruff check . && ruff format --check . && mypy
pytest -m "not slow"          # unit and integration tests
pytest -m slow                # 30 day end-to-end learning simulation
python scripts/simulate.py    # simulation with a text report (optional --plot with matplotlib)
```

The detection core (`sampling`, `calibration`, `features`, `model`, `learning`, `engine`, `profiles`) is pure Python + numpy and runs without Home Assistant. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/EVALUATION.md](docs/EVALUATION.md).

CERAN® is a registered trademark of SCHOTT AG. It is used here only to describe the type of cooktop; this project is not affiliated with SCHOTT AG.

---

## Kurzanleitung (Deutsch)

> [!WARNING]
> Diese Integration ist ein Komfortwerkzeug auf Basis statistischer Erkennung. Sie ist **keine Sicherheitseinrichtung** und darf nicht zur Verhinderung von Bränden oder Schäden eingesetzt werden. Die Erkennung kann falsch oder verzögert sein.

**Was macht der Herdwächter?** Er erkennt allein anhand der Gesamtleistung am Stromzähler, ob ein Glaskeramik-Kochfeld mit Strahlheizkörpern („Ceran-Kochfeld") eingeschaltet ist. Er erkennt die typische Taktung der Kochzonen, lernt die Leistungsstufen deines Kochfelds und wird mit deinen Labels immer besser. Alles läuft lokal.

**Installation über HACS:**

1. HACS öffnen → Menü (⋮) → **Benutzerdefinierte Repositories**.
2. `https://github.com/edethio/ha-herdwaechter` eintragen, Typ **Integration**, **Hinzufügen**.
3. **Herdwächter** suchen, **Herunterladen**, Home Assistant **neu starten**.
4. **Einstellungen → Geräte & Dienste → Integration hinzufügen → Herdwächter**.

**Einrichtung:**

- **Netzleistungs-Sensor** wählen (Bezug positiv, sonst „Vorzeichen umkehren").
- Bei **PV mit Nulleinspeisung oder Akku**: die AC-Leistung des Wechselrichters und/oder die Akku-Entladeleistung als **Erzeugungs-Sensoren** angeben und das **Abfrageintervall des Wechselrichters möglichst kurz** einstellen (1–5 s).
- Geräte mit eigenem Powermeter (Spülmaschine, Waschmaschine) als **abzuziehende Geräte** angeben.
- Optional eine vorhandene Entität als **externe Label-Quelle** wählen.

**Entitäten:** *Herd* (Binärsensor), *Wahrscheinlichkeit*, *Sicherheit*, *Lernstatus*, *Restlast* (standardmäßig deaktiviert), Schalter *Label Herd ist an*, Taste *Fehlalarm*, Schalter *Lernen pausieren*, Taste *Modell zurücksetzen* (standardmäßig deaktiviert).

**So lernt er am besten:**

- Schalte beim Kochen den Schalter **Label Herd ist an** ein und danach wieder aus – etwa jedes zweite Mal reicht. Vergisst du das Ausschalten, schaltet er nach der eingestellten Höchstdauer selbst ab.
- Meldet er „an", obwohl der Herd aus ist, drücke **Fehlalarm**.
- Optional: **Aktive Rückfragen** in den Optionen aktivieren; die Companion-App fragt dann bei unsicheren Fällen mit „Ja"/„Nein".
- **Per Sprache:** Einstellungen → Sprachassistenten → Freigeben → Entität *Label Herd ist an* freigeben, Alias „Herd" vergeben. Dann: „Schalte Herd ein" / „Schalte Herd aus".

**Grenzen:** Induktion und Gas werden nicht erkannt. Das Update-Intervall des Zählers sollte unter 10 s liegen (ideal 1–5 s). Backofen und manche Mikrowellenprogramme können anfangs Fehlalarme auslösen, bis du sie per *Fehlalarm* markiert hast. Die Erkennung erfolgt, sobald die Kochzone zu takten beginnt, typischerweise 2–5 Minuten nach dem Einschalten.

**Datenschutz:** Alles bleibt lokal, keine Cloud, keine Telemetrie. Gespeichert werden nur Merkmalsvektoren mit Labels und Modellparameter, keine Rohdaten.
