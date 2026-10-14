# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-14

Initial release.

### Added

- Detection of a running radiant glass-ceramic cooktop from the total household power ("ceran" profile).
- 1 s sample-and-hold resampling with gap handling, automatic update interval estimation and data quality reporting.
- Reconstruction of the household load for PV/battery systems with zero export (generation sensors with learned lag compensation) and subtraction of separately metered appliances.
- Automatic calibration of noise floor, edge threshold (500–900 W) and sensor lag.
- Explainable prior knowledge that works from day one; personal online logistic regression model with power level learning, holdout quality measurement and positive-unlabelled safe mixing.
- Entities: binary sensor, probability, confidence, learning state, residual load, label switch with automatic off, false alarm button, pause learning switch, reset model button.
- Optional active questions via event and Companion app notifications.
- Services `answer_question`, `add_label`, `export_training_data`; diagnostics without entity IDs.
- Config flow and options flow (English and German).
- Synthetic household simulator, unit/integration tests and a 30 day end-to-end learning test.

[0.1.0]: https://github.com/edethio/ha-herdwaechter/releases/tag/v0.1.0
