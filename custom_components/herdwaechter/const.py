"""Constants for the Herdwächter integration.

This module must not import Home Assistant so that the detection core
(sampling, calibration, features, model, learning, engine) stays usable and
testable without Home Assistant.
"""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "herdwaechter"

# Config entry data / options keys
CONF_NAME: Final = "name"
CONF_PROFILE: Final = "profile"
CONF_GRID_SENSOR: Final = "grid_sensor"
CONF_INVERT_GRID: Final = "invert_grid"
CONF_GENERATION_SENSORS: Final = "generation_sensors"
CONF_SUBTRACT_SENSORS: Final = "subtract_sensors"
CONF_LABEL_ENTITY: Final = "label_entity"
CONF_ON_THRESHOLD: Final = "on_threshold"
CONF_OFF_THRESHOLD: Final = "off_threshold"
CONF_EVAL_INTERVAL: Final = "evaluation_interval"
CONF_WINDOW_MINUTES: Final = "window_minutes"
CONF_MAX_LABEL_MINUTES: Final = "max_label_minutes"
CONF_QUESTIONS_ENABLED: Final = "questions_enabled"
CONF_NOTIFY_SERVICE: Final = "notify_service"
CONF_MAX_QUESTIONS_PER_DAY: Final = "max_questions_per_day"

PROFILE_CERAN: Final = "ceran"
DEFAULT_PROFILE: Final = PROFILE_CERAN

# Option defaults
DEFAULT_ON_THRESHOLD: Final = 75
DEFAULT_OFF_THRESHOLD: Final = 35
DEFAULT_EVAL_INTERVAL: Final = 30
DEFAULT_WINDOW_MINUTES: Final = 10
DEFAULT_MAX_LABEL_MINUTES: Final = 120
DEFAULT_QUESTIONS_ENABLED: Final = False
DEFAULT_MAX_QUESTIONS_PER_DAY: Final = 3

# Option limits (validated by the options flow selectors)
MIN_EVAL_INTERVAL: Final = 10
MAX_EVAL_INTERVAL: Final = 120
MIN_WINDOW_MINUTES: Final = 5
MAX_WINDOW_MINUTES: Final = 15
MIN_MAX_LABEL_MINUTES: Final = 10
MAX_MAX_LABEL_MINUTES: Final = 480
MAX_QUESTIONS_LIMIT: Final = 20

# Sampling
RING_BUFFER_SECONDS: Final = 1800
MAX_GRID_HOLD_SECONDS: Final = 900
AUX_HOLD_SECONDS: Final = 30
MIN_VALID_FRACTION: Final = 0.8
UPDATE_INTERVAL_OK: Final = 10.0
UPDATE_INTERVAL_BAD: Final = 15.0

# Calibration
DEFAULT_EDGE_THRESHOLD: Final = 700.0
MIN_EDGE_THRESHOLD: Final = 500.0
MAX_EDGE_THRESHOLD: Final = 900.0
MAX_LAG_SECONDS: Final = 20

# Hysteresis / output
CONSECUTIVE_EVALUATIONS: Final = 2
NO_CYCLE_TIMEOUT_SECONDS: Final = 12 * 60
NEUTRAL_EVALUATIONS_AFTER_LABEL: Final = 2

# Active questions
QUESTION_P_LOW: Final = 0.40
QUESTION_P_HIGH: Final = 0.80
QUESTION_TIMEOUT_SECONDS: Final = 30 * 60
QUESTION_ACTION_PREFIX: Final = "HERDWAECHTER"

# Learning states (enum sensor options)
STATE_CALIBRATING: Final = "calibrating"
STATE_PRIOR_ONLY: Final = "prior_only"
STATE_LEARNING: Final = "learning"
STATE_TRAINED: Final = "trained"
STATE_INSUFFICIENT_DATA: Final = "insufficient_data"
LEARNING_STATES: Final = [
    STATE_CALIBRATING,
    STATE_PRIOR_ONLY,
    STATE_LEARNING,
    STATE_TRAINED,
    STATE_INSUFFICIENT_DATA,
]

# Events and services
EVENT_QUESTION: Final = "herdwaechter_question"
EVENT_MOBILE_APP_ACTION: Final = "mobile_app_notification_action"
SERVICE_ANSWER_QUESTION: Final = "answer_question"
SERVICE_ADD_LABEL: Final = "add_label"
SERVICE_EXPORT_TRAINING_DATA: Final = "export_training_data"
ATTR_CONFIG_ENTRY_ID: Final = "config_entry_id"
ATTR_QUESTION_ID: Final = "question_id"
ATTR_ANSWER: Final = "answer"
ATTR_START: Final = "start"
ATTR_END: Final = "end"
ATTR_COOKING: Final = "cooking"
ATTR_PROBABILITY: Final = "probability"
ATTR_ENTRY_ID: Final = "entry_id"

# Persistence
STORAGE_VERSION: Final = 1
STORAGE_MINOR_VERSION: Final = 1
STORAGE_SAVE_DELAY: Final = 120
