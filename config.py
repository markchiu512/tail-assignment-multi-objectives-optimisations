"""Sanitised configuration for the public portfolio version.

The production system reads equivalent settings from a secured deployment.
This file intentionally contains no real aircraft registers, workspace names,
warehouse identifiers, or operational restrictions.
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


MTL_DATE_OVERRIDE = ""
MTL_DATE = os.environ.get("MTL_DATE", MTL_DATE_OVERRIDE) or (
    datetime.now(ZoneInfo("Europe/London")).date() + timedelta(days=3)
).strftime("%d/%m/%Y")
DATE_PREFIX = datetime.strptime(MTL_DATE, "%d/%m/%Y").strftime("%d%b%Y").upper()

# Operational constraints are deployment configuration, not public source data.
LEASE_AIRCRAFTREG: list[str] = []
IRIS_UK_AIRCRAFTREG: list[str] = []
IRIS_EU_AIRCRAFTREG: list[str] = []
AUTOLAND_AIRCRAFTREG: list[str] = []
CYPRUS_PROHIBITED: list[str] = []
MANUAL_DROP_AIRCRAFTREG: list[str] = []
STANDBY_AIRCRAFT: dict[str, str] = {}

# Enable these route-specific rules by supplying approved deployment values.
CAPE_VERDE_IATA: list[str] = []
CYPRUS_IATA: list[str] = []
KEF_IATA: list[str] = []
NEO_AIRCRAFT_TYPES: list[int] = []
CANCELLED_FLIGHTNUMBER: list[int] = []

GROUND_EVENTS_SOURCE = os.environ.get("GROUND_EVENTS_SOURCE", "databricks")
GROUNDEVENTS_FILE = os.environ.get("GROUNDEVENTS_FILE", "ground_events.xls")

# Databricks paths are supplied by the deployment environment.
VOLUME_BASE = os.environ.get("OPTI_VOLUME_BASE", "/Volumes/<catalog>/<schema>/tail_allocation")
VOLUME_RESULTS_BASE = os.environ.get("OPTI_VOLUME_RESULTS_BASE", VOLUME_BASE)
VOLUME_INPUT_BASE = os.environ.get("OPTI_VOLUME_INPUT_BASE", VOLUME_BASE)
VOLUME_INPUT_DIRECTORY = f"{VOLUME_INPUT_BASE}/data/input"
VOLUME_OUTPUT_DIRECTORY = f"{VOLUME_RESULTS_BASE}/data/output/{DATE_PREFIX}"
VOLUME_INTERMEDIATE_DIRECTORY = f"{VOLUME_RESULTS_BASE}/data/intermediate/{DATE_PREFIX}"
VOLUME_OUTPUT_BASE = f"{VOLUME_RESULTS_BASE}/data/output"
VOLUME_INTERMEDIATE_BASE = f"{VOLUME_RESULTS_BASE}/data/intermediate"
VOLUME_MODEL_DIRECTORY = f"{VOLUME_BASE}/model"
IS_DEMO_RESULTS_VOLUME = VOLUME_RESULTS_BASE != VOLUME_BASE

_STAGING_ROOT = os.environ.get("OPTI_STAGING_ROOT", "/tmp/tail_allocation_staging")
INPUT_DIRECTORY = f"{_STAGING_ROOT}/input"
OUTPUT_DIRECTORY = f"{_STAGING_ROOT}/output/{DATE_PREFIX}"
INTERMEDIATE_DIRECTORY = f"{_STAGING_ROOT}/intermediate/{DATE_PREFIX}"
OUTPUT_BASE = f"{_STAGING_ROOT}/output"
INTERMEDIATE_BASE = f"{_STAGING_ROOT}/intermediate"
ML_MODELS_STAGING_DIRECTORY = f"{_STAGING_ROOT}/model"

if os.environ.get("OPTI_MODELS_DIRECTORY"):
    ML_MODELS_DIRECTORY = os.environ["OPTI_MODELS_DIRECTORY"]
elif os.path.isfile("model/pooled_lgbm_target_baseline_fuel_kg.joblib"):
    ML_MODELS_DIRECTORY = "model"
else:
    ML_MODELS_DIRECTORY = ML_MODELS_STAGING_DIRECTORY

FUEL_PRICE = float(os.environ.get("FUEL_PRICE", "1.20"))
EPS_VALUES = [round(i * 0.001, 3) for i in range(21)]
SELECTED_EPS = None
AUTO_SELECT_EPS = True
AUTO_SELECT_EPS_METHOD = "kneedle"
AUTO_THRESHOLD = True
MIN_SAVINGS_THRESHOLD = 300.0
FILTER_BY_TOP_GROUPS_INDEX = 9999
FUEL_BIAS = 1.2
SOLVER_BACKEND = os.environ.get("SOLVER_BACKEND", "highs")
