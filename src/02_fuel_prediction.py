"""
fuel_prediction.py  —  v2.0.0  Stage 2: Fuel Prediction

Reads the per-leg planning frame from Stage 1 and applies the pooled
LightGBM models to produce per-leg fuel and airborne predictions.

Input  : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_feature_engineering.csv
         (1 row per flight leg — all legs, no filtering)

Output : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_fuel_prediction.csv
         (same rows + pred_baseline_fuel_kg, pred_airborne_hours,
          pred_trip_fuel_kg)

Models : ml_model_upgrade_v2/models/
           pooled_lgbm_target_baseline_fuel_kg.joblib
           pooled_lgbm_target_airborne_hours.joblib
           label_encoders.joblib

Feature contract (must match feature_parity.yaml):
  route_awy_dist_km, sched_block_hours, perf_type, flt_no_cat,
  route (dep_iata-arr_iata), mtow, month, day_of_week

References
----------
v1.x equivalent: src/ml_prediction.py  (kept for reference — do not delete)
Model training:  ml_model_upgrade_v2/src/04_train.py
Feature parity:  ml_model_upgrade_v2/data/interim/feature_parity.yaml
"""

import numpy as np
import polars as pl
import joblib
import yaml

import config
from config import (
    DATE_PREFIX,
    INTERMEDIATE_DIRECTORY,
)

INPUT_FILE  = f"{DATE_PREFIX}_feature_engineering.csv"
OUTPUT_FILE = f"{DATE_PREFIX}_fuel_prediction.csv"

FUEL_TARGET     = "target_baseline_fuel_kg"
AIRBORNE_TARGET = "target_airborne_hours"


def _build_X(df: pl.DataFrame, feature_cols: list[str], encoders: dict) -> np.ndarray:
    """Transform Polars frame into numpy feature matrix.

    Matches the build_X() logic in 06_serve.py / 04_train.py exactly:
    - Categorical columns: OrdinalEncoder.transform (unseen → __MISSING__)
    - Numeric columns: cast Float64, fill null with median
    """
    arrays = []
    for col in feature_cols:
        series = df[col]
        if col in encoders:
            vals = series.fill_null("__MISSING__").cast(pl.Utf8).to_numpy().reshape(-1, 1)
            arrays.append(encoders[col].transform(vals).astype(np.float32).ravel())
        else:
            arr = series.cast(pl.Float64, strict=False)
            median = arr.median()
            if median is None:
                median = 0.0
            arrays.append(arr.fill_null(median).to_numpy().astype(np.float32))
    return np.column_stack(arrays)


def main():
    print("=" * 70)
    print("Stage 2  Fuel Prediction")
    print("=" * 70)

    # ── Load Stage 1 output ───────────────────────────────────────────────────
    in_path = f"{INTERMEDIATE_DIRECTORY}/{INPUT_FILE}"
    df = pl.read_csv(in_path, infer_schema_length=5000, null_values=["null", "NULL", ""])
    print(f"\nRows : {len(df):,}   Source: {in_path}")

    # ── Load feature parity contract ──────────────────────────────────────────
    print(f"Models dir    : {config.ML_MODELS_DIRECTORY}")
    parity_path = f"{config.ML_MODELS_DIRECTORY}/feature_parity.yaml"
    with open(parity_path) as f:
        parity = yaml.safe_load(f)
    feature_cols = parity["feature_cols"]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # ── Build categorical features missing from Stage 1 ───────────────────────
    # flt_no_cat: string version of flt_no
    # route:      dep_iata-arr_iata concatenation (matches training)
    df = df.with_columns([
        pl.col("flt_no").cast(pl.Utf8).alias("flt_no_cat"),
        pl.concat_str(["dep_iata", "arr_iata"], separator="-").alias("route"),
    ])

    # awy_dist_km from Stage 1 is the route distance; rename to match parity
    if "awy_dist_km" in df.columns and "route_awy_dist_km" not in df.columns:
        df = df.rename({"awy_dist_km": "route_awy_dist_km"})

    # max_tow from Stage 1; rename to mtow (feature parity name)
    if "max_tow" in df.columns and "mtow" not in df.columns:
        df = df.rename({"max_tow": "mtow"})

    # ── Verify all features are present ───────────────────────────────────────
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features required by parity contract: {missing}")

    # ── Load models and encoders ──────────────────────────────────────────────
    encoders    = joblib.load(f"{config.ML_MODELS_DIRECTORY}/label_encoders.joblib")
    fuel_bundle = joblib.load(f"{config.ML_MODELS_DIRECTORY}/pooled_lgbm_{FUEL_TARGET}.joblib")
    air_bundle  = joblib.load(f"{config.ML_MODELS_DIRECTORY}/pooled_lgbm_{AIRBORNE_TARGET}.joblib")

    print(f"\nFuel model    : pooled LGBM  "
          f"train_n={fuel_bundle['train_n']:,}  "
          f"test_MAPE={fuel_bundle['test_mape']:.2f}%")
    print(f"Airborne model: pooled LGBM  "
          f"train_n={air_bundle['train_n']:,}  "
          f"test_MAPE={air_bundle['test_mape']:.2f}%")

    # ── Build feature matrix and predict ─────────────────────────────────────
    X = _build_X(df, feature_cols, encoders)

    pred_baseline_fuel = np.clip(fuel_bundle["model"].predict(X), 0, None)
    pred_airborne      = np.clip(air_bundle["model"].predict(X),  0, None)

    perf_corr = df["avg_perf_corr"].cast(pl.Float64, strict=False).fill_null(1.0).to_numpy()
    pred_trip_fuel = np.clip(pred_baseline_fuel * perf_corr, 0, None)

    print(f"\npred_baseline_fuel_kg : "
          f"mean={pred_baseline_fuel.mean():.0f}  "
          f"min={pred_baseline_fuel.min():.0f}  "
          f"max={pred_baseline_fuel.max():.0f}")
    print(f"pred_trip_fuel_kg     : "
          f"mean={pred_trip_fuel.mean():.0f}  "
          f"min={pred_trip_fuel.min():.0f}  "
          f"max={pred_trip_fuel.max():.0f}")
    print(f"pred_airborne_hours   : "
          f"mean={pred_airborne.mean():.2f}  "
          f"min={pred_airborne.min():.2f}  "
          f"max={pred_airborne.max():.2f}")

    # ── Attach predictions and write ──────────────────────────────────────────
    df = df.hstack(pl.DataFrame({
        "pred_baseline_fuel_kg": pred_baseline_fuel,
        "pred_trip_fuel_kg":     pred_trip_fuel,
        "pred_airborne_hours":   pred_airborne,
    }))

    out_path = f"{INTERMEDIATE_DIRECTORY}/{OUTPUT_FILE}"
    df.write_csv(out_path)

    print(f"\nRows out : {len(df):,}")
    print(f"Written  : {out_path}")
    print("\nStage 2 Fuel Prediction complete.")


if __name__ == "__main__":
    main()
