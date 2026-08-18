"""
cost_enrichment.py  —  v2.0.0  Stage 3: Cost Enrichment

Aggregates the per-leg ML predictions from Stage 2 into one row per tail
(Line of Flight level), then joins FH and cycle rates from the cost index.

Input  : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_fuel_prediction.csv
         (1 row per flight leg, with pred_baseline_fuel_kg + pred_airborne_hours)

Output : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_lof_enrichment.csv
         (1 row per tail — ready for Stage 4 eligibility filter)

Aggregations per tail (sorted by departure time):
  Route                  — full LoF string e.g. "LGW-AMS-CDG-LGW"
  base                   — first departure airport; "A to B" when out-station
  total_sectors          — number of legs
  sectors_even           — True when tail returns to starting base
  total_block_time_hours — sum of sched_block_hours across all legs
  flt_numbers            — hyphen-joined flight numbers in departure order
  total_baseline_fuel    — sum of pred_baseline_fuel_kg across all legs
  total_trip_fuel        — sum of pred_trip_fuel_kg across all legs
  total_deg_burn         — total_trip_fuel - total_baseline_fuel
  total_airborne_hours   — sum of pred_airborne_hours across all legs
  avg_perf_corr          — taken from last leg (stable per tail per day)
  max_tow, seat_config   — taken from last leg

Cost index join:
  Joins cost_index.csv on aircraftreg → Total FH Rate, Total Cycle Rate,
  aircrafttype (Type column), plus AOC, Engine Category, Sharklets metadata.

References
----------
v1.x equivalent: src/databricks_joining.py + src/joined_clean.py (cost index part)
  (both kept for reference — do not delete)
"""

import polars as pl

from config import (
    DATE_PREFIX,
    INPUT_DIRECTORY,
    INTERMEDIATE_DIRECTORY,
)

INPUT_FILE  = f"{DATE_PREFIX}_fuel_prediction.csv"
OUTPUT_FILE = f"{DATE_PREFIX}_lof_enrichment.csv"


def _build_lof(df: pl.DataFrame) -> pl.DataFrame:
    """Sort legs by departure epoch, then build per-tail LoF aggregates."""

    df = df.sort(["aircraftreg", "_dep_epoch"])

    # ── Route string ──────────────────────────────────────────────────────────
    routes = (
        df.group_by("aircraftreg", maintain_order=True)
        .agg(
            pl.concat_str([
                pl.col("dep_iata").first(),
                pl.lit("-"),
                pl.col("arr_iata").str.concat("-"),
            ]).alias("Route")
        )
    )
    df = df.join(routes, on="aircraftreg", how="left")

    # ── Sector count + even check ─────────────────────────────────────────────
    df = df.with_columns(
        pl.col("Route").str.count_matches("-").alias("total_sectors")
    ).with_columns(
        (pl.col("total_sectors") % 2 == 0).alias("sectors_even")
    )

    # ── Base ──────────────────────────────────────────────────────────────────
    df = df.with_columns([
        pl.col("Route").str.slice(0, 3).alias("base"),
        pl.col("Route").str.slice(-3).alias("_final_base"),
    ]).with_columns(
        pl.when(pl.col("base") != pl.col("_final_base"))
        .then(pl.concat_str([pl.col("base"), pl.lit(" to "), pl.col("_final_base")]))
        .otherwise(pl.col("base"))
        .alias("base")
    ).drop("_final_base")

    # ── Per-tail fuel / time sums ─────────────────────────────────────────────
    sums = (
        df.group_by("aircraftreg")
        .agg([
            pl.col("pred_baseline_fuel_kg").cast(pl.Float64, strict=False).sum().alias("total_baseline_fuel"),
            pl.col("pred_trip_fuel_kg").cast(pl.Float64, strict=False).sum().alias("total_trip_fuel"),
            pl.col("pred_airborne_hours").cast(pl.Float64, strict=False).sum().alias("total_pred_act_hours"),
            pl.col("sched_block_hours").cast(pl.Float64, strict=False).sum().alias("total_block_time_hours"),
            pl.col("route_awy_dist_km").cast(pl.Float64, strict=False).sum().alias("lof_distance"),
        ])
    )
    df = df.join(sums, on="aircraftreg", how="left")

    df = df.with_columns(
        (pl.col("total_trip_fuel") - pl.col("total_baseline_fuel")).alias("total_deg_burn")
    )

    # ── Flight numbers (sorted by departure epoch) ────────────────────────────
    flt_nums = (
        df.sort(["aircraftreg", "_dep_epoch"])
        .group_by("aircraftreg", maintain_order=True)
        .agg(
            pl.col("flt_no").cast(pl.Int64, strict=False)
              .cast(pl.Utf8).str.concat("-").alias("flt_numbers")
        )
    )
    df = df.join(flt_nums, on="aircraftreg", how="left")

    # ── One row per tail (keep last leg for representative scalar values) ─────
    df = df.unique(subset=["aircraftreg"], keep="last")

    return df


def main():
    print("=" * 70)
    print("Stage 3  Cost Enrichment")
    print("=" * 70)

    # ── Load Stage 2 output ───────────────────────────────────────────────────
    in_path = f"{INTERMEDIATE_DIRECTORY}/{INPUT_FILE}"
    df = pl.read_csv(in_path, infer_schema_length=5000, null_values=["null", "NULL", ""])
    print(f"\nLegs in : {len(df):,}   Source: {in_path}")

    # ── Use dep_epoch from Stage 1 for correct chronological sort ────────────
    df = df.with_columns(pl.col("dep_epoch").alias("_dep_epoch"))

    # ── Build LoF aggregates → one row per tail ───────────────────────────────
    df = _build_lof(df)
    print(f"Tails   : {len(df):,}")

    # ── Drop internal columns ─────────────────────────────────────────────────
    df = df.drop(["_dep_epoch", "dep_epoch"])

    # ── Join cost index ───────────────────────────────────────────────────────
    cost_path = f"{INPUT_DIRECTORY}/cost_index.csv"
    df_cost = pl.read_csv(cost_path)
    _cost_rename = {"Total FH Rate": "total_fh_rate", "Total Cycle Rate": "total_cycle_rate"}
    if "Type" in df_cost.columns and "aircrafttype" not in df_cost.columns:
        _cost_rename["Type"] = "aircrafttype"
    df_cost = df_cost.rename(_cost_rename)
    df = df.join(df_cost, on="aircraftreg", how="left")

    n_missing_cost = df["total_fh_rate"].null_count()
    if n_missing_cost:
        print(f"  WARNING: {n_missing_cost} tails not found in cost_index — fh/cycle rates will be null")

    # ── Drop leg-level and redundant columns not needed downstream ───────────
    drop_cols = [
        "row_num", "flt_no", "flightstatuscode", "dep_iata", "arr_iata", "latest_lido_data",
        "month", "day_of_week", "flt_no_cat", "route", "Sharklets", "AOC",
        "route_awy_dist_km", "sched_block_hours",
        "pred_baseline_fuel_kg", "pred_trip_fuel_kg", "pred_airborne_hours",
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = f"{INTERMEDIATE_DIRECTORY}/{OUTPUT_FILE}"
    df.write_csv(out_path)

    print(f"Columns : {df.columns}")
    print(f"Written : {out_path}")
    print("\nStage 3 Cost Enrichment complete.")


if __name__ == "__main__":
    main()
