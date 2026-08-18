"""
planning_ingest.py  —  v2.0.0  Stage 1: Planning Ingest

Reads the daily APM planning export (one row per flight leg) and produces a
clean, ML-ready flight-leg frame for Stage 2 (Fuel Prediction).

Responsibilities
----------------
- Load the APM CSV (produced by the Databricks SQL query in queries.py)
- Rename columns to pipeline convention
- Strip perf_type whitespace; flag unknown codes
- Parse ISO-8601 departure/arrival datetimes → flt_date + sched_block_hours
- Add calendar features: month, day_of_week
- Route distance: replace NULL or physically-impossible awy_dist_km
  (awy < haversine = corrupt LIDO record) with great-circle distance
- Write one row per flight leg to:
    {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_feature_engineering.csv

This stage does NOT:
  - Filter or drop any tails
  - Build LoF / Route strings
  - Aggregate to one row per tail
  - Join cost index

All of that happens in later stages.

References
----------
v1.x equivalent: src/databricks_joining.py  (kept for reference — do not delete)
SQL source:       src/queries.py  daily_input_query()
"""

import math

import polars as pl

from config import (
    DATE_PREFIX,
    INPUT_DIRECTORY,
    INTERMEDIATE_DIRECTORY,
)

OUTPUT_FILE = f"{DATE_PREFIX}_feature_engineering.csv"

VALID_PERF_TYPES = {"19111", "20214", "2014W", "20251", "21251"}


def _resolve_route_distance(df: pl.DataFrame) -> pl.DataFrame:
    """Replace NULL or impossible awy_dist_km with haversine great-circle distance.

    Impossible = awy_dist_km < haversine (physically impossible; indicates a
    corrupt LIDO record). Same safety logic as v1.x fill_gc_distance().
    """
    import airportsdata
    airports = airportsdata.load("IATA")
    coords = {
        code: (ap["lat"], ap["lon"])
        for code, ap in airports.items()
        if ap.get("lat") is not None and ap.get("lon") is not None
    }

    def _gc(dep, arr):
        if dep == arr or dep not in coords or arr not in coords:
            return None
        lat1, lon1 = coords[dep]
        lat2, lon2 = coords[arr]
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
             + math.cos(phi1) * math.cos(phi2)
             * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))

    awy = df["awy_dist_km"].to_list()
    deps = df["dep_iata"].to_list()
    arrs = df["arr_iata"].to_list()

    n_null = n_corrupt = n_filled = n_unresolved = 0
    resolved: list[float | None] = []

    for a_val, dep, arr in zip(awy, deps, arrs):
        gc = _gc(dep, arr)
        if a_val is None:
            n_null += 1
            resolved.append(gc if gc is not None else (n_unresolved := n_unresolved + 1, None)[1])
            if gc is not None:
                n_filled += 1
        elif gc is not None and a_val < gc:
            n_corrupt += 1
            resolved.append(gc)
            n_filled += 1
        else:
            resolved.append(a_val)

    print(f"  awy_dist_km: {n_null} NULL + {n_corrupt} corrupt (awy < GC) "
          f"-> {n_filled} replaced via GC, {n_unresolved} unresolved")

    return df.with_columns(pl.Series("awy_dist_km", resolved, dtype=pl.Float64))


def main():
    print("=" * 70)
    print("Stage 1  Planning Ingest")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    plan_path = f"{INPUT_DIRECTORY}/{DATE_PREFIX}_Opti_Tail_With_MTOW__APM_and_Seat_Capacity.csv"
    df = pl.read_csv(plan_path, infer_schema_length=10000, null_values=["null", "NULL", ""])
    print(f"\nRows: {len(df):,}   Source: {plan_path}")

    # ── Rename to pipeline convention ─────────────────────────────────────────
    df = df.rename({
        "tailnumber":          "aircraftreg",
        "flightnumber":        "flt_no",
        "departureairportcode": "dep_iata",
        "arrivalairportcode":   "arr_iata",
        "acft_mtow":           "max_tow",
        "perf_corr":           "avg_perf_corr",
        "aircraftcapacity":    "seat_config",
    })

    # ── perf_type: strip whitespace, warn unknowns ────────────────────────────
    df = df.with_columns(pl.col("perf_type").str.strip_chars())
    unknown = (
        df.filter(~pl.col("perf_type").is_in(list(VALID_PERF_TYPES)))
        ["perf_type"].drop_nulls().unique().to_list()
    )
    if unknown:
        n = df.filter(~pl.col("perf_type").is_in(list(VALID_PERF_TYPES))).shape[0]
        print(f"  WARNING: {n} rows with unknown perf_type (kept): {unknown}")

    # ── Parse datetimes → flt_date, sched_block_hours ─────────────────────────
    df = df.with_columns(
        pl.col("scheduledeparturedatetime").str.slice(0, 10)
          .str.to_date(strict=False).alias("flt_date")
    )

    # Strip the UTC offset before parsing so we don't depend on %z/%:z handling or
    # the use_utc/time_zone kwarg (both vary across Polars versions). All legs operate
    # in the same European timezone on any given day, so naive epoch is correct for
    # within-day chronological sort and sched_block_hours calculation.
    # Handles: +01:00, +0100, -05:00, Z
    def _epoch_expr(col: str) -> pl.Expr:
        stripped = pl.col(col).str.replace(r'(Z|[+-]\d{2}:?\d{2})$', '')
        return pl.coalesce([
            stripped.str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f", strict=False).dt.epoch(time_unit="s"),
            stripped.str.to_datetime(format="%Y-%m-%d %H:%M:%S",      strict=False).dt.epoch(time_unit="s"),
        ])

    df = df.with_columns([
        _epoch_expr("scheduledeparturedatetime").alias("_dep_s"),
        _epoch_expr("schedulearrivaldatetime").alias("_arr_s"),
    ])

    n_null_epoch = df.filter(pl.col("_dep_s").is_null()).height
    if n_null_epoch:
        print(f"  WARNING: {n_null_epoch}/{len(df)} rows have unparseable "
              f"scheduledeparturedatetime — chronological sort may be affected")

    df = df.with_columns(
        ((pl.col("_arr_s") - pl.col("_dep_s")) / 3600.0).alias("sched_block_hours"),
        pl.col("_dep_s").alias("dep_epoch"),
    ).drop(["_dep_s", "_arr_s"]).rename({
        "scheduledeparturedatetime": "sched_dep_datetime",
        "schedulearrivaldatetime":   "sched_arr_datetime",
    })

    # ── Calendar features ─────────────────────────────────────────────────────
    df = df.with_columns([
        pl.col("flt_date").dt.month().cast(pl.Int32).alias("month"),
        pl.col("flt_date").dt.weekday().cast(pl.Int32).alias("day_of_week"),
    ])

    # ── Route distance: GC safety layer ──────────────────────────────────────
    print("\nRoute distance resolution:")
    df = _resolve_route_distance(df)

    # ── Write ─────────────────────────────────────────────────────────────────
    out_path = f"{INTERMEDIATE_DIRECTORY}/{OUTPUT_FILE}"
    df.write_csv(out_path)

    print(f"\nRows out : {len(df):,}")
    print(f"Columns  : {df.columns}")
    print(f"Written  : {out_path}")
    print("\nStage 1 Planning Ingest complete.")


if __name__ == "__main__":
    main()
