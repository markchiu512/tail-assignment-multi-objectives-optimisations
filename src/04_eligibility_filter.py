"""
eligibility_filter.py  —  v2.0.0  Stage 4: Eligibility Filter

Reads the per-tail enriched frame from Stage 3 and produces the final
eligible set for the assignment optimiser.

Responsibilities
----------------
- Filter out tails that must not be optimised:
    - Wet-lease tails that fall outside the available cost model
    - LEASE_AIRCRAFTREG (config)
    - MANUAL_DROP_AIRCRAFTREG (config)
    - Tails on maintenance ground events (TOPS xlsx, non-ESP/HSP codes)
    - Odd-sector tails (sectors_even == False) — cannot swap mid-LoF
    - Out-station bases (base length != 3 or 5 chars) — unroutable
- Inject spare / standby aircraft (TOPS ESP/HSP codes + STANDBY_AIRCRAFT config)
  with cost index rates and avg_perf_corr defaulting to 1.0
- LGW terminal identification: flights 6300-6599 → base = "LGW-S"

Input  : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_lof_enrichment.csv
Output : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eligibility_filter.csv

References
----------
v1.x equivalent: src/joined_clean.py  (kept for reference — do not delete)
"""

import polars as pl

import json

from config import (
    DATE_PREFIX,
    MTL_DATE,
    INPUT_DIRECTORY,
    INTERMEDIATE_DIRECTORY,
    LEASE_AIRCRAFTREG,
    MANUAL_DROP_AIRCRAFTREG,
    GROUNDEVENTS_FILE,
    STANDBY_AIRCRAFT,
    GROUND_EVENTS_SOURCE,
)

INPUT_FILE  = f"{DATE_PREFIX}_lof_enrichment.csv"
OUTPUT_FILE = f"{DATE_PREFIX}_eligibility_filter.csv"


def _read_ground_events() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (df_maintenance, df_spare) from TOPS ground events.

    Source is controlled by GROUND_EVENTS_SOURCE in config:
      "databricks" — reads pre-fetched CSV from INTERMEDIATE_DIRECTORY
                     (written by run_pipeline_job or app before Stage 4 runs).
      "xls"        — reads manually uploaded TOPS XLS file (legacy path).
    Both paths produce the same (df_maintenance, df_spare) pair.
    """
    if GROUND_EVENTS_SOURCE == "databricks":
        csv_path = f"{INPUT_DIRECTORY}/{DATE_PREFIX}_ground_events_databricks.csv"
        df = pl.read_csv(csv_path, null_values=["null", "NULL", ""])
        df = df.rename({"tailnumber": "aircraftreg"})
        df = df.select(["aircraftreg", "checktypecode", "base"])
        df_spare = df.filter(pl.col("checktypecode").is_in(["ESP", "HSP"]))
        df_mtl   = df.filter(~pl.col("checktypecode").is_in(["ESP", "HSP"]))
        return df_mtl, df_spare

    # "xls" path — original manual file logic
    import shutil, tempfile, os
    src = f"{INPUT_DIRECTORY}/{GROUNDEVENTS_FILE}"
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy2(src, tmp_path)
        import pandas as pd
        df_pd = pd.read_excel(tmp_path)
        df = pl.from_pandas(df_pd)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    df = df.rename({
        "Act_Rgn_No":    "aircraftreg",
        "Apt_Iata_Code": "base",
    })
    df = df.filter(pl.col("Planned_Start_Date") == MTL_DATE)
    df = df.select(["aircraftreg", "Check_Typ_Code", "base"])

    df_spare = df.filter(pl.col("Check_Typ_Code").is_in(["ESP", "HSP"]))
    df_mtl   = df.filter(~pl.col("Check_Typ_Code").is_in(["ESP", "HSP"]))

    return df_mtl, df_spare


def _inject_spare_aircraft(
    df: pl.DataFrame,
    df_spare: pl.DataFrame,
    cost_path: str,
) -> pl.DataFrame:
    """Concat spare aircraft rows (already enriched tails keep their Stage 3 rates)."""
    df = df.with_columns(pl.col("avg_perf_corr").fill_null(1.0))

    if len(df_spare) == 0:
        return df

    # Join cost index only for the new spare rows — main fleet already has rates from Stage 3
    df_cost = pl.read_csv(cost_path)
    _cost_rename = {"Total FH Rate": "total_fh_rate", "Total Cycle Rate": "total_cycle_rate"}
    if "Type" in df_cost.columns and "aircrafttype" not in df_cost.columns:
        _cost_rename["Type"] = "aircrafttype"
    df_cost = df_cost.rename(_cost_rename)
    df_spare = df_spare.join(df_cost, on="aircraftreg", how="left")
    df_spare = df_spare.with_columns(pl.lit(1.0).alias("avg_perf_corr"))

    return pl.concat([df, df_spare], how="diagonal")


def _lgw_terminal(df: pl.DataFrame) -> pl.DataFrame:
    """Flag LGW South terminal routes (flight numbers 6300-6599)."""
    if "flt_numbers" not in df.columns:
        return df

    first_flt = (
        pl.col("flt_numbers").str.split("-").list.first()
        .cast(pl.Int64, strict=False)
    )
    df = df.with_columns(
        pl.when((first_flt >= 6300) & (first_flt <= 6599))
        .then(pl.lit("LGW-S"))
        .otherwise(pl.col("base"))
        .alias("base")
    )
    return df


def main():
    print("=" * 70)
    print("Stage 4  Eligibility Filter")
    print("=" * 70)

    # ── Load Stage 3 output ───────────────────────────────────────────────────
    in_path = f"{INTERMEDIATE_DIRECTORY}/{INPUT_FILE}"
    df = pl.read_csv(in_path, null_values=["null", "NULL", ""])
    n_in = len(df)
    print(f"\nTails in : {n_in}")

    # ── Load TOPS ground events ───────────────────────────────────────────────
    cost_path = f"{INPUT_DIRECTORY}/cost_index.csv"
    _empty_ge = pl.DataFrame(schema={"aircraftreg": pl.Utf8, "checktypecode": pl.Utf8, "base": pl.Utf8})
    print(f"\nGround events source: {GROUND_EVENTS_SOURCE}")
    try:
        df_mtl, df_spare = _read_ground_events()
        print(f"  Maintenance tails  : {df_mtl['aircraftreg'].n_unique()}")
        print(f"  Spare tails (TOPS) : {df_spare['aircraftreg'].n_unique()}")
        mtl_regs = df_mtl["aircraftreg"].to_list()
    except Exception as e:
        print(f"  WARNING: could not read ground events ({e}) — skipping maintenance filter")
        df_mtl = _empty_ge
        df_spare = _empty_ge
        mtl_regs = []

    # ── Build drop list ───────────────────────────────────────────────────────
    drop_regs = set(
        LEASE_AIRCRAFTREG
        + MANUAL_DROP_AIRCRAFTREG
        + mtl_regs
    )

    # Write ground event exclusions for Stage 6 summary report
    _ground_exclusions = {
        "source": GROUND_EVENTS_SOURCE,
        "maintenance": sorted(df_mtl["aircraftreg"].to_list()) if len(df_mtl) else [],
        "spare": sorted(df_spare["aircraftreg"].to_list()) if len(df_spare) else [],
    }
    _excl_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_ground_exclusions.json"
    with open(_excl_path, "w", encoding="utf-8") as _f:
        json.dump(_ground_exclusions, _f, indent=2)

    # ── Apply eligibility filters ─────────────────────────────────────────────
    df_eligible = df.filter(
        (~pl.col("aircraftreg").str.starts_with("EI"))
        & (~pl.col("aircraftreg").is_in(list(drop_regs)))
        & (pl.col("sectors_even") == True)
        & (pl.col("base").str.len_chars().is_in([3, 5]))
    )

    print(f"\nAfter filters:")
    print(f"  EI / dropped / maintenance / odd-sector / out-station removed")
    print(f"  Eligible tails : {len(df_eligible)}")

    # ── Inject TOPS spare aircraft ────────────────────────────────────────────
    df_eligible = _inject_spare_aircraft(df_eligible, df_spare, cost_path)

    # ── STANDBY_AIRCRAFT injection — TODO: implement when needed ─────────────

    # ── LGW terminal identification ───────────────────────────────────────────
    df_eligible = _lgw_terminal(df_eligible)

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = f"{INTERMEDIATE_DIRECTORY}/{OUTPUT_FILE}"
    df_eligible.write_csv(out_path)

    print(f"\nTails out : {len(df_eligible)}")
    print(f"Removed   : {n_in - len(df_eligible)}")
    print(f"Columns   : {df_eligible.columns}")
    print(f"Written   : {out_path}")
    print("\nStage 4 Eligibility Filter complete.")


if __name__ == "__main__":
    main()
