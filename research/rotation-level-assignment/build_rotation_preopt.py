"""
build_rotation_preopt_20260802.py
Research/shadow mode — no production files touched.

Builds rotation-level pre-optimisation table for A319 G-reg tails, 02 Aug 2026.
Sources:
  - 20260802_fuel_prediction.csv  (flight legs)
  - mtow_envelope.csv             (MTOW block-hour ceilings)
  - cost_index.csv                (fh_rate, cycle_rate per tail)

"""

import sys
import io
import os
import shutil
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import polars as pl

WORK_DIR = Path(os.environ.get("ROTATION_RESEARCH_DIR", Path(__file__).parent / "data"))
INPUT_CSV = WORK_DIR / "fuel_prediction.csv"
ENVELOPE_CSV = WORK_DIR / "mtow_envelope.csv"
COST_CSV = WORK_DIR / "cost_index.csv"
OUTPUT_CSV = WORK_DIR / "rotation_preopt.csv"

FUEL_PRICE = float(os.environ.get("FUEL_PRICE", "1.20"))
ANOMALY_BLOCK_HOURS = 6.0

# ── 1. Load and inspect ───────────────────────────────────────────────────────
raw = pl.read_csv(INPUT_CSV)
print("=" * 70)
print("SOURCE DATA CHECK")
print("=" * 70)
print(f"  Total rows in 20260802_fuel_prediction.csv: {len(raw)}")

# Confirm sched_dep_datetime / sched_arr_datetime presence and format
for col in ("sched_dep_datetime", "sched_arr_datetime"):
    if col not in raw.columns:
        print(f"\n  MISSING COLUMN: {col} — aborting.")
        sys.exit(1)

sample_dep = raw["sched_dep_datetime"][0]
sample_arr = raw["sched_arr_datetime"][0]
n_null_dep = raw["sched_dep_datetime"].null_count()
n_null_arr = raw["sched_arr_datetime"].null_count()

print(f"\n  sched_dep_datetime: present, sample='{sample_dep}', nulls={n_null_dep}")
print(f"  sched_arr_datetime: present, sample='{sample_arr}', nulls={n_null_arr}")
print(f"  Timezone offset detected: +01:00 (BST/CEST) from sample values")

if n_null_dep > 0 or n_null_arr > 0:
    print(f"\n  FLAG: Null datetimes found — dep_nulls={n_null_dep}, arr_nulls={n_null_arr}")
    null_rows = raw.filter(
        pl.col("sched_dep_datetime").is_null() | pl.col("sched_arr_datetime").is_null()
    )
    print(null_rows.select(["aircraftreg", "flt_no", "sched_dep_datetime", "sched_arr_datetime"]))

# ── 2. Filter ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FILTER")
print("=" * 70)

df_all_a319 = raw.filter(pl.col("perf_type").cast(pl.Utf8) == "19111")
print(f"  After perf_type==19111 (A319): {len(df_all_a319)} legs")

df = df_all_a319.filter(pl.col("aircraftreg").str.starts_with("G-"))
print(f"  After aircraftreg starts with 'G-': {len(df)} legs")

# ── 3. Parse datetimes — strip tz offset, parse as naive local ───────────────
# Offset is +01:00 (BST) throughout. We strip it and treat as local wall-clock
# time, consistent with how build_rotation_preopt.py handled the 31 Jul data.
# All times in this file are therefore BST (UTC+1).
for col in ("sched_dep_datetime", "sched_arr_datetime"):
    df = df.with_columns(
        pl.col(col)
          .str.replace(r"[+-]\d{2}:\d{2}$", "")
          .str.replace(r" ", "T")
          .str.to_datetime(format="%Y-%m-%dT%H:%M:%S", strict=False)
          .alias(col)
    )

# Check for any parse failures (nulls introduced by strict=False)
n_parse_fail_dep = df["sched_dep_datetime"].null_count()
n_parse_fail_arr = df["sched_arr_datetime"].null_count()
if n_parse_fail_dep > 0 or n_parse_fail_arr > 0:
    print(f"\n  FLAG: datetime parse failures — dep={n_parse_fail_dep}, arr={n_parse_fail_arr}")
    print(df.filter(
        pl.col("sched_dep_datetime").is_null() | pl.col("sched_arr_datetime").is_null()
    ).select(["aircraftreg", "flt_no", "dep_iata", "arr_iata"]))

# Flag anomaly legs
df = df.with_columns(
    (pl.col("sched_block_hours") > ANOMALY_BLOCK_HOURS).alias("is_anomaly")
)

# Sort per tail by departure time before pairing
df = df.sort(["aircraftreg", "sched_dep_datetime"])

# ── 4. Load envelope (A319 type_code=9) ───────────────────────────────────────
env = pl.read_csv(ENVELOPE_CSV).filter(pl.col("type_code") == 9)
env_lookup = {
    int(row["mtow"]): row["max_block_hours_decimal"]
    for row in env.iter_rows(named=True)
    if row["max_block_hours_decimal"] is not None
}
print(f"\n  A319 MTOW envelope: {env_lookup}")

# Sorted ascending so we find the smallest ceiling that covers the leg
env_sorted = sorted(env_lookup.items())  # [(64000, 2.75), (66000, 3.5), (68000, 4.4167)]

def min_mtow_for_block(max_leg_block: float):
    """Return smallest MTOW whose per-leg ceiling >= max_leg_block, or None if none covers it."""
    for mtow_val, ceiling in env_sorted:
        if max_leg_block <= ceiling:
            return mtow_val
    return None

# ── 5. Build rotations ────────────────────────────────────────────────────────
rotation_rows = []
unpaired_log  = []  # track unpaired legs with reason

for (aircraftreg,), grp in df.group_by(["aircraftreg"], maintain_order=True):
    legs = grp.sort("sched_dep_datetime").to_dicts()
    i = 0
    while i < len(legs):
        leg = legs[i]

        # Try to pair with next leg as out-and-back
        if i + 1 < len(legs):
            nxt = legs[i + 1]
            is_outback = (
                leg["arr_iata"] == nxt["dep_iata"] and
                nxt["arr_iata"] == leg["dep_iata"]
            )
        else:
            is_outback = False

        if is_outback:
            out_leg = leg
            in_leg  = nxt
            is_anom = out_leg["is_anomaly"] or in_leg["is_anomaly"]
            mtow_int = int(float(str(out_leg["mtow"])))
            env_max  = env_lookup.get(mtow_int)

            out_block = out_leg["sched_block_hours"]
            in_block  = in_leg["sched_block_hours"]
            max_leg   = max(out_block, in_block)

            if env_max is not None:
                eligible = (out_block <= env_max) and (in_block <= env_max)
                margin   = env_max - max_leg
            else:
                eligible = None
                margin   = None

            rotation_rows.append({
                "aircraftreg":                   aircraftreg,
                "flt_date":                      str(out_leg["flt_date"]),
                "type_flag":                     "rotation",
                "origin_iata":                   out_leg["dep_iata"],
                "away_iata":                     out_leg["arr_iata"],
                "outbound_flt_no":               out_leg["flt_no"],
                "inbound_flt_no":                in_leg["flt_no"],
                "rotation_start":                str(out_leg["sched_dep_datetime"]),
                "away_arr_datetime":             str(out_leg["sched_arr_datetime"]),
                "away_dep_datetime":             str(in_leg["sched_dep_datetime"]),
                "rotation_end":                  str(in_leg["sched_arr_datetime"]),
                "outbound_block_hours":          out_block,
                "inbound_block_hours":           in_block,
                "total_block_hours":             out_block + in_block,
                "outbound_pred_baseline_fuel_kg": out_leg["pred_baseline_fuel_kg"],
                "inbound_pred_baseline_fuel_kg":  in_leg["pred_baseline_fuel_kg"],
                "total_pred_baseline_fuel_kg":    out_leg["pred_baseline_fuel_kg"] + in_leg["pred_baseline_fuel_kg"],
                "outbound_pred_airborne_hours":  out_leg["pred_airborne_hours"],
                "inbound_pred_airborne_hours":   in_leg["pred_airborne_hours"],
                "total_pred_airborne_hours":     out_leg["pred_airborne_hours"] + in_leg["pred_airborne_hours"],
                "mtow":                          mtow_int,
                "avg_perf_corr":                 out_leg["avg_perf_corr"],
                "route_awy_dist_km_out":         out_leg["route_awy_dist_km"],
                "route_awy_dist_km_in":          in_leg["route_awy_dist_km"],
                "seat_config":                   out_leg["seat_config"],
                "envelope_max_block_hours":      env_max,
                "eligible_perleg":               eligible,
                "margin_block":                  margin,
                "min_mtow":                      min_mtow_for_block(max_leg),
                "is_anomaly":                    is_anom,
                "swappable":                     (not is_anom) and (eligible is True),
            })
            i += 2

        else:
            # Standalone leg — determine reason
            leg_block = leg["sched_block_hours"]
            mtow_int  = int(float(str(leg["mtow"])))
            env_max   = env_lookup.get(mtow_int)
            is_anom   = leg["is_anomaly"]

            if env_max is not None:
                eligible = leg_block <= env_max
                margin   = env_max - leg_block
            else:
                eligible = None
                margin   = None

            # Determine reason for being unpaired
            if i + 1 >= len(legs):
                unpaired_reason = "last_leg_no_return"
            else:
                nxt = legs[i + 1]
                if leg["arr_iata"] != nxt["dep_iata"]:
                    unpaired_reason = f"next_leg_different_origin({nxt['dep_iata']}!={leg['arr_iata']})"
                elif nxt["arr_iata"] != leg["dep_iata"]:
                    unpaired_reason = f"next_leg_no_return_to_base({nxt['arr_iata']}!={leg['dep_iata']})"
                else:
                    unpaired_reason = "unknown"

            unpaired_log.append({
                "aircraftreg": aircraftreg,
                "flt_no": leg["flt_no"],
                "dep_iata": leg["dep_iata"],
                "arr_iata": leg["arr_iata"],
                "rotation_start": str(leg["sched_dep_datetime"]),
                "rotation_end": str(leg["sched_arr_datetime"]),
                "reason": unpaired_reason,
            })

            rotation_rows.append({
                "aircraftreg":                   aircraftreg,
                "flt_date":                      str(leg["flt_date"]),
                "type_flag":                     "single_leg",
                "origin_iata":                   leg["dep_iata"],
                "away_iata":                     leg["arr_iata"],
                "outbound_flt_no":               leg["flt_no"],
                "inbound_flt_no":                None,
                "rotation_start":                str(leg["sched_dep_datetime"]),
                "away_arr_datetime":             str(leg["sched_arr_datetime"]),
                "away_dep_datetime":             None,
                "rotation_end":                  str(leg["sched_arr_datetime"]),
                "outbound_block_hours":          leg_block,
                "inbound_block_hours":           None,
                "total_block_hours":             leg_block,
                "outbound_pred_baseline_fuel_kg": leg["pred_baseline_fuel_kg"],
                "inbound_pred_baseline_fuel_kg":  None,
                "total_pred_baseline_fuel_kg":    leg["pred_baseline_fuel_kg"],
                "outbound_pred_airborne_hours":  leg["pred_airborne_hours"],
                "inbound_pred_airborne_hours":   None,
                "total_pred_airborne_hours":     leg["pred_airborne_hours"],
                "mtow":                          mtow_int,
                "avg_perf_corr":                 leg["avg_perf_corr"],
                "route_awy_dist_km_out":         leg["route_awy_dist_km"],
                "route_awy_dist_km_in":          None,
                "seat_config":                   leg["seat_config"],
                "envelope_max_block_hours":      env_max,
                "eligible_perleg":               eligible,
                "margin_block":                  margin,
                "min_mtow":                      min_mtow_for_block(leg_block),
                "is_anomaly":                    is_anom,
                "swappable":                     False,
            })
            i += 1

result = pl.DataFrame(rotation_rows)

# ── 6. Join cost index ────────────────────────────────────────────────────────
cost_df = pl.read_csv(COST_CSV).select([
    pl.col("aircraftreg"),
    pl.col("Total FH Rate").alias("fh_rate"),
    pl.col("Total Cycle Rate").alias("cycle_rate"),
])
result = result.join(cost_df, on="aircraftreg", how="left")

missing_cost_tails = (
    result.filter(pl.col("fh_rate").is_null())["aircraftreg"].unique().to_list()
)
if missing_cost_tails:
    n_missing = result.filter(pl.col("fh_rate").is_null()).height
    print(f"\n  FLAG: {n_missing} rows missing from cost_index: {sorted(missing_cost_tails)}")
else:
    print("\n  Cost index: all tails matched.")

# ── 7. Cost formula ───────────────────────────────────────────────────────────
result = result.with_columns(
    pl.when(pl.col("type_flag") == "rotation")
      .then(pl.lit(2))
      .otherwise(pl.lit(1))
      .alias("n_cycles")
)
result = result.with_columns([
    (pl.col("avg_perf_corr") * FUEL_PRICE * pl.col("total_pred_baseline_fuel_kg")).alias("fuel_cost"),
    (pl.col("fh_rate") * pl.col("total_pred_airborne_hours")).alias("fh_cost"),
    (pl.col("cycle_rate") * pl.col("n_cycles")).alias("cycle_cost"),
])
result = result.with_columns(
    (pl.col("fuel_cost") + pl.col("fh_cost") + pl.col("cycle_cost")).alias("total_cost")
)

# ── 8. Report ─────────────────────────────────────────────────────────────────
n_rotations  = result.filter(pl.col("type_flag") == "rotation").height
n_standalone = result.filter(pl.col("type_flag") == "single_leg").height
n_anomaly    = result.filter(pl.col("is_anomaly")).height
n_ineligible = result.filter(pl.col("eligible_perleg") == False).height
n_swappable  = result.filter(pl.col("swappable")).height
n_tails      = result["aircraftreg"].n_unique()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Input legs (A319 G-reg):  {len(df)}")
print(f"  Output rows:              {len(result)}")
print(f"    Rotations:              {n_rotations}")
print(f"    Single legs:            {n_standalone}")
print(f"  Unique tails:             {n_tails}")
print(f"  Anomaly rows (block>{ANOMALY_BLOCK_HOURS}h): {n_anomaly}")
print(f"  Ineligible per-leg:       {n_ineligible}")
print(f"  Swappable rotations:      {n_swappable}")

print("\n  MTOW distribution:")
mtow_dist = (
    result.group_by("mtow")
          .agg(pl.len().alias("n_rows"),
               pl.col("type_flag").filter(pl.col("type_flag") == "rotation").len().alias("n_rotations"))
          .sort("mtow")
)
for row in mtow_dist.iter_rows(named=True):
    print(f"    mtow={row['mtow']:,}  rows={row['n_rows']}  rotations={row['n_rotations']}")

if unpaired_log:
    print(f"\n  Unpaired legs ({len(unpaired_log)}):")
    for u in unpaired_log:
        print(f"    {u['aircraftreg']}  flt {u['flt_no']}  "
              f"{u['dep_iata']}→{u['arr_iata']}  "
              f"{u['rotation_start']}–{u['rotation_end']}  "
              f"reason: {u['reason']}")
else:
    print("\n  Unpaired legs: none")

print("\n  Ineligible rotations (envelope violated):")
inel = result.filter(pl.col("eligible_perleg") == False)
if len(inel) == 0:
    print("    none")
else:
    for row in inel.iter_rows(named=True):
        print(f"    {row['aircraftreg']}  {row['origin_iata']}-{row['away_iata']}  "
              f"out={row['outbound_block_hours']:.3f}h  in={row.get('inbound_block_hours') or 'N/A'}h  "
              f"envelope={row['envelope_max_block_hours']}h  margin={row['margin_block']:.3f}h")

# ── 9. Gantt-style timeline ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TIMELINE — A319 G-reg tails, 02 Aug 2026 (times in BST / local wall clock)")
print("=" * 70)

# Build hour axis labels
HOURS = list(range(4, 26))  # 04:00 to 01:00 next day (25=01:00+1)
BAR_WIDTH = 96  # characters for the 04:00–25:00 span = 21 hours
SCALE_START = 4 * 60   # 04:00 in minutes from midnight
SCALE_END   = 25 * 60  # 01:00 next day in minutes from midnight
SCALE_SPAN  = SCALE_END - SCALE_START  # 21 * 60 = 1260 minutes

def time_to_pos(dt_str: str) -> float:
    """Convert datetime string to minutes from midnight, clamped to scale."""
    try:
        t = dt_str[11:16]  # HH:MM
        h, m = int(t[:2]), int(t[3:5])
        mins = h * 60 + m
        # Handle next-day arrivals (e.g. 01:09 → treat as 25:09)
        if mins < SCALE_START:
            mins += 24 * 60
        return mins
    except Exception:
        return SCALE_START

def render_bar(start_str: str, end_str: str, label: str, swappable: bool, type_flag: str) -> str:
    start_min = time_to_pos(start_str)
    end_min   = time_to_pos(end_str)
    start_pos = int((start_min - SCALE_START) / SCALE_SPAN * BAR_WIDTH)
    end_pos   = int((end_min   - SCALE_START) / SCALE_SPAN * BAR_WIDTH)
    start_pos = max(0, min(BAR_WIDTH - 1, start_pos))
    end_pos   = max(start_pos + 1, min(BAR_WIDTH, end_pos))
    bar_len   = end_pos - start_pos

    if type_flag == "single_leg":
        fill = "~"
        bracket = ("(", ")")
    elif swappable:
        fill = "="
        bracket = ("[", "]")
    else:
        fill = "-"
        bracket = ("{", "}")

    inner = label[:bar_len - 2] if bar_len > 2 else label[:bar_len]
    if bar_len >= 2:
        bar = bracket[0] + inner.center(bar_len - 2, fill) + bracket[1]
    else:
        bar = bracket[0] * bar_len

    line = " " * start_pos + bar + " " * (BAR_WIDTH - end_pos)
    return line

# Hour ruler
ruler_parts = []
for h in range(4, 26):
    label = f"{h % 24:02d}"
    pos = int((h * 60 - SCALE_START) / SCALE_SPAN * BAR_WIDTH)
    ruler_parts.append((pos, label))

ruler = list(" " * BAR_WIDTH)
for pos, label in ruler_parts:
    for j, ch in enumerate(label):
        idx = pos + j
        if 0 <= idx < BAR_WIDTH:
            ruler[idx] = ch
ruler_str = "".join(ruler)

tick = list(" " * BAR_WIDTH)
for pos, _ in ruler_parts:
    if 0 <= pos < BAR_WIDTH:
        tick[pos] = "|"
tick_str = "".join(tick)

tail_col_w = 10
print(f"\n  {'TAIL':<{tail_col_w}}  {ruler_str}")
print(f"  {'':>{tail_col_w}}  {tick_str}")
print(f"  Legend: [=ROUTE=] swappable rotation   {{-ROUTE-}} non-swappable   (~leg~) single leg")
print()

# Group by tail, sort tails alphabetically for clean display
tails_sorted = sorted(result["aircraftreg"].unique().to_list())

for tail in tails_sorted:
    tail_rows = result.filter(pl.col("aircraftreg") == tail).sort("rotation_start")
    line_parts = [" "] * BAR_WIDTH
    labels_parts = []

    for row in tail_rows.iter_rows(named=True):
        route_label = f"{row['origin_iata']}-{row['away_iata']}"
        bar_str = render_bar(
            row["rotation_start"], row["rotation_end"],
            route_label, row["swappable"], row["type_flag"]
        )
        # Merge bar into line_parts (non-space chars win)
        for k, ch in enumerate(bar_str):
            if ch != " ":
                line_parts[k] = ch

    print(f"  {tail:<{tail_col_w}}  {''.join(line_parts)}")

print()

# ── 10. Write output ──────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
    tmp_path = tmp.name
try:
    result.write_csv(tmp_path)
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    shutil.move(tmp_path, OUTPUT_CSV)
except Exception:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    raise

print(f"Written: {OUTPUT_CSV}  ({len(result)} rows)")
print(f"Columns: {result.columns}")
