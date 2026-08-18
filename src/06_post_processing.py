"""
swap_approval.py  —  v2.0.0  Stage 6: Swap Approval

Reads the full assignment results from Stage 5, decomposes independent swap
cycles within each group, applies the savings threshold, and produces the
fleet-delivery-ready swap list.

Responsibilities
----------------
- Decompose swap subgroups: detect independent cycles within each group_index
  and assign sub-group suffixes (a, b, c, ...) so fleet delivery can approve
  individual cycles without needing to approve the whole group
- Read ε* metadata from the eps_sweep_summary.json written by Stage 5
- Compute effective threshold:
    AUTO_THRESHOLD = True  → Kneedle elbow on composite benefit distribution
    AUTO_THRESHOLD = False → MIN_SAVINGS_THRESHOLD (fixed dollar cutoff)
- Filter to groups above threshold and changed=True rows only
- Sort by group savings descending (highest-value groups first)
- Write summary report (.txt) and cleaned swap list (.csv)

Input  : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_assignment_optimisation.csv
         {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_summary.json  (optional)
         {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_skipped_groups.json      (optional)

Output : {OUTPUT_DIRECTORY}/{DATE_PREFIX}_swap_approval.csv
         {OUTPUT_DIRECTORY}/{DATE_PREFIX}_swap_approval_summary.txt

References
----------
v1.x equivalent: src/post_processing.py  (kept for reference — do not delete)
"""

import json
import os

import polars as pl

from config import (
    AUTO_SELECT_EPS,
    AUTO_SELECT_EPS_METHOD,
    AUTO_THRESHOLD,
    DATE_PREFIX,
    FUEL_BIAS,
    FUEL_PRICE,
    GROUND_EVENTS_SOURCE,
    GROUNDEVENTS_FILE,
    INTERMEDIATE_DIRECTORY,
    MIN_SAVINGS_THRESHOLD,
    OUTPUT_DIRECTORY,
    SOLVER_BACKEND,
)

INPUT_FILE              = f"{DATE_PREFIX}_assignment_optimisation.csv"
OUTPUT_CSV              = f"{DATE_PREFIX}_swap_approval.csv"
OUTPUT_SUMMARY          = f"{DATE_PREFIX}_swap_approval_summary.txt"
EPS_SUMMARY_JSON        = f"{DATE_PREFIX}_eps_sweep_summary.json"
SKIPPED_GROUPS_JSON     = f"{DATE_PREFIX}_skipped_groups.json"
GROUND_EXCLUSIONS_JSON  = f"{DATE_PREFIX}_ground_exclusions.json"


# ── Kneedle threshold helper ──────────────────────────────────────────────────

def _kneedle_threshold(values: list[float]) -> float:
    """Elbow of the sorted-descending composite benefit cumulative curve."""
    if len(values) < 2:
        return 0.0

    vals = sorted(values, reverse=True)
    total = sum(vals)
    if total == 0:
        return 0.0

    cumulative, running = [], 0.0
    for v in vals:
        running += v
        cumulative.append(running)

    n = len(vals)
    xs = [i / (n - 1) for i in range(n)]
    ys = [c / total for c in cumulative]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_max == x_min or y_max == y_min:
        return vals[0]

    xs_n = [(x - x_min) / (x_max - x_min) for x in xs]
    ys_n = [(y - y_min) / (y_max - y_min) for y in ys]

    best_idx = max(range(n), key=lambda i: ys_n[i] - xs_n[i])
    return vals[best_idx]


# ── Swap subgroup decomposition ───────────────────────────────────────────────

def decompose_swap_subgroups(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect independent swap cycles within each group_index and assign
    sub-group suffixes (a, b, c, ...).

    A group of 5 changed tails might contain a 2-way swap (A<->B) and a
    3-way swap (C->D->E->C). These are independent — fleet delivery can
    approve one without the other. Labels them as e.g. LGW_003a, LGW_003b.
    """
    new_group_indices = df["group_index"].to_list()
    aircraftreg       = df["aircraftreg"].to_list()
    opti_aircraftreg  = df["opti_aircraftreg"].to_list()
    group_indices     = df["group_index"].to_list()

    for grp in set(group_indices):
        row_idxs     = [i for i, g in enumerate(group_indices) if g == grp]
        changed_idxs = [i for i in row_idxs if aircraftreg[i] != opti_aircraftreg[i]]

        if len(changed_idxs) <= 1:
            continue

        reg_to_row = {aircraftreg[i]: i for i in row_idxs}

        visited, cycles = set(), []
        for start in changed_idxs:
            if start in visited:
                continue
            cycle, current = [], start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                next_reg = opti_aircraftreg[current]
                current = reg_to_row.get(next_reg, current)
            if cycle:
                cycles.append(cycle)

        if len(cycles) <= 1:
            continue

        for suffix_idx, cycle in enumerate(cycles):
            suffix = chr(ord("a") + suffix_idx)
            for row_idx in cycle:
                new_group_indices[row_idx] = f"{grp}_{suffix}"

    return df.with_columns(pl.Series("group_index", new_group_indices))


# ── Summary report ────────────────────────────────────────────────────────────

def _write_summary(
    df: pl.DataFrame,
    group_summary: pl.DataFrame,
    eps_applied,
    eps_star,
    auto_select_method: str,
    effective_threshold: float,
) -> None:
    total_current = df["current_cost"].sum()
    total_opti    = df["opti_cost"].sum()
    total_savings = total_current - total_opti
    fuel_delta    = df["fuel_delta"].sum()
    n_changed     = df["changed"].sum()
    n_groups      = df["group_index"].n_unique()

    out_path = f"{OUTPUT_DIRECTORY}/{OUTPUT_SUMMARY}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CONFIGURATION:\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Input File          : {INPUT_FILE}\n")
        if GROUND_EVENTS_SOURCE == "databricks":
            f.write(f"  Ground Events       : silver_stream_tops.ground_activity_details (auto)\n")
        else:
            f.write(f"  Ground Events File  : {GROUNDEVENTS_FILE} (manual XLS)\n")
        f.write(f"  Fuel Price          : ${FUEL_PRICE} per kg\n")
        f.write(f"  Fuel Bias           : {FUEL_BIAS:.2f}x\n")

        if eps_applied is not None:
            if AUTO_SELECT_EPS and eps_star is not None:
                _labels = {"kneedle": "Kneedle", "chord": "Max-Chord", "curvature": "Max-k"}
                label = _labels.get(auto_select_method or "kneedle",
                                    (auto_select_method or "kneedle").title())
                f.write(f"  eps Applied         : {eps_applied*100:.2f}% "
                        f"(nearest grid to {label} eps*={eps_star*100:.4f}%, auto)\n")
            else:
                f.write(f"  eps Applied         : {eps_applied*100:.2f}% (manual)\n")
        else:
            f.write(f"  eps Applied         : None (cost-optimal)\n")

        if AUTO_THRESHOLD:
            f.write(f"  Savings Threshold   : ${effective_threshold:,.0f} "
                    f"(auto — Kneedle on composite benefit)\n")
        else:
            f.write(f"  Savings Threshold   : ${MIN_SAVINGS_THRESHOLD} (manual)\n")

        _solver_label = "Gurobi Cluster Manager" if SOLVER_BACKEND == "gurobi" else "Pyomo + HiGHS"
        f.write(f"  ML Model            : LightGBM (pooled v2)\n")
        f.write(f"  MIP Solver          : {_solver_label}\n\n")

        f.write("=" * 80 + "\n")
        f.write("OVERALL RESULTS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Total Current Cost      : ${total_current:,.2f}\n")
        f.write(f"  Total Optimised Cost    : ${total_opti:,.2f}\n")
        f.write(f"  Total Savings           : ${total_savings:,.2f}\n")
        if total_current > 0:
            f.write(f"  Savings %               : {total_savings / total_current * 100:.2f}%\n")
        f.write(f"  Fuel Savings            : {fuel_delta:.2f} kg\n")
        if total_savings > 0:
            f.write(f"  Fuel Savings % of Total : {fuel_delta * FUEL_PRICE / total_savings * 100:,.2f}%\n")
        f.write(f"  Total Reassignments     : {n_changed}\n")
        f.write(f"  Total Groups Optimised  : {n_groups}\n")

        # Ground event exclusions
        excl_path = f"{INTERMEDIATE_DIRECTORY}/{GROUND_EXCLUSIONS_JSON}"
        if os.path.exists(excl_path):
            try:
                with open(excl_path) as _ef:
                    excl = json.load(_ef)
                mtl_list  = excl.get("maintenance", [])
                spare_list = excl.get("spare", [])
                src_label  = "Databricks table" if excl.get("source") == "databricks" else "Manual XLS"
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"GROUND EVENTS EXCLUDED  (source: {src_label}):\n")
                f.write("=" * 80 + "\n")
                f.write(f"  Maintenance ({len(mtl_list)} tails):\n")
                if mtl_list:
                    for reg in mtl_list:
                        f.write(f"    {reg}\n")
                else:
                    f.write("    (none)\n")
                f.write(f"  Spare / Standby ({len(spare_list)} tails):\n")
                if spare_list:
                    for reg in spare_list:
                        f.write(f"    {reg}\n")
                else:
                    f.write("    (none)\n")
            except Exception:
                pass

        f.write("\n" + "=" * 80 + "\n")
        f.write("SAVINGS BY GROUP INDEX (sorted by savings):\n")
        f.write("=" * 80 + "\n")
        for row in group_summary.iter_rows(named=True):
            f.write(
                f"  {row['group_index']:<15}  "
                f"Savings: ${row['savings']:>10,.2f}   "
                f"Fuel Savings: {row['fuel_delta']:>10,.3f} kg   "
                f"Reassignments: {int(row['changed']):>3}\n"
            )

        # Skipped groups
        skipped_path = f"{INTERMEDIATE_DIRECTORY}/{SKIPPED_GROUPS_JSON}"
        if os.path.exists(skipped_path):
            with open(skipped_path) as sf:
                skipped = json.load(sf)
            if skipped:
                f.write("\n" + "=" * 80 + "\n")
                f.write("SKIPPED GROUPS (INFEASIBLE CONSTRAINTS):\n")
                f.write("=" * 80 + "\n")
                for g in skipped:
                    f.write(f"  {g['group_index']:<12}  Reason: {g['reason']}\n")

    print(f"Summary written  : {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Stage 6  Swap Approval")
    print("=" * 80)

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # ── Load Stage 5 output ───────────────────────────────────────────────────
    in_path = f"{INTERMEDIATE_DIRECTORY}/{INPUT_FILE}"
    df = pl.read_csv(in_path)
    print(f"\nRows in  : {len(df)}   groups: {df['group_index'].n_unique()}")

    # ── Decompose swap subgroups ──────────────────────────────────────────────
    df = decompose_swap_subgroups(df)
    print(f"Groups after decomposition : {df['group_index'].n_unique()}")

    # ── Read eps metadata from Stage 5 ────────────────────────────────────────
    eps_applied = None
    eps_star = None
    auto_select_method = "kneedle"
    sweep_path = f"{INTERMEDIATE_DIRECTORY}/{EPS_SUMMARY_JSON}"
    if os.path.exists(sweep_path):
        try:
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            first = next(iter(sweep_data.values()), {})
            eps_applied        = first.get("auto_selected_eps")
            eps_star           = first.get("eps_star") or first.get("kneedle_eps_star")
            auto_select_method = first.get("auto_select_method", "kneedle")
        except Exception:
            pass

    # ── Compute effective threshold ───────────────────────────────────────────
    effective_threshold = float(MIN_SAVINGS_THRESHOLD)
    b_k_threshold = None

    if AUTO_THRESHOLD:
        grp_agg = df.group_by("group_index").agg(
            pl.col("savings").sum(),
            pl.col("fuel_delta").sum(),
        )
        b_k_values = grp_agg.with_columns(
            (pl.col("savings") + (FUEL_BIAS - 1.0) * pl.col("fuel_delta") * FUEL_PRICE)
            .alias("b_k")
        )["b_k"].to_list()
        effective_threshold = _kneedle_threshold(b_k_values)
        b_k_threshold = effective_threshold
        print(f"Auto threshold (Kneedle) : ${effective_threshold:,.0f} composite benefit "
              f"(FUEL_BIAS={FUEL_BIAS})")
    else:
        print(f"Manual threshold         : ${MIN_SAVINGS_THRESHOLD}")

    # ── Group-level summary (before threshold filter, for report) ─────────────
    group_summary = (
        df.group_by("group_index")
        .agg(
            pl.col("savings").sum(),
            pl.col("changed").sum(),
            pl.col("fuel_delta").sum(),
        )
        .filter(pl.col("savings") != 0)
        .sort("savings", descending=True)
    )

    # ── Write summary report ──────────────────────────────────────────────────
    _write_summary(df, group_summary, eps_applied, eps_star,
                   auto_select_method, effective_threshold)

    # ── Apply threshold filter ────────────────────────────────────────────────
    if AUTO_THRESHOLD and b_k_threshold is not None:
        grp_agg = df.group_by("group_index").agg(
            pl.col("savings").sum(),
            pl.col("fuel_delta").sum(),
        ).with_columns(
            (pl.col("savings") + (FUEL_BIAS - 1.0) * pl.col("fuel_delta") * FUEL_PRICE)
            .alias("b_k")
        )
        approved_groups = grp_agg.filter(pl.col("b_k") >= b_k_threshold)["group_index"].to_list()
    else:
        approved_groups = (
            group_summary.filter(pl.col("savings") >= MIN_SAVINGS_THRESHOLD)["group_index"].to_list()
        )

    df_approved = df.filter(
        pl.col("group_index").is_in(approved_groups) & (pl.col("changed") == True)
    )

    # ── Sort by group savings descending ─────────────────────────────────────
    grp_savings = (
        df_approved.group_by("group_index")
        .agg(pl.col("savings").sum().alias("_group_savings"))
    )
    df_approved = (
        df_approved.join(grp_savings, on="group_index")
        .sort(["_group_savings", "group_index"], descending=[True, False])
        .drop("_group_savings")
    )

    # ── Select output columns ─────────────────────────────────────────────────
    select_cols = [
        "group_index", "flt_date", "base",
        "aircraftreg", "Route", "flt_numbers", "total_sectors",
        "opti_aircraftreg", "perf_type", "mtow", "seat_config", "savings",
    ]
    df_out = df_approved.select([c for c in select_cols if c in df_approved.columns])

    # Format savings as +$NNN / -$NNN
    df_out = df_out.with_columns(
        pl.when(pl.col("savings") >= 0)
        .then(pl.lit("+$") + pl.col("savings").round(0).cast(pl.Int64).abs().cast(pl.Utf8))
        .otherwise(pl.lit("-$") + pl.col("savings").round(0).cast(pl.Int64).abs().cast(pl.Utf8))
        .alias("savings")
    )

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = f"{OUTPUT_DIRECTORY}/{OUTPUT_CSV}"
    df_out.write_csv(out_path)

    print(f"\nApproved swaps   : {len(df_out)}  across {df_out['group_index'].n_unique()} groups")
    print(f"Swap list written: {out_path}")
    print(f"Columns          : {df_out.columns}")
    print("\nStage 6 Swap Approval complete.")


if __name__ == "__main__":
    main()
