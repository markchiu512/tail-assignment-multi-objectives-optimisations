"""
rotation_mip_20260802.py
========================
Research/shadow mode — no production files touched.
No assignment_optimisation.csv is read anywhere in this script.

Source: a319_greg_rotations_preopt_20260802.csv, cost_index.csv
Output: rotation_mip_results_20260802.csv, rotation_mip_report_20260802.txt

Parts:
  0  — Counts (tail mtow capability vs rotation min_mtow requirement)
  2  — Fixed anchors + per-tail base timeline (reported before the solve)
  1  — Eligibility (MTOW downward-compat + base availability from fixed-event timeline)
  3  — Conflict pairs + cost matrix
  4a — E1: status-quo feasibility
  4b — LP relaxation
  4c — E3: exclusion attribution
  4d — Main MIP solve + saving
  5  — ε-constraint sweep (fuel focus, 0.0%–1.0% in 0.05% steps, 1.1%–2.0% in 0.1% steps)
"""

import sys
import os
import io
import time
import shutil
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import polars as pl

try:
    from pyomo.environ import (
        ConcreteModel, Var, Binary, Reals, Objective,
        ConstraintList, minimize, value, SolverFactory,
    )
except ImportError as e:
    print(f"ERROR: Pyomo not available: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
RES_DIR = Path(os.environ.get("ROTATION_RESEARCH_DIR", Path(__file__).parent / "data"))
PREOPT_CSV = RES_DIR / "rotation_preopt.csv"
COST_CSV = RES_DIR / "cost_index.csv"
OUT_CSV = RES_DIR / "rotation_mip_results.csv"
OUT_REPORT = RES_DIR / "rotation_mip_report.txt"

FUEL_PRICE = float(os.environ.get("FUEL_PRICE", "1.20"))
GAP_MIN = int(os.environ.get("TURNAROUND_GAP_MINUTES", "50"))

# ε-constraint sweep: 0.0% to 2.0% in 0.2% steps (11 points, same density as production)
EPS_VALUES = (
    [round(i * 0.0005, 4) for i in range(21)]          # 0.00% to 1.00% in 0.05% steps
    + [round(0.01 + i * 0.001, 4) for i in range(1, 11)]  # 1.10% to 2.00% in 0.10% steps
)  # 31 points total

# ---------------------------------------------------------------------------
# Report collector
# ---------------------------------------------------------------------------
_lines = []

def p(*args):
    s = " ".join(str(a) for a in args)
    _lines.append(s)
    print(s)

def section(title):
    p()
    p("=" * 72)
    p(f"  {title}")
    p("=" * 72)

# ---------------------------------------------------------------------------
# Load + parse
# ---------------------------------------------------------------------------
section("LOADING DATA")

df_raw = pl.read_csv(PREOPT_CSV, try_parse_dates=False)

for col in ["rotation_start", "rotation_end", "away_arr_datetime", "away_dep_datetime"]:
    if col in df_raw.columns:
        df_raw = df_raw.with_columns(
            pl.col(col)
              .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
              .alias(col)
        )

p(f"Preopt rows    : {len(df_raw)}")

df_rot    = df_raw.filter(pl.col("type_flag") == "rotation")
df_sl     = df_raw.filter(pl.col("type_flag") == "single_leg")
df_swap   = df_rot.filter(pl.col("swappable"))
df_noswap = df_rot.filter(~pl.col("swappable"))
df_fixed  = pl.concat([df_sl, df_noswap])

p(f"  Rotations      : {df_rot.height}  ({df_swap.height} swappable, {df_noswap.height} non-swappable)")
p(f"  Single legs    : {df_sl.height}")
p(f"  Fixed anchors  : {df_fixed.height}  (non-swappable rotations + single legs)")

# ---------------------------------------------------------------------------
# PART 0 — COUNTS
# ---------------------------------------------------------------------------
section("PART 0 — COUNTS")

# (a) Tail MTOW capability — one row per tail, use first occurrence
tail_first = (
    df_raw.sort("rotation_start")
          .group_by("aircraftreg")
          .agg([
              pl.first("mtow").alias("tail_mtow"),
              pl.first("avg_perf_corr"),
              pl.first("fh_rate"),
              pl.first("cycle_rate"),
              pl.first("origin_iata").alias("first_base"),
          ])
)
tail_props = {row["aircraftreg"]: row for row in tail_first.iter_rows(named=True)}
all_tails  = sorted(tail_props.keys())
T          = len(all_tails)

tail_mtow_dist = (
    tail_first.group_by("tail_mtow")
              .agg(pl.len().alias("n_tails"))
              .sort("tail_mtow")
)
p(f"\n(a) Tail MTOW capability ({T} unique tails):")
for row in tail_mtow_dist.iter_rows(named=True):
    p(f"    tail_mtow={row['tail_mtow']:,} : {row['n_tails']} tails")

# (b) Rotation min_mtow requirement — swappable rotations only
p(f"\n(b) Rotation min_mtow requirement ({df_swap.height} swappable rotations):")
rot_mm_dist = (
    df_swap.group_by("min_mtow")
           .agg(pl.len().alias("n"))
           .sort("min_mtow")
)
for row in rot_mm_dist.iter_rows(named=True):
    p(f"    min_mtow={row['min_mtow']} : {row['n']} rotations")

# (c) Fixed anchors — min_mtow presence and exclusion reason
p(f"\n(c) Fixed anchors (swappable=False — excluded from assignment pool entirely):")
p(f"    Single legs : {df_sl.height}")
p(f"    Non-swappable rotations : {df_noswap.height}")
for row in df_noswap.iter_rows(named=True):
    p(f"      {row['aircraftreg']}  {row['origin_iata']}-{row['away_iata']}  "
      f"tail_mtow={row['mtow']:,}  min_mtow={row['min_mtow']}  "
      f"(tail_mtow < min_mtow → ineligible on own tail)")
p(f"\n    NOTE: min_mtow IS populated for fixed anchors (not blank). They are excluded")
p(f"    by swappable=False. Both non-swappable rotations fail because tail_mtow < min_mtow.")
sl_mm = df_sl.select("min_mtow").to_series().to_list()
sl_null = sum(1 for v in sl_mm if v is None)
p(f"    Single legs: {sl_null} have min_mtow=None (anomaly), {len(sl_mm)-sl_null} have a value.")

# ---------------------------------------------------------------------------
# Swappable rotation list (decision variables, indexed 0..N-1)
# ---------------------------------------------------------------------------
rot_swap_list = (
    df_swap.sort(["rotation_start", "aircraftreg"])
           .to_dicts()
)
N = len(rot_swap_list)
p(f"\nN (decision variables) : {N}")
p(f"T (candidate tails)    : {T}")

# ---------------------------------------------------------------------------
# PART 2 — FIXED ANCHORS + PER-TAIL BASE TIMELINE
# ---------------------------------------------------------------------------
section("PART 2 — FIXED ANCHORS AND PER-TAIL BASE TIMELINE")

GAP = timedelta(minutes=GAP_MIN)
DAY_START = datetime(2026, 8, 2,  0, 0, 0)
DAY_END   = datetime(2026, 8, 3,  6, 0, 0)   # extend past midnight for overnight legs

p(f"\nOwn-tail gaps < {GAP_MIN}min in the source schedule (original schedule must stay feasible):")
by_tail_swap = defaultdict(list)
for r_idx, rot in enumerate(rot_swap_list):
    by_tail_swap[rot["aircraftreg"]].append((r_idx, rot))

short_gap_pairs = []
for tail, pairs in sorted(by_tail_swap.items()):
    pairs.sort(key=lambda x: x[1]["rotation_start"])
    for k in range(len(pairs) - 1):
        _, r1 = pairs[k]
        _, r2 = pairs[k + 1]
        gap_m = (r2["rotation_start"] - r1["rotation_end"]).total_seconds() / 60
        if gap_m < GAP_MIN:
            short_gap_pairs.append({
                "tail": tail,
                "flt1": r1["outbound_flt_no"],
                "flt2": r2["outbound_flt_no"],
                "gap_min": gap_m,
            })
            p(f"  {tail}  flt {r1['outbound_flt_no']} → flt {r2['outbound_flt_no']}  gap={gap_m:.0f}min")

p(f"\n  Total own-tail short-gap pairs: {len(short_gap_pairs)}")
p(f"  These are exempt from the no-overlap constraint when t==own1==own2.")

# Build timeline from FIXED events only (single legs + non-swappable rotations)
fixed_by_tail = defaultdict(list)
for row in df_fixed.iter_rows(named=True):
    fixed_by_tail[row["aircraftreg"]].append(row)
for t in fixed_by_tail:
    fixed_by_tail[t].sort(key=lambda x: x["rotation_start"])


def build_fixed_timeline(tail: str) -> list[dict]:
    """
    Availability windows derived from FIXED events only.
    Each segment: {base, avail_from, avail_to}.
    avail_to is the start of the next fixed event (exclusive).
    GAP is applied after each fixed event end before the tail is available again.
    Tails with no fixed events are available at their first_base all day.
    """
    fixed_evs = fixed_by_tail.get(tail, [])
    home_base  = tail_props[tail]["first_base"]

    segments   = []
    cur_base   = home_base
    avail_from = DAY_START

    for ev in fixed_evs:
        ev_start = ev["rotation_start"]
        ev_end   = ev["rotation_end"]
        if ev_start is None or ev_end is None:
            continue
        # Window closes GAP before the fixed event starts (tail must arrive/prep in time)
        win_end = ev_start - GAP
        if avail_from < win_end:
            segments.append({
                "base":       cur_base,
                "avail_from": avail_from,
                "avail_to":   win_end,
            })
        # Single leg moves the tail; rotation returns to origin
        cur_base   = ev["away_iata"] if ev["type_flag"] == "single_leg" else ev["origin_iata"]
        avail_from = ev_end   # immediately available after fixed event ends

    if avail_from < DAY_END:
        segments.append({
            "base":       cur_base,
            "avail_from": avail_from,
            "avail_to":   DAY_END,
        })
    return segments


tail_timelines = {t: build_fixed_timeline(t) for t in all_tails}

p(f"\nPer-tail base timeline (built from fixed events only, GAP={GAP_MIN}min):")
p(f"  {'Tail':<12}  {'Fixed':>5}  Availability windows")
for tail in sorted(all_tails):
    n_f  = len(fixed_by_tail.get(tail, []))
    segs = tail_timelines[tail]
    seg_strs = [
        f"{s['base']}@{s['avail_from'].strftime('%H:%M')}-{s['avail_to'].strftime('%H:%M')}"
        for s in segs
    ]
    p(f"  {tail:<12}  {n_f:>5}  {', '.join(seg_strs)}")

# ---------------------------------------------------------------------------
# PART 1 / C — ELIGIBILITY
# ---------------------------------------------------------------------------
section("PART 1 — ELIGIBILITY MASK")

p(f"Rule 1 (MTOW):  tail_mtow >= rotation_min_mtow  (downward-compatible; min_mtow from preopt column)")
p(f"Rule 2 (Base):  fixed-event timeline has a segment where base==origin AND")
p(f"                avail_from<=rot_start-{GAP_MIN}min AND avail_to>=rot_end")
p(f"Own tail:       always eligible (original schedule is feasible by definition)")


def tail_available(tail: str, rot: dict) -> bool:
    rot_start = rot["rotation_start"]
    rot_end   = rot["rotation_end"]
    origin    = rot["origin_iata"]
    if rot_start is None or rot_end is None:
        return False
    # Tail must be free from (rot_start - GAP) through rot_end
    required_from = rot_start - GAP
    for seg in tail_timelines[tail]:
        if (seg["base"]       == origin
                and seg["avail_from"] <= required_from
                and seg["avail_to"]   >= rot_end):
            return True
    return False


eligible    = {}   # {(r_idx, tail): bool}
excl_reason = {}   # {(r_idx, tail): str}

for r_idx, rot in enumerate(rot_swap_list):
    own_tail  = rot["aircraftreg"]
    rot_min_m = rot.get("min_mtow")   # guaranteed non-None for swappable rotations

    for tail in all_tails:
        if tail == own_tail:
            eligible[(r_idx, tail)]    = True
            excl_reason[(r_idx, tail)] = "own_tail"
            continue

        # Rule 1 — MTOW downward-compatibility
        if rot_min_m is None or tail_props[tail]["tail_mtow"] < rot_min_m:
            eligible[(r_idx, tail)]    = False
            excl_reason[(r_idx, tail)] = "mtow"
            continue

        # Rule 2 — base availability in fixed-event timeline
        if not tail_available(tail, rot):
            eligible[(r_idx, tail)]    = False
            excl_reason[(r_idx, tail)] = "base_or_sl"
            continue

        eligible[(r_idx, tail)]    = True
        excl_reason[(r_idx, tail)] = "eligible"

elig_list = [(r, t) for (r, t), v in eligible.items() if v]
p(f"\nEligible (r,t) pairs : {len(elig_list)} / {N*T}  ({100*len(elig_list)/(N*T):.1f}%)")

cross_elig = [(r, t) for (r, t) in elig_list if rot_swap_list[r]["aircraftreg"] != t]
p(f"  Own-tail pairs     : {N}  (one per rotation)")
p(f"  Cross-tail pairs   : {len(cross_elig)}")

# Rotations with no alternative tail
no_alt = [r for r in range(N) if sum(1 for t in all_tails
          if t != rot_swap_list[r]["aircraftreg"] and eligible.get((r, t), False)) == 0]
if no_alt:
    p(f"\n  Rotations with zero alternative tails ({len(no_alt)}) — can only stay on own tail:")
    for r in no_alt:
        rot = rot_swap_list[r]
        p(f"    r={r:3d}  {rot['aircraftreg']}  {rot['origin_iata']}-{rot['away_iata']}  "
          f"flt {rot['outbound_flt_no']}")

# ---------------------------------------------------------------------------
# PART 3 — CONFLICT PAIRS + COST MATRIX
# ---------------------------------------------------------------------------
section("PART 3 — CONFLICT PAIRS AND COST MATRIX")


def rotations_conflict(r1: dict, r2: dict) -> bool:
    """True if [r1_start-GAP, r1_end) and [r2_start-GAP, r2_end) overlap."""
    s1, e1 = r1["rotation_start"], r1["rotation_end"]
    s2, e2 = r2["rotation_start"], r2["rotation_end"]
    return not (e1 <= s2 - GAP or e2 <= s1 - GAP)


conflict_pairs = []
for i in range(N):
    for j in range(i + 1, N):
        r1 = rot_swap_list[i]
        r2 = rot_swap_list[j]
        if not rotations_conflict(r1, r2):
            continue
        # Add every conflicting pair unconditionally.
        # The no-overlap constraint inside solve_mip exempts t==own1==own2 (short-gap
        # same-tail pairs) internally — that exemption is sufficient.
        # The previous shared_alts guard incorrectly skipped direct swap pairs (A↔B)
        # where each rotation's only cross-eligible tail was the other's own tail,
        # allowing a solver to double-assign one tail to overlapping rotations.
        conflict_pairs.append((i, j))

p(f"Conflict pairs (time-overlap): {len(conflict_pairs)}")

# Cost matrix — IMPORTANT: all rates from ASSIGNED tail t, never from rot's original tail
# F: fuel-cost matrix (perf_corr * FUEL_PRICE * baseline_kg) — used as objective in ε-solve
C = {}
F = {}
for r_idx, rot in enumerate(rot_swap_list):
    fuel_kg    = rot["total_pred_baseline_fuel_kg"]
    airborne_h = rot["total_pred_airborne_hours"]
    n_cyc      = rot["n_cycles"]
    for tail in all_tails:
        if not eligible.get((r_idx, tail), False):
            continue
        tp = tail_props[tail]
        C[(r_idx, tail)] = (
            tp["avg_perf_corr"] * FUEL_PRICE * fuel_kg
            + tp["fh_rate"]    * airborne_h
            + tp["cycle_rate"] * n_cyc
        )
        F[(r_idx, tail)] = tp["avg_perf_corr"] * FUEL_PRICE * fuel_kg

baseline_total      = sum(rot["total_cost"]                   for rot in rot_swap_list)
baseline_fuel_cost  = sum(
    tail_props[rot["aircraftreg"]]["avg_perf_corr"] * FUEL_PRICE * rot["total_pred_baseline_fuel_kg"]
    for rot in rot_swap_list
)
baseline_fuel_kg    = sum(
    tail_props[rot["aircraftreg"]]["avg_perf_corr"] * rot["total_pred_baseline_fuel_kg"]
    for rot in rot_swap_list
)
p(f"Baseline cost (sum of total_cost, own-tail): ${baseline_total:,.2f}")
p(f"Baseline fuel cost (own-tail):               ${baseline_fuel_cost:,.2f}")
p(f"Baseline actual fuel kg (own-tail):           {baseline_fuel_kg:,.1f} kg")


# ---------------------------------------------------------------------------
# MIP/LP solver helper
# ---------------------------------------------------------------------------
def solve_mip(
    elig_l,
    conf_pairs,
    cost_mat,
    force_assignment=None,
    use_lp=False,
):
    m      = ConcreteModel()
    domain = Reals if use_lp else Binary
    m.x    = Var(elig_l, domain=domain, bounds=(0, 1) if use_lp else None)

    m.obj = Objective(
        expr  = sum(cost_mat[k] * m.x[k] for k in elig_l if k in cost_mat),
        sense = minimize,
    )

    # Coverage: each rotation assigned exactly once
    m.coverage = ConstraintList()
    for r_idx in range(N):
        et = [t for (ri, t) in elig_l if ri == r_idx]
        if not et:
            m.coverage.add(0 == 1)   # forces INFEASIBLE
        else:
            m.coverage.add(sum(m.x[r_idx, t] for t in et) == 1)

    # No-overlap: overlapping rotation pairs on the same candidate tail
    elig_set   = set(elig_l)
    m.no_overlap = ConstraintList()
    for (r1, r2) in conf_pairs:
        own1 = rot_swap_list[r1]["aircraftreg"]
        own2 = rot_swap_list[r2]["aircraftreg"]
        for t in all_tails:
            # Exempt: both rotations originally on the same tail — gap may be < MIN_GAP
            # in the source schedule and must remain feasible.
            if t == own1 and t == own2:
                continue
            if (r1, t) in elig_set and (r2, t) in elig_set:
                m.no_overlap.add(m.x[r1, t] + m.x[r2, t] <= 1)

    # Force (E1 status quo)
    m.forced = ConstraintList()
    if force_assignment:
        for r_idx, tail in force_assignment.items():
            if (r_idx, tail) in elig_set:
                m.forced.add(m.x[r_idx, tail] == 1)

    solver = SolverFactory("appsi_highs")
    try:
        solver.options["time_limit"] = 300
    except Exception:
        pass

    t0 = time.time()
    try:
        result  = solver.solve(m)
        elapsed = time.time() - t0
        try:
            status = str(result.solver.termination_condition)
        except Exception:
            status = "unknown"

        obj_val    = None
        assignment = {}
        frac_count = 0
        try:
            obj_val = value(m.obj)
            for (r, t) in elig_l:
                v = value(m.x[r, t])
                if v is None:
                    continue
                if use_lp:
                    if 1e-4 < v < 1 - 1e-4:
                        frac_count += 1
                else:
                    if v > 0.5:
                        assignment[r] = t
        except Exception:
            pass

        return obj_val, assignment, status, elapsed, frac_count

    except Exception as e:
        return None, {}, f"ERROR: {e}", time.time() - t0, 0


def solve_mip_fuel(elig_l, conf_pairs, fuel_mat, cost_mat, cost_budget):
    """Minimise fuel cost subject to total cost <= cost_budget.
    Same structure as solve_mip; adds one extra budget constraint.
    Returns (fuel_obj, cost_obj, assignment, status, elapsed).
    """
    m        = ConcreteModel()
    m.x      = Var(elig_l, domain=Binary)
    elig_set = set(elig_l)

    m.obj = Objective(
        expr  = sum(fuel_mat[k] * m.x[k] for k in elig_l if k in fuel_mat),
        sense = minimize,
    )

    m.coverage = ConstraintList()
    for r_idx in range(N):
        et = [t for (ri, t) in elig_l if ri == r_idx]
        if not et:
            m.coverage.add(0 == 1)
        else:
            m.coverage.add(sum(m.x[r_idx, t] for t in et) == 1)

    m.no_overlap = ConstraintList()
    for (r1, r2) in conf_pairs:
        own1 = rot_swap_list[r1]["aircraftreg"]
        own2 = rot_swap_list[r2]["aircraftreg"]
        for t in all_tails:
            if t == own1 and t == own2:
                continue
            if (r1, t) in elig_set and (r2, t) in elig_set:
                m.no_overlap.add(m.x[r1, t] + m.x[r2, t] <= 1)

    # ε-constraint: total composite cost must not exceed budget
    m.budget = ConstraintList()
    m.budget.add(
        sum(cost_mat[k] * m.x[k] for k in elig_l if k in cost_mat) <= cost_budget
    )

    solver = SolverFactory("appsi_highs")
    try:
        solver.options["time_limit"] = 300
    except Exception:
        pass

    t0 = time.time()
    try:
        result  = solver.solve(m)
        elapsed = time.time() - t0
        try:
            status = str(result.solver.termination_condition)
        except Exception:
            status = "unknown"

        fuel_obj = None
        cost_obj = None
        assignment = {}
        try:
            fuel_obj = value(m.obj)
            cost_obj = sum(
                cost_mat.get((r, t), 0.0) * value(m.x[r, t])
                for (r, t) in elig_l
                if value(m.x[r, t]) is not None and value(m.x[r, t]) > 0.5
            )
            for (r, t) in elig_l:
                v = value(m.x[r, t])
                if v is not None and v > 0.5:
                    assignment[r] = t
        except Exception:
            pass

        return fuel_obj, cost_obj, assignment, status, elapsed

    except Exception as e:
        return None, None, {}, f"ERROR: {e}", time.time() - t0


# =========================================================================
# POST-SOLVE FEASIBILITY ASSERTION
# =========================================================================
# Build fixed-event lookup once (shared across all assertion calls)
# fixed_ops_by_tail: tail -> list of {start, end, orig_tail, flt, kind}
_fixed_ops_by_tail: dict[str, list[dict]] = defaultdict(list)
for row in df_fixed.to_dicts():
    # rotation_start/end are already datetime objects (cast at load time)
    _fixed_ops_by_tail[row["aircraftreg"]].append({
        "start":     row["rotation_start"],
        "end":       row["rotation_end"],
        "orig_tail": row["aircraftreg"],
        "flt":       row["outbound_flt_no"],
        "kind":      row["type_flag"],
    })

def assert_solution_feasible(assignment: dict, label: str) -> None:
    """
    For every tail, collect ALL operations assigned to it:
      - swappable rotations  (from assignment dict)
      - non-swappable rotations + single legs  (fixed, always on orig tail)
    Sort by start time. Check every consecutive pair has gap >= PREP_TIME,
    EXCEPT pairs where BOTH ops were originally on that same tail
    (grandfathered short gaps from the source schedule).

    Raises AssertionError loudly on any violation — never warns and continues.
    Reports a clean summary on pass.
    """
    PREP_TIME = timedelta(minutes=GAP_MIN)

    # Collect swappable ops keyed by assigned tail
    swap_ops_by_tail: dict[str, list[dict]] = defaultdict(list)
    for r_idx, tail in assignment.items():
        rot = rot_swap_list[r_idx]
        swap_ops_by_tail[tail].append({
            "start":     rot["rotation_start"],
            "end":       rot["rotation_end"],
            "orig_tail": rot["aircraftreg"],
            "flt":       rot["outbound_flt_no"],
            "kind":      "rotation",
        })

    violations = []
    tails_checked = set(swap_ops_by_tail.keys()) | set(_fixed_ops_by_tail.keys())

    for tail in sorted(tails_checked):
        ops = swap_ops_by_tail.get(tail, []) + _fixed_ops_by_tail.get(tail, [])
        ops.sort(key=lambda x: x["start"])

        for k in range(len(ops) - 1):
            a = ops[k]
            b = ops[k + 1]
            gap = b["start"] - a["end"]

            if gap >= PREP_TIME:
                continue  # OK

            # Check exemption: both ops were originally on this tail in the
            # source schedule — short gaps from the original plan are grandfathered.
            if a["orig_tail"] == tail and b["orig_tail"] == tail:
                continue  # grandfathered

            violations.append({
                "tail":      tail,
                "flt_a":     a["flt"],
                "orig_a":    a["orig_tail"],
                "end_a":     a["end"],
                "flt_b":     b["flt"],
                "orig_b":    b["orig_tail"],
                "start_b":   b["start"],
                "gap_min":   int(gap.total_seconds() / 60),
            })

    p(f"\n  [ASSERT] {label}")
    if violations:
        p(f"  *** FEASIBILITY VIOLATION — {len(violations)} illegal gap(s) ***")
        for v in violations:
            p(f"    tail {v['tail']}:  flt {v['flt_a']} (orig={v['orig_a']}) ends {v['end_a'].strftime('%H:%M')}"
              f"  →  flt {v['flt_b']} (orig={v['orig_b']}) starts {v['start_b'].strftime('%H:%M')}"
              f"  gap={v['gap_min']}min (need >={GAP_MIN})")
        raise AssertionError(
            f"[{label}] Solution is physically infeasible: "
            f"{len(violations)} gap violation(s). See output above."
        )
    else:
        p(f"  All gaps >= {GAP_MIN}min (or grandfathered). Solution is feasible. OK")


# =========================================================================
# PART 4a — E1: STATUS QUO FEASIBILITY
# =========================================================================
section("PART 4a — E1: STATUS QUO FEASIBILITY")

# Step 1: verify every rotation's own tail is in the eligibility mask
sq_blocked = []
for r_idx, rot in enumerate(rot_swap_list):
    own = rot["aircraftreg"]
    if not eligible.get((r_idx, own), False):
        sq_blocked.append((r_idx, rot, excl_reason.get((r_idx, own), "?")))

if sq_blocked:
    p(f"FAIL: {len(sq_blocked)} rotations whose own tail is NOT eligible — model is over-constrained:")
    for (r_idx, rot, reason) in sq_blocked:
        p(f"  r={r_idx:3d}  {rot['aircraftreg']}  flt={rot['outbound_flt_no']}  "
          f"{rot['origin_iata']}-{rot['away_iata']}  reason={reason}")
    p("Stopping.")
    sys.exit(1)
else:
    p("All own-tail pairs eligible in mask — eligibility does not block status quo.")

# Step 2: check conflict pairs for same-own-tail violations
sq_conflict_violations = []
for (i, j) in conflict_pairs:
    ti = rot_swap_list[i]["aircraftreg"]
    tj = rot_swap_list[j]["aircraftreg"]
    if ti == tj:
        ri = rot_swap_list[i]
        rj = rot_swap_list[j]
        g  = (rj["rotation_start"] - ri["rotation_end"]).total_seconds() / 60
        sq_conflict_violations.append((ti, ri["outbound_flt_no"], rj["outbound_flt_no"], g))

if sq_conflict_violations:
    p(f"\n{len(sq_conflict_violations)} conflict pairs on the same own tail (gap < {GAP_MIN}min):")
    for (t, f1, f2, g) in sq_conflict_violations:
        p(f"  {t}  flt {f1} → {f2}  gap={g:.0f}min")
    p("These are exempt in solve_mip (t==own1==own2 → skip constraint).")
else:
    p("\nNo conflict pairs share the same own tail.")

# Step 3: solve MIP with status quo forced
sq_force = {r: rot_swap_list[r]["aircraftreg"] for r in range(N)}
p("\nSolving MIP with status quo forced (x[r, own_tail]=1 for all r)...")
sq_obj, sq_assign, sq_status, sq_time, _ = solve_mip(
    elig_list, conflict_pairs, C, force_assignment=sq_force
)

p(f"  Status             : {sq_status}  ({sq_time:.2f}s)")
if sq_obj is not None:
    diff = sq_obj - baseline_total
    p(f"  Objective          : ${sq_obj:,.2f}")
    p(f"  Baseline (col sum) : ${baseline_total:,.2f}")
    p(f"  Difference         : ${diff:+,.2f}  {'OK (< $1)' if abs(diff) < 1.0 else 'NOTE: mismatch'}")
    if abs(diff) >= 1.0:
        p(f"  The cost formula C[r,t] uses tail_props[t] rates. If total_cost in the")
        p(f"  preopt CSV was computed with the same formula and own-tail rates, these")
        p(f"  should agree. A mismatch indicates a data join issue.")
    assert_solution_feasible(sq_assign, "E1 status-quo")
else:
    p("  INFEASIBLE or failed — model is over-constrained.")
    p("  See conflict-pair violation list above for the likely cause.")

# =========================================================================
# PART 4b — LP RELAXATION
# =========================================================================
section("PART 4b — LP RELAXATION (integrality check)")

p("Solving LP relaxation (Binary → Reals, same constraints, no force)...")
lp_obj, _, lp_status, lp_time, lp_frac = solve_mip(
    elig_list, conflict_pairs, C, use_lp=True
)

p(f"  LP status          : {lp_status}  ({lp_time:.2f}s)")
if lp_obj is not None:
    p(f"  LP objective       : ${lp_obj:,.2f}")
    p(f"  Fractional vars    : {lp_frac}  (0 < x < 1 with eps=1e-4)")
    if lp_frac == 0:
        p("  LP is integral — totally unimodular structure, B&B trivial.")
    else:
        p(f"  {lp_frac} fractional variable(s) — B&B needed to recover integer solution.")
    if sq_obj is not None:
        p(f"  LP saving vs status quo: ${sq_obj - lp_obj:,.2f}")
else:
    p("  LP failed.")

# =========================================================================
# PART 4c — E3: EXCLUSION ATTRIBUTION
# =========================================================================
section("PART 4c — E3: EXCLUSION ATTRIBUTION")

excl_counts = defaultdict(int)
for reason in excl_reason.values():
    excl_counts[reason] += 1

p(f"\nAll (r,t) pairs: {N*T}")
for reason, cnt in sorted(excl_counts.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / (N * T)
    p(f"  {reason:<36}: {cnt:6d}  ({pct:.1f}%)")

# Break down base_or_sl: base_mismatch vs window_too_tight
base_miss  = 0
win_tight  = 0
for r_idx, rot in enumerate(rot_swap_list):
    origin = rot["origin_iata"]
    for tail in all_tails:
        if excl_reason.get((r_idx, tail)) != "base_or_sl":
            continue
        has_base = any(s["base"] == origin for s in tail_timelines.get(tail, []))
        if not has_base:
            base_miss += 1
        else:
            win_tight += 1

total_base_sl = excl_counts["base_or_sl"]
p(f"\n  base_or_sl breakdown (total={total_base_sl}):")
p(f"    base_mismatch   (tail never visits rotation origin)  : {base_miss}")
p(f"    window_too_tight (tail at base but no window covers)  : {win_tight}")

# How many conflict-pair constraints were added to the MIP
elig_set = set(elig_list)
no_overlap_count = 0
for (r1, r2) in conflict_pairs:
    own1 = rot_swap_list[r1]["aircraftreg"]
    own2 = rot_swap_list[r2]["aircraftreg"]
    for t in all_tails:
        if t == own1 and t == own2:
            continue
        if (r1, t) in elig_set and (r2, t) in elig_set:
            no_overlap_count += 1

p(f"\n  No-overlap constraints added to MIP : {no_overlap_count}")
p(f"  (These are not exclusions — pairs remain eligible, just mutually exclusive on each shared tail)")

# =========================================================================
# PART 4d — MAIN MIP SOLVE
# =========================================================================
section(f"PART 4d — MAIN MIP SOLVE (gap={GAP_MIN}min)")

p(f"  N rotations (decision vars) : {N}")
p(f"  T tails (candidates)        : {T}")
p(f"  Eligible (r,t) pairs        : {len(elig_list)}")
p(f"  Conflict pairs              : {len(conflict_pairs)}")
p(f"\n  Solving...")

mip_obj, mip_assign, mip_status, mip_time, _ = solve_mip(
    elig_list, conflict_pairs, C
)

if mip_obj is None:
    p(f"  FAILED / INFEASIBLE: {mip_status}  ({mip_time:.1f}s)")
    mip_assign = {}
else:
    saving  = baseline_total - mip_obj
    changed = sum(1 for r, t in mip_assign.items()
                  if rot_swap_list[r]["aircraftreg"] != t)

    p(f"  Solve status        : {mip_status}  ({mip_time:.2f}s)")
    p(f"  Baseline total      : ${baseline_total:,.2f}")
    p(f"  Optimised total     : ${mip_obj:,.2f}")
    p(f"  Saving vs baseline  : ${saving:,.2f}")
    p(f"  Rotations changed   : {changed} / {N}")

    changes = []
    for r_idx, new_tail in mip_assign.items():
        rot = rot_swap_list[r_idx]
        if rot["aircraftreg"] != new_tail:
            cost_after = C.get((r_idx, new_tail), float("nan"))
            changes.append({
                "r_idx":       r_idx,
                "flt":         rot["outbound_flt_no"],
                "origin":      rot["origin_iata"],
                "dest":        rot["away_iata"],
                "from_tail":   rot["aircraftreg"],
                "to_tail":     new_tail,
                "cost_before": rot["total_cost"],
                "cost_after":  cost_after,
                "saving":      rot["total_cost"] - cost_after,
            })

    changes.sort(key=lambda x: -x["saving"])
    if changes:
        p(f"\n  Changed rotations (sorted by individual saving):")
        for ch in changes:
            p(f"    flt {str(ch['flt']):>5}  {ch['origin']}-{ch['dest']:<3}  "
              f"{ch['from_tail']} → {ch['to_tail']}  "
              f"save=${ch['saving']:>8,.0f}")
    else:
        p("\n  No rotations changed tail — status quo is optimal under these constraints.")

    assert_solution_feasible(mip_assign, "Main MIP (cost-optimal)")

# =========================================================================
# OUTPUT CSV
# =========================================================================
section("OUTPUT CSV")

# ── Swap-chain decomposition ─────────────────────────────────────────────
# A swap chain is a connected component in the tail-swap graph:
# edge exists between rotation r1 and r2 when the assigned tail of one
# is the original tail of the other (i.e. they're linked through a shared tail).
# Rotations where assigned_tail == original are unchanged and get no group_index.
#
# group_index format: <origin_iata of first changed rotation in chain>_<NNN>
# Numbering is sequential per origin_iata across all chains (same logic as before,
# but now one label per chain, shared by all rotations in that chain).

def _build_swap_chains(rot_list, assignment):
    """Return list[str|None] — group_index per rotation in rot_list order."""
    n = len(rot_list)
    orig  = [rot_list[i]["aircraftreg"]    for i in range(n)]
    new   = [assignment.get(i, orig[i])    for i in range(n)]

    # Only changed rotations participate
    changed_idx = [i for i in range(n) if orig[i] != new[i]]

    # Build adjacency: r1 and r2 are in the same chain if they share a tail
    # (new[r1]==orig[r2] or new[r2]==orig[r1])
    adj: dict[int, list[int]] = defaultdict(list)
    for a in changed_idx:
        for b in changed_idx:
            if a >= b:
                continue
            if new[a] == orig[b] or new[b] == orig[a]:
                adj[a].append(b)
                adj[b].append(a)

    # BFS connected components over changed rotations only
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in changed_idx:
        if start in visited:
            continue
        comp = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    # Assign labels: base = origin_iata of the earliest rotation in the chain
    base_ctr: dict[str, int] = defaultdict(int)
    labels: list[str | None] = [None] * n
    for comp in components:
        comp_sorted = sorted(comp, key=lambda i: rot_list[i]["rotation_start"])
        base = rot_list[comp_sorted[0]]["origin_iata"]
        base_ctr[base] += 1
        label = f"{base}_{base_ctr[base]:03d}"
        for i in comp:
            labels[i] = label

    return labels

rot_group_index = _build_swap_chains(rot_swap_list, mip_assign)

result_rows = []
for r_idx, rot in enumerate(rot_swap_list):
    # Use MIP assignment if available, else own tail
    new_tail   = mip_assign.get(r_idx, rot["aircraftreg"])
    orig_tail  = rot["aircraftreg"]
    cost_opt   = C.get((r_idx, new_tail), float("nan"))
    cost_orig  = rot["total_cost"]

    fuel_kg    = rot["total_pred_baseline_fuel_kg"]
    airborne_h = rot["total_pred_airborne_hours"]
    n_cyc      = rot["n_cycles"]

    tp_orig    = tail_props[orig_tail]
    tp_new     = tail_props[new_tail]

    # Component costs — original tail
    fuel_cost_orig  = tp_orig["avg_perf_corr"] * FUEL_PRICE * fuel_kg
    fh_cost_orig    = tp_orig["fh_rate"]        * airborne_h
    cycle_cost_orig = tp_orig["cycle_rate"]     * n_cyc

    # Component costs — assigned tail
    fuel_cost_opt   = tp_new["avg_perf_corr"]  * FUEL_PRICE * fuel_kg
    fh_cost_opt     = tp_new["fh_rate"]         * airborne_h
    cycle_cost_opt  = tp_new["cycle_rate"]      * n_cyc

    # Saving decomposition (positive = saves money / fuel)
    fuel_delta_kg    = (tp_orig["avg_perf_corr"] - tp_new["avg_perf_corr"]) * fuel_kg
    fuel_cost_saving = fuel_cost_orig  - fuel_cost_opt
    fh_rate_saving   = fh_cost_orig    - fh_cost_opt
    cycle_saving     = cycle_cost_orig - cycle_cost_opt

    result_rows.append({
        "group_index":          rot_group_index[r_idx],
        "aircraftreg_original": orig_tail,
        "outbound_flt_no":      rot["outbound_flt_no"],
        "inbound_flt_no":       rot["inbound_flt_no"],
        "origin_iata":          rot["origin_iata"],
        "away_iata":            rot["away_iata"],
        "rotation_start":       rot["rotation_start"].strftime("%Y-%m-%d %H:%M:%S"),
        "rotation_end":         rot["rotation_end"].strftime("%Y-%m-%d %H:%M:%S"),
        "min_mtow_req":         rot["min_mtow"],
        "tail_mtow_orig":       rot["mtow"],
        "assigned_tail":        new_tail,
        "tail_mtow_assigned":   tp_new["tail_mtow"],
        "cost_original":        round(cost_orig, 4),
        "cost_optimised":       round(cost_opt, 4),
        "saving":               round(cost_orig - cost_opt, 4),
        "fuel_delta_kg":        round(fuel_delta_kg, 4),
        "fuel_cost_saving":     round(fuel_cost_saving, 4),
        "fh_rate_saving":       round(fh_rate_saving, 4),
        "cycle_saving":         round(cycle_saving, 4),
        "changed":              orig_tail != new_tail,
        "n_eligible_tails":     sum(1 for t in all_tails if eligible.get((r_idx, t), False)),
    })

result_df = pl.DataFrame(result_rows)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
    tmp_path = tmp.name
_out_csv_actual = OUT_CSV
try:
    result_df.write_csv(tmp_path)
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    shutil.move(tmp_path, OUT_CSV)
except PermissionError:
    # Use a fallback when an output file is locked by another local process.
    _out_csv_actual = OUT_CSV.replace(".csv", "_v2.csv")
    try:
        shutil.move(tmp_path, _out_csv_actual)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    p(f"  WARNING: {OUT_CSV} is locked — written to fallback: {_out_csv_actual}")
except Exception:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    raise

p(f"Written: {_out_csv_actual}  ({len(result_rows)} rows)")

# =========================================================================
# PART 5 — ε-CONSTRAINT SWEEP (fuel focus, 0.0% to 2.0%)
# =========================================================================
section("PART 5 — ε-CONSTRAINT SWEEP")

if mip_obj is None:
    p("Main MIP failed — skipping ε sweep.")
else:
    c_star = mip_obj   # cost-optimal solution is the budget baseline

    sweep_rows = []   # one row per ε point
    p(f"\n  c* (cost-optimal) = ${c_star:,.2f}")
    p(f"  Sweeping ε = {[f'{e*100:.1f}%' for e in EPS_VALUES]}")
    p()
    p(f"  {'ε':>7}  {'Budget':>14}  {'Fuel obj':>14}  {'Cost':>14}  "
      f"{'Fuel save vs c*':>16}  {'Cost penalty':>13}  Status  Time")
    p(f"  {'-'*7}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*16}  {'-'*13}  ------  ----")

    # Baseline fuel at ε=0 is the cost-optimal fuel (from main MIP)
    cost_opt_fuel = sum(
        F.get((r, t), 0.0)
        for r, t in mip_assign.items()
    )

    for eps in EPS_VALUES:
        budget = c_star * (1.0 + eps)
        f_obj, c_obj, eps_assign, eps_status, eps_time = solve_mip_fuel(
            elig_list, conflict_pairs, F, C, budget
        )

        if f_obj is None:
            p(f"  {eps*100:6.2f}%  {'—':>14}  {'INFEASIBLE':>14}  {'—':>14}  {'—':>16}  {'—':>13}  {eps_status}  {eps_time:.1f}s")
            sweep_rows.append({
                "eps":              eps,
                "eps_pct":          round(eps * 100, 2),
                "budget":           round(budget, 2),
                "fuel_cost_obj":    None,
                "total_cost":       None,
                "fuel_kg":          None,
                "fuel_saving_vs_c0": None,
                "cost_penalty":     None,
                "n_changed":        None,
                "status":           eps_status,
            })
            continue

        # Fuel kg from the fuel-cost objective
        eps_fuel_kg = f_obj / FUEL_PRICE
        fuel_save   = cost_opt_fuel - f_obj      # positive = less fuel cost than c*
        cost_pen    = (c_obj - c_star) if c_obj is not None else float("nan")
        n_ch        = sum(1 for r, t in eps_assign.items() if rot_swap_list[r]["aircraftreg"] != t)

        assert_solution_feasible(eps_assign, f"ε={eps*100:.2f}%")

        p(f"  {eps*100:6.2f}%  ${budget:>13,.2f}  ${f_obj:>13,.2f}  "
          f"${c_obj:>13,.2f}  ${fuel_save:>+15,.2f}  ${cost_pen:>+12,.2f}  "
          f"{eps_status:<8}  {eps_time:.1f}s")

        sweep_rows.append({
            "eps":               eps,
            "eps_pct":           round(eps * 100, 2),
            "budget":            round(budget, 2),
            "fuel_cost_obj":     round(f_obj, 2),
            "total_cost":        round(c_obj, 2) if c_obj is not None else None,
            "fuel_kg":           round(eps_fuel_kg, 1),
            "fuel_saving_vs_c0": round(fuel_save, 2),
            "cost_penalty":      round(cost_pen, 2),
            "n_changed":         n_ch,
            "status":            eps_status,
        })

    # ── Write sweep CSV ────────────────────────────────────────────────────
    out_sweep_csv = _out_csv_actual.replace(".csv", "_eps_sweep.csv")
    sweep_df = pl.DataFrame(sweep_rows)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    try:
        sweep_df.write_csv(tmp_path)
        if os.path.exists(out_sweep_csv):
            os.remove(out_sweep_csv)
        shutil.move(tmp_path, out_sweep_csv)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    p(f"\n  Sweep CSV: {out_sweep_csv}")

    # ── Plotly elbow curve ─────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        valid = [r for r in sweep_rows if r["fuel_cost_obj"] is not None]
        xs          = [r["eps_pct"]           for r in valid]
        fuel_saves  = [r["fuel_saving_vs_c0"] for r in valid]
        cost_pens   = [r["cost_penalty"]       for r in valid]
        fuel_kgs    = [r["fuel_kg"]            for r in valid]
        n_changed   = [r["n_changed"]          for r in valid]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Primary: fuel saving vs ε=0 solution
        fig.add_trace(go.Scatter(
            x=xs, y=fuel_saves,
            mode="lines+markers",
            name="Fuel cost saving vs ε=0 ($)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=8),
            hovertemplate=(
                "ε = %{x:.2f}%<br>"
                "Fuel saving: $%{y:,.0f}<br>"
                "Fuel kg: %{customdata[0]:,.0f}<br>"
                "Rotations changed: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(fuel_kgs, n_changed)),
        ), secondary_y=False)

        # Secondary: cost penalty vs cost-optimal
        fig.add_trace(go.Scatter(
            x=xs, y=cost_pens,
            mode="lines+markers",
            name="Cost penalty vs c* ($)",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate=(
                "ε = %{x:.2f}%<br>"
                "Cost penalty: $%{y:,.0f}<extra></extra>"
            ),
        ), secondary_y=True)

        # Zero reference line
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey",
                      secondary_y=False)

        fig.update_layout(
            title=dict(
                text="Rotation MIP ε-sweep — A320neo G-reg 02 Aug 2026<br>"
                     "<sup>Objective: minimise fuel cost | Budget: cost ≤ c*(1+ε)</sup>",
                font=dict(size=15),
            ),
            xaxis=dict(
                title="ε (%)",
                tickvals=xs,
                ticktext=[f"{x:.2f}%" for x in xs],
                gridcolor="#e5e5e5",
            ),
            yaxis=dict(
                title="Fuel cost saving vs ε=0 ($)",
                gridcolor="#e5e5e5",
                zeroline=True, zerolinecolor="#aaa",
            ),
            yaxis2=dict(
                title="Cost penalty vs c* ($)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=900, height=520,
        )

        out_html = _out_csv_actual.replace(".csv", "_eps_sweep.html")
        fig.write_html(out_html, include_plotlyjs="cdn")
        p(f"  Plotly chart: {out_html}")

    except ImportError:
        p("  plotly not installed — skipping chart (pip install plotly)")


# =========================================================================
# WRITE REPORT
# =========================================================================
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("ROTATION MIP — 2026-08-02\n")
    f.write("=" * 72 + "\n\n")
    for line in _lines:
        f.write(line + "\n")

p(f"Report written: {OUT_REPORT}")
