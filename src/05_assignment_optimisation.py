"""
assignment_optimisation.py  —  v2.0.0  Stage 5: Assignment Optimisation

Reads the eligible tail set from Stage 4 and solves the minimum-cost
aircraft-to-route assignment via Pyomo MIP (HiGHS) or Gurobi Cluster Manager.

Responsibilities
----------------
- Group eligible tails by (flt_date, base, aircrafttype, mtow, seat_config)
- Build per-group cost matrix: fuel + FH + cycle costs
- Solve optimal assignment with forbidden-pair constraints:
    Cape Verde: only IRIS tails
    Cyprus:     CYPRUS_PROHIBITED tails excluded
    KEF:        only AUTOLAND-capable tails (or NEO types)
- Optional ε-constraint sweep: minimise fuel s.t. total cost ≤ c*(1+ε)
- Auto-select ε* via Kneedle / chord / curvature method if AUTO_SELECT_EPS = True
- Write skipped-group JSON and (optional) ε-sweep CSVs

Input  : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eligibility_filter.csv
Output : {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_assignment_optimisation.csv
         {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_skipped_groups.json
         {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_results.csv   (if sweep)
         {INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_summary.json  (if sweep)

Column notes vs v1.x
--------------------
- v2 uses  perf_type  where v1.x used  aircrafttype
- v2 uses  mtow       where v1.x used  max_tow
- FH/cycle rate columns: total_fh_rate / total_cycle_rate / opti_fh_rate / opti_cycle_rate (all lowercase)
- FH cost uses total_pred_act_hours (predicted airborne hours), not block hours

References
----------
v1.x equivalent: src/opti_tails_algorithm.py
"""

import json
import os

import numpy as np
import polars as pl
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Set,
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    Var,
    minimize,
    value,
)

from config import (
    AUTO_SELECT_EPS,
    AUTO_SELECT_EPS_METHOD,
    AUTOLAND_AIRCRAFTREG,
    CAPE_VERDE_IATA,
    CYPRUS_IATA,
    CYPRUS_PROHIBITED,
    DATE_PREFIX,
    EPS_VALUES,
    FUEL_PRICE,
    INTERMEDIATE_DIRECTORY,
    IRIS_EU_AIRCRAFTREG,
    IRIS_UK_AIRCRAFTREG,
    KEF_IATA,
    NEO_AIRCRAFT_TYPES,
    SELECTED_EPS,
    SOLVER_BACKEND,
)

INPUT_FILE  = f"{DATE_PREFIX}_eligibility_filter.csv"
OUTPUT_FILE = f"{DATE_PREFIX}_assignment_optimisation.csv"


# ── ε-selection helpers (identical to v1.x) ──────────────────────────────────

def _kneedle_select_eps(sweep_summary: dict) -> tuple[float, float] | tuple[None, None]:
    from scipy.interpolate import PchipInterpolator

    eps_keys = sorted(sweep_summary.keys(), key=float)
    if len(eps_keys) < 2:
        return None, None

    xs, ys = [], []
    for k in eps_keys:
        s = sweep_summary[k]
        baseline_usd = s.get("baseline_fuel_usd", 0.0)
        fuel_usd = s.get("total_fuel_cost_usd", 0.0)
        xs.append(float(k))
        ys.append(baseline_usd - fuel_usd)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    interp = PchipInterpolator(xs_arr, ys_arr)
    x_fine = np.linspace(xs_arr[0], xs_arr[-1], 10_000)
    y_fine = interp(x_fine)

    x_min, x_max = float(x_fine[0]), float(x_fine[-1])
    y_min, y_max = float(y_fine.min()), float(y_fine.max())

    if x_max == x_min or y_max == y_min:
        return xs[0], xs[0]

    xs_n = (x_fine - x_min) / (x_max - x_min)
    ys_n = (y_fine - y_min) / (y_max - y_min)

    best_idx = int(np.argmax(ys_n - xs_n))
    eps_star = round(float(x_fine[best_idx]), 6)
    eps_applied = min(xs, key=lambda e: abs(e - eps_star))
    return eps_star, eps_applied


def _chord_select_eps(sweep_summary: dict) -> tuple[float, float] | tuple[None, None]:
    from scipy.interpolate import PchipInterpolator

    eps_keys = sorted(sweep_summary.keys(), key=float)
    if len(eps_keys) < 2:
        return None, None

    xs, ys = [], []
    for k in eps_keys:
        s = sweep_summary[k]
        baseline_usd = s.get("baseline_fuel_usd", 0.0)
        fuel_usd = s.get("total_fuel_cost_usd", 0.0)
        xs.append(float(k))
        ys.append(baseline_usd - fuel_usd)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    interp = PchipInterpolator(xs_arr, ys_arr)
    x_fine = np.linspace(xs_arr[0], xs_arr[-1], 10_000)
    y_fine = interp(x_fine)

    x0, y0 = x_fine[0], y_fine[0]
    x1, y1 = x_fine[-1], y_fine[-1]
    dx, dy = x1 - x0, y1 - y0
    line_len = np.hypot(dx, dy)
    if line_len < 1e-12:
        return xs[0], xs[0]

    # Signed perpendicular offset from the endpoint chord; > 0 means the curve
    # lies ABOVE the chord. Taking np.abs() here would let argmax land on a
    # point BELOW the chord — a convex "anti-elbow" — which fabricates an elbow
    # on curves that have none (e.g. an accelerating saving curve, where the
    # correct answer is ε* = 0). The signed form restricts the search to the
    # genuine above-chord elbow and makes this method algebraically identical
    # to Kneedle on monotone-increasing curves.
    dist = (dx * (y_fine - y0) - dy * (x_fine - x0)) / line_len
    best_idx = int(np.argmax(dist))
    eps_star = round(float(x_fine[best_idx]), 6)
    eps_applied = min(xs, key=lambda e: abs(e - eps_star))
    return eps_star, eps_applied


def _curvature_select_eps(sweep_summary: dict) -> tuple[float, float] | tuple[None, None]:
    from scipy.interpolate import PchipInterpolator

    eps_keys = sorted(sweep_summary.keys(), key=float)
    if len(eps_keys) < 2:
        return None, None

    xs, ys = [], []
    for k in eps_keys:
        s = sweep_summary[k]
        baseline_usd = s.get("baseline_fuel_usd", 0.0)
        fuel_usd = s.get("total_fuel_cost_usd", 0.0)
        xs.append(float(k))
        ys.append(baseline_usd - fuel_usd)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    x_min, x_max = float(xs_arr[0]), float(xs_arr[-1])
    y_min, y_max = float(ys_arr.min()), float(ys_arr.max())
    if x_max == x_min or y_max == y_min:
        return xs[0], xs[0]

    xs_n = (xs_arr - x_min) / (x_max - x_min)
    ys_n = (ys_arr - y_min) / (y_max - y_min)

    interp = PchipInterpolator(xs_n, ys_n)
    x_fine = np.linspace(xs_n[0], xs_n[-1], 10_000)

    dy  = interp(x_fine, 1)
    d2y = interp(x_fine, 2)
    kappa = np.abs(d2y) / (1.0 + dy ** 2) ** 1.5

    best_idx = int(np.argmax(kappa))
    eps_star_n = float(x_fine[best_idx])
    eps_star = round(eps_star_n * (x_max - x_min) + x_min, 6)
    eps_applied = min(xs, key=lambda e: abs(e - eps_star))
    return eps_star, eps_applied


# ── Solvers (identical to v1.x) ───────────────────────────────────────────────

def pyomo_gurobi_highs_assignment(cost_matrix, forbidden_pairs=None):
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n_rows == 1 and n_cols == 1:
        return np.array([0]), np.array([0])

    rows = list(range(n_rows))
    cols = list(range(n_cols))
    is_square = (n_rows == n_cols)

    cost = {(i, j): float(cost_matrix[i, j]) for i in rows for j in cols}

    model = ConcreteModel()
    model.rows = Set(initialize=rows)
    model.cols = Set(initialize=cols)
    model.x = Var(model.rows, model.cols, domain=Binary)

    model.obj = Objective(
        expr=sum(cost[i, j] * model.x[i, j] for i in rows for j in cols),
        sense=minimize,
    )

    if is_square or n_rows <= n_cols:
        model.row_assign = Constraint(model.rows,
            rule=lambda m, i: sum(m.x[i, j] for j in cols) == 1)
    else:
        model.row_assign = Constraint(model.rows,
            rule=lambda m, i: sum(m.x[i, j] for j in cols) <= 1)

    if is_square or n_cols <= n_rows:
        model.col_assign = Constraint(model.cols,
            rule=lambda m, j: sum(m.x[i, j] for i in rows) == 1)
    else:
        model.col_assign = Constraint(model.cols,
            rule=lambda m, j: sum(m.x[i, j] for i in rows) <= 1)

    if forbidden_pairs:
        for (i, j) in forbidden_pairs:
            if i in rows and j in cols:
                model.x[i, j].fix(0)

    solver = SolverFactory("appsi_highs")
    solver.options["output_flag"] = False
    solver.options["mip_rel_gap"] = 0
    solver.options["time_limit"] = 300

    result = solver.solve(model)

    if (result.solver.status != SolverStatus.ok or
            result.solver.termination_condition != TerminationCondition.optimal):
        raise RuntimeError(
            f"Solver failed: status={result.solver.status}, "
            f"termination={result.solver.termination_condition}"
        )

    row_ind, col_ind = [], []
    for i in rows:
        for j in cols:
            if value(model.x[i, j]) > 0.5:
                row_ind.append(i)
                col_ind.append(j)
                break

    order = np.argsort(row_ind)
    return np.array(row_ind)[order], np.array(col_ind)[order]


def pyomo_fuel_constrained_assignment(fuel_matrix, cost_matrix, cost_budget, forbidden_pairs=None):
    """Minimise fuel cost s.t. total cost ≤ cost_budget."""
    n_rows, n_cols = fuel_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return None, None
    if n_rows == 1 and n_cols == 1:
        return np.array([0]), np.array([0])

    rows = list(range(n_rows))
    cols = list(range(n_cols))
    is_square = (n_rows == n_cols)

    fuel = {(i, j): float(fuel_matrix[i, j]) for i in rows for j in cols}
    cost = {(i, j): float(cost_matrix[i, j]) for i in rows for j in cols}

    model = ConcreteModel()
    model.rows = Set(initialize=rows)
    model.cols = Set(initialize=cols)
    model.x = Var(model.rows, model.cols, domain=Binary)

    model.obj = Objective(
        expr=sum(fuel[i, j] * model.x[i, j] for i in rows for j in cols),
        sense=minimize,
    )

    if is_square or n_rows <= n_cols:
        model.row_assign = Constraint(model.rows,
            rule=lambda m, i: sum(m.x[i, j] for j in cols) == 1)
    else:
        model.row_assign = Constraint(model.rows,
            rule=lambda m, i: sum(m.x[i, j] for j in cols) <= 1)

    if is_square or n_cols <= n_rows:
        model.col_assign = Constraint(model.cols,
            rule=lambda m, j: sum(m.x[i, j] for i in rows) == 1)
    else:
        model.col_assign = Constraint(model.cols,
            rule=lambda m, j: sum(m.x[i, j] for i in rows) <= 1)

    model.budget = Constraint(
        expr=sum(cost[i, j] * model.x[i, j] for i in rows for j in cols) <= cost_budget
    )

    if forbidden_pairs:
        for (i, j) in forbidden_pairs:
            if i in rows and j in cols:
                model.x[i, j].fix(0)

    solver = SolverFactory("appsi_highs")
    solver.options["output_flag"] = False
    solver.options["mip_rel_gap"] = 0
    solver.options["time_limit"] = 300

    result = solver.solve(model)

    if (result.solver.status != SolverStatus.ok or
            result.solver.termination_condition != TerminationCondition.optimal):
        return None, None

    row_ind, col_ind = [], []
    for i in rows:
        for j in cols:
            if value(model.x[i, j]) > 0.5:
                row_ind.append(i)
                col_ind.append(j)
                break

    order = np.argsort(row_ind)
    return np.array(row_ind)[order], np.array(col_ind)[order]


def _gurobi_cm_env():
    import gurobipy as gp
    return gp.Env(params={
        "CSManager":     os.environ.get("GUROBI_MANAGER", ""),
        "CSAPIAccessID": os.environ.get("GUROBI_ACCESS_ID", ""),
        "CSAPISecret":   os.environ.get("GUROBI_SECRET", ""),
        "CSAppName":     os.environ.get("GUROBI_APP_NAME", "OPTI_TAIL_ALLOCATION_KEY"),
        "CSGroup":       os.environ.get("GUROBI_GROUP", "Prod:0"),
        "OutputFlag":    0,
        "LogToConsole":  0,
    })


def _gurobi_call_with_retry(fn, *args, retries=3, backoff=5, **kwargs):
    """Call fn(*args, **kwargs), retrying on transient GurobiError (network drops, code 23)."""
    import time
    import gurobipy
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except gurobipy.GurobiError as exc:
            last_exc = exc
            print(f"[Gurobi] transient error (attempt {attempt + 1}/{retries}): {exc}", flush=True)
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_exc


def gurobi_cm_assignment(cost_matrix, forbidden_pairs=None):
    import gurobipy as gp
    from gurobipy import GRB

    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n_rows == 1 and n_cols == 1:
        return np.array([0]), np.array([0])

    rows = list(range(n_rows))
    cols = list(range(n_cols))
    is_square = (n_rows == n_cols)

    env = _gurobi_cm_env()
    with gp.Model(env=env) as model:
        model.setParam("MIPGap", 0)
        model.setParam("TimeLimit", 300)

        x = model.addVars(rows, cols, vtype=GRB.BINARY, name="x")
        model.setObjective(
            gp.quicksum(float(cost_matrix[i, j]) * x[i, j] for i in rows for j in cols),
            GRB.MINIMIZE,
        )

        if is_square or n_rows <= n_cols:
            for i in rows:
                model.addConstr(gp.quicksum(x[i, j] for j in cols) == 1)
        else:
            for i in rows:
                model.addConstr(gp.quicksum(x[i, j] for j in cols) <= 1)

        if is_square or n_cols <= n_rows:
            for j in cols:
                model.addConstr(gp.quicksum(x[i, j] for i in rows) == 1)
        else:
            for j in cols:
                model.addConstr(gp.quicksum(x[i, j] for i in rows) <= 1)

        if forbidden_pairs:
            for (i, j) in forbidden_pairs:
                if i in rows and j in cols:
                    model.addConstr(x[i, j] == 0)

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi Cluster Manager failed: status={model.Status}")

        row_ind, col_ind = [], []
        for i in rows:
            for j in cols:
                if x[i, j].X > 0.5:
                    row_ind.append(i)
                    col_ind.append(j)
                    break

    order = np.argsort(row_ind)
    return np.array(row_ind)[order], np.array(col_ind)[order]


def gurobi_cm_fuel_constrained_assignment(fuel_matrix, cost_matrix, cost_budget, forbidden_pairs=None):
    import gurobipy as gp
    from gurobipy import GRB

    n_rows, n_cols = fuel_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return None, None
    if n_rows == 1 and n_cols == 1:
        return np.array([0]), np.array([0])

    rows = list(range(n_rows))
    cols = list(range(n_cols))
    is_square = (n_rows == n_cols)

    env = _gurobi_cm_env()
    with gp.Model(env=env) as model:
        model.setParam("MIPGap", 0)
        model.setParam("TimeLimit", 300)

        x = model.addVars(rows, cols, vtype=GRB.BINARY, name="x")
        model.setObjective(
            gp.quicksum(float(fuel_matrix[i, j]) * x[i, j] for i in rows for j in cols),
            GRB.MINIMIZE,
        )

        if is_square or n_rows <= n_cols:
            for i in rows:
                model.addConstr(gp.quicksum(x[i, j] for j in cols) == 1)
        else:
            for i in rows:
                model.addConstr(gp.quicksum(x[i, j] for j in cols) <= 1)

        if is_square or n_cols <= n_rows:
            for j in cols:
                model.addConstr(gp.quicksum(x[i, j] for i in rows) == 1)
        else:
            for j in cols:
                model.addConstr(gp.quicksum(x[i, j] for i in rows) <= 1)

        model.addConstr(
            gp.quicksum(float(cost_matrix[i, j]) * x[i, j] for i in rows for j in cols) <= cost_budget
        )

        if forbidden_pairs:
            for (i, j) in forbidden_pairs:
                if i in rows and j in cols:
                    model.addConstr(x[i, j] == 0)

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return None, None

        row_ind, col_ind = [], []
        for i in rows:
            for j in cols:
                if x[i, j].X > 0.5:
                    row_ind.append(i)
                    col_ind.append(j)
                    break

    order = np.argsort(row_ind)
    return np.array(row_ind)[order], np.array(col_ind)[order]


# ── Feasibility check ────────────────────────────────────────────────────────

def _check_feasibility(n, forbidden_pairs):
    for j in range(n):
        if all((i, j) in forbidden_pairs for i in range(n)):
            return False, f"route index {j} has no eligible aircraft"
    for i in range(n):
        if all((i, j) in forbidden_pairs for j in range(n)):
            return False, f"aircraft index {i} has no eligible route"
    return True, ""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(csv_filename: str) -> pl.DataFrame:
    """Load eligibility_filter output and clean rate columns."""
    df = pl.read_csv(f"{INTERMEDIATE_DIRECTORY}/{csv_filename}")
    df = df.rename({c: c.strip() for c in df.columns})

    df = df.with_columns(
        pl.col("flt_date").cast(pl.Utf8).str.strip_chars()
            .str.to_datetime(strict=False)
            .dt.strftime("%d/%m/%Y")
            .str.to_datetime("%d/%m/%Y")
    )

    for col_name in ["total_fh_rate", "total_cycle_rate"]:
        if col_name in df.columns:
            df = df.with_columns(
                pl.col(col_name).cast(pl.Utf8).str.strip_chars()
                    .str.replace_all(r"[\$,]", "")
                    .str.strip_chars()
                    .str.replace(r"^[-]?$", "0")
                    .str.replace("nan", "0")
                    .str.replace("None", "0")
                    .str.replace(r"^\s*$", "0")
                    .cast(pl.Float64)
            )

    critical_columns = [
        "flt_date", "base", "aircraftreg", "aircrafttype", "perf_type", "mtow",
        "avg_perf_corr", "total_baseline_fuel", "total_pred_act_hours",
        "total_sectors", "total_fh_rate", "total_cycle_rate", "seat_config",
    ]
    if "Route" in df.columns:
        critical_columns.append("Route")
    if "flt_numbers" in df.columns:
        critical_columns.append("flt_numbers")

    before = len(df)
    df = df.drop_nulls(subset=critical_columns)
    dropped = before - len(df)
    if dropped:
        print(f"  WARNING: {dropped} rows dropped (null in critical columns)")

    if "total_trip_fuel" not in df.columns:
        df = df.with_columns(
            (pl.col("total_baseline_fuel") * pl.col("avg_perf_corr")).alias("total_trip_fuel")
        ).with_columns(
            (pl.col("total_trip_fuel") - pl.col("total_baseline_fuel")).alias("total_deg_burn")
        )

    return df


# ── Group optimisation ────────────────────────────────────────────────────────

def optimize_group(group_df, group_index, forbidden_pairs=None):
    n = len(group_df)
    if n == 0:
        return None

    if forbidden_pairs is None:
        forbidden_pairs = set()

    route_in_df = "Route" in group_df.columns

    aircraft_perf_corr  = group_df["avg_perf_corr"].to_numpy()
    aircraft_fh_rate    = group_df["total_fh_rate"].to_numpy()
    aircraft_cycle_rate = group_df["total_cycle_rate"].to_numpy()

    route_fuel    = group_df["total_baseline_fuel"].to_numpy()
    route_hours   = group_df["total_pred_act_hours"].to_numpy()
    route_sectors = group_df["total_sectors"].to_numpy()

    fuel_cost  = np.outer(aircraft_perf_corr, route_fuel) * FUEL_PRICE
    fh_cost    = np.outer(aircraft_fh_rate, route_hours)
    cycle_cost = np.outer(aircraft_cycle_rate, route_sectors)
    cost_matrix = fuel_cost + fh_cost + cycle_cost

    current_cost = np.trace(cost_matrix)

    feasible, reason = _check_feasibility(n, forbidden_pairs)
    if not feasible:
        aircraftreg_list    = group_df["aircraftreg"].to_list()
        base_list           = group_df["base"].to_list()
        perf_type_list      = group_df["perf_type"].to_list()
        mtow_list           = group_df["mtow"].to_list()
        avg_perf_corr_list  = group_df["avg_perf_corr"].to_list()
        seat_config_list    = group_df["seat_config"].to_list()
        total_trip_fuel_list  = group_df["total_trip_fuel"].to_list()
        total_deg_burn_list   = group_df["total_deg_burn"].to_list()
        flt_date_list       = group_df["flt_date"].to_list()
        route_fuel_list     = route_fuel.tolist()
        route_hours_list    = route_hours.tolist()
        route_sectors_list  = route_sectors.tolist()
        fh_rate_list        = aircraft_fh_rate.tolist()
        cycle_rate_list     = aircraft_cycle_rate.tolist()
        if route_in_df:
            route_list       = group_df["Route"].to_list()
            flt_numbers_list = group_df["flt_numbers"].to_list()

        results = []
        for i in range(n):
            row = {
                "group_index": group_index,
                "flt_date":    flt_date_list[i],
                "aircraftreg": aircraftreg_list[i],
                "base":        base_list[i],
                "perf_type":   perf_type_list[i],
                "mtow":        mtow_list[i],
                "avg_perf_corr": avg_perf_corr_list[i],
                "seat_config": seat_config_list[i],
                "total_trip_fuel":       total_trip_fuel_list[i],
                "total_deg_burn":        total_deg_burn_list[i],
                "total_baseline_fuel":   route_fuel_list[i],
                "total_pred_act_hours": route_hours_list[i],
                "total_sectors":         route_sectors_list[i],
                "total_fh_rate":   fh_rate_list[i],
                "total_cycle_rate": cycle_rate_list[i],
                "opti_aircraftreg":              aircraftreg_list[i],
                "opti_aircraftreg_avg_perf_corr": avg_perf_corr_list[i],
                "opti_seat_config":              seat_config_list[i],
                "opti_total_baseline_fuel":      route_fuel_list[i],
                "opti_fh_rate":   fh_rate_list[i],
                "opti_cycle_rate": cycle_rate_list[i],
                "opti_fuel_used": route_fuel_list[i] * avg_perf_corr_list[i],
                "fuel_delta":     0.0,
                "current_cost":   float(cost_matrix[i, i]),
                "opti_cost":      float(cost_matrix[i, i]),
                "savings":        0.0,
                "changed":        False,
            }
            if route_in_df:
                row["Route"]       = route_list[i]
                row["flt_numbers"] = flt_numbers_list[i]
            results.append(row)

        return results, float(current_cost), float(current_cost), 0.0, True

    if SOLVER_BACKEND == "gurobi":
        row_ind, col_ind = _gurobi_call_with_retry(
            gurobi_cm_assignment, cost_matrix, forbidden_pairs
        )
    else:
        row_ind, col_ind = pyomo_gurobi_highs_assignment(cost_matrix, forbidden_pairs)

    optimal_assignment = col_ind
    optimal_cost = cost_matrix[np.arange(n), optimal_assignment].sum()
    savings = current_cost - optimal_cost

    # Hungarian validation — square unconstrained groups only (forbidden pairs break TU).
    # Runs at ε=0 cost objective. Never influences the output; logs a warning if MIP
    # and Hungarian disagree by more than floating-point noise.
    if not forbidden_pairs and n > 1:
        from scipy.optimize import linear_sum_assignment
        h_row, h_col = linear_sum_assignment(cost_matrix)
        h_cost = cost_matrix[h_row, h_col].sum()
        if abs(h_cost - optimal_cost) > 1e-4:
            print(
                f"  [VALIDATION] group {group_index}: MIP cost={optimal_cost:.4f}, "
                f"Hungarian cost={h_cost:.4f}, delta={optimal_cost - h_cost:.4f}",
                flush=True,
            )

    reverse_assignment = np.empty(n, dtype=int)
    reverse_assignment[optimal_assignment] = np.arange(n)

    aircraftreg_list    = group_df["aircraftreg"].to_list()
    base_list           = group_df["base"].to_list()
    perf_type_list      = group_df["perf_type"].to_list()
    mtow_list           = group_df["mtow"].to_list()
    avg_perf_corr_list  = group_df["avg_perf_corr"].to_list()
    seat_config_list    = group_df["seat_config"].to_list()
    total_trip_fuel_list  = group_df["total_trip_fuel"].to_list()
    total_deg_burn_list   = group_df["total_deg_burn"].to_list()
    flt_date_list       = group_df["flt_date"].to_list()
    route_fuel_list     = route_fuel.tolist()
    route_hours_list    = route_hours.tolist()
    route_sectors_list  = route_sectors.tolist()
    fh_rate_list        = aircraft_fh_rate.tolist()
    cycle_rate_list     = aircraft_cycle_rate.tolist()
    if route_in_df:
        route_list       = group_df["Route"].to_list()
        flt_numbers_list = group_df["flt_numbers"].to_list()

    results = []
    for i in range(n):
        optimal_aircraft_idx = reverse_assignment[i]
        opti_fuel_used = route_fuel_list[i] * avg_perf_corr_list[optimal_aircraft_idx]
        current_cost_route = cost_matrix[i, i]
        opti_cost_route = cost_matrix[optimal_aircraft_idx, i]

        row = {
            "group_index": group_index,
            "flt_date":    flt_date_list[i],
            "aircraftreg": aircraftreg_list[i],
            "base":        base_list[i],
            "perf_type":   perf_type_list[i],
            "mtow":        mtow_list[i],
            "avg_perf_corr": avg_perf_corr_list[i],
            "seat_config": seat_config_list[i],
            "total_trip_fuel":        total_trip_fuel_list[i],
            "total_deg_burn":         total_deg_burn_list[i],
            "total_baseline_fuel":    route_fuel_list[i],
            "total_pred_act_hours": route_hours_list[i],
            "total_sectors":        route_sectors_list[i],
            "total_fh_rate":   fh_rate_list[i],
            "total_cycle_rate": cycle_rate_list[i],
            "opti_aircraftreg":              aircraftreg_list[optimal_aircraft_idx],
            "opti_aircraftreg_avg_perf_corr": avg_perf_corr_list[optimal_aircraft_idx],
            "opti_seat_config":              seat_config_list[optimal_aircraft_idx],
            "opti_total_baseline_fuel":      route_fuel_list[i],
            "opti_fh_rate":   fh_rate_list[optimal_aircraft_idx],
            "opti_cycle_rate": cycle_rate_list[optimal_aircraft_idx],
            "opti_fuel_used": opti_fuel_used,
            "fuel_delta":     total_trip_fuel_list[i] - opti_fuel_used,
            "current_cost":   float(current_cost_route),
            "opti_cost":      float(opti_cost_route),
            "savings":        float(current_cost_route - opti_cost_route),
            "changed":        i != optimal_aircraft_idx,
        }
        if route_in_df:
            row["Route"]       = route_list[i]
            row["flt_numbers"] = flt_numbers_list[i]
        results.append(row)

    return results, float(current_cost), float(optimal_cost), float(savings), False


def optimize_group_eps(group_df, group_index, forbidden_pairs, eps_values, c_star):
    """ε-sweep for one group. Returns list of (eps, result_rows)."""
    n = len(group_df)
    if n == 0 or not eps_values:
        return []

    route_in_df = "Route" in group_df.columns

    aircraft_perf_corr  = group_df["avg_perf_corr"].to_numpy()
    aircraft_fh_rate    = group_df["total_fh_rate"].to_numpy()
    aircraft_cycle_rate = group_df["total_cycle_rate"].to_numpy()
    route_fuel    = group_df["total_baseline_fuel"].to_numpy()
    route_hours   = group_df["total_pred_act_hours"].to_numpy()
    route_sectors = group_df["total_sectors"].to_numpy()

    fuel_matrix  = np.outer(aircraft_perf_corr, route_fuel) * FUEL_PRICE
    fh_matrix    = np.outer(aircraft_fh_rate, route_hours)
    cycle_matrix = np.outer(aircraft_cycle_rate, route_sectors)
    cost_matrix  = fuel_matrix + fh_matrix + cycle_matrix

    aircraftreg_list    = group_df["aircraftreg"].to_list()
    base_list           = group_df["base"].to_list()
    perf_type_list      = group_df["perf_type"].to_list()
    mtow_list           = group_df["mtow"].to_list()
    avg_perf_corr_list  = group_df["avg_perf_corr"].to_list()
    seat_config_list    = group_df["seat_config"].to_list()
    total_trip_fuel_list  = group_df["total_trip_fuel"].to_list()
    total_deg_burn_list   = group_df["total_deg_burn"].to_list()
    flt_date_list       = group_df["flt_date"].to_list()
    route_fuel_list     = route_fuel.tolist()
    route_hours_list    = route_hours.tolist()
    route_sectors_list  = route_sectors.tolist()
    fh_rate_list        = aircraft_fh_rate.tolist()
    cycle_rate_list     = aircraft_cycle_rate.tolist()
    if route_in_df:
        route_list       = group_df["Route"].to_list()
        flt_numbers_list = group_df["flt_numbers"].to_list()

    eps_results = []

    for eps in eps_values:
        budget = float(c_star) * (1.0 + eps)
        if SOLVER_BACKEND == "gurobi":
            row_ind, col_ind = _gurobi_call_with_retry(
                gurobi_cm_fuel_constrained_assignment,
                fuel_matrix, cost_matrix, budget, forbidden_pairs
            )
        else:
            row_ind, col_ind = pyomo_fuel_constrained_assignment(
                fuel_matrix, cost_matrix, budget, forbidden_pairs
            )
        if row_ind is None:
            continue

        reverse_assignment = np.empty(n, dtype=int)
        reverse_assignment[col_ind] = np.arange(n)

        result_rows = []
        for i in range(n):
            opt_idx = reverse_assignment[i]
            opti_fuel_used = route_fuel_list[i] * avg_perf_corr_list[opt_idx]
            row = {
                "epsilon":     eps,
                "group_index": group_index,
                "flt_date":    flt_date_list[i],
                "aircraftreg": aircraftreg_list[i],
                "base":        base_list[i],
                "perf_type":   perf_type_list[i],
                "mtow":        mtow_list[i],
                "avg_perf_corr": avg_perf_corr_list[i],
                "seat_config": seat_config_list[i],
                "total_trip_fuel":        total_trip_fuel_list[i],
                "total_deg_burn":         total_deg_burn_list[i],
                "total_baseline_fuel":    route_fuel_list[i],
                "total_pred_act_hours": route_hours_list[i],
                "total_sectors":          route_sectors_list[i],
                "total_fh_rate":   fh_rate_list[i],
                "total_cycle_rate": cycle_rate_list[i],
                "opti_aircraftreg":              aircraftreg_list[opt_idx],
                "opti_aircraftreg_avg_perf_corr": avg_perf_corr_list[opt_idx],
                "opti_seat_config":              seat_config_list[opt_idx],
                "opti_total_baseline_fuel":      route_fuel_list[i],
                "opti_fh_rate":   fh_rate_list[opt_idx],
                "opti_cycle_rate": cycle_rate_list[opt_idx],
                "opti_fuel_used": opti_fuel_used,
                "fuel_delta":     total_trip_fuel_list[i] - opti_fuel_used,
                "current_cost":   float(cost_matrix[i, i]),
                "opti_cost":      float(cost_matrix[opt_idx, i]),
                "savings":        float(cost_matrix[i, i] - cost_matrix[opt_idx, i]),
                "changed":        i != opt_idx,
            }
            if route_in_df:
                row["Route"]       = route_list[i]
                row["flt_numbers"] = flt_numbers_list[i]
            result_rows.append(row)

        eps_results.append((eps, result_rows))

    return eps_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Stage 5  Assignment Optimisation")
    print(f"Solver  : {SOLVER_BACKEND.upper()}")
    print(f"Input   : {INPUT_FILE}")
    print("=" * 80)

    df = load_data(INPUT_FILE)
    print(f"\nTails loaded : {len(df)}")
    print("Grouping by flt_date, base, aircrafttype, mtow, seat_config...")

    partitions = df.partition_by(
        ["flt_date", "base", "aircrafttype", "mtow", "seat_config"],
        as_dict=True,
    )
    print(f"Groups       : {len(partitions)}")

    all_iris  = set(IRIS_UK_AIRCRAFTREG + IRIS_EU_AIRCRAFTREG)
    cv_set    = set(CAPE_VERDE_IATA)
    cy_set    = set(CYPRUS_IATA)
    kef_set   = set(KEF_IATA)
    auto_set  = set(AUTOLAND_AIRCRAFTREG)
    neo_types = set(NEO_AIRCRAFT_TYPES)
    cy_ban    = set(CYPRUS_PROHIBITED)

    all_results    = []
    all_eps_rows   = []
    skipped_groups = []
    total_current_cost  = 0.0
    total_optimal_cost  = 0.0
    total_savings       = 0.0
    groups_processed    = 0
    base_group_counters = {}

    eps_group_summary = {
        eps: {
            "total_opti_cost":    0.0,
            "total_fuel_cost_kg": 0.0,
            "baseline_fuel_kg":   0.0,
            "n_changed":          0,
            "n_groups":           0,
        }
        for eps in EPS_VALUES
    }

    for group_key, group in partitions.items():
        base = group_key[1]

        if base not in base_group_counters:
            base_group_counters[base] = 1
        else:
            base_group_counters[base] += 1
        group_index = f"{base}_{base_group_counters[base]:03d}"

        aircraft_in_group = group["aircraftreg"].to_list()
        aircrafttype_val = group_key[2]

        forbidden_pairs: set = set()
        suffix = ""

        if "Route" in group.columns:
            routes = group["Route"].to_list()
            base_iata = base.replace("LGW-S", "LGW")

            group_iatas: set = set()
            for route in routes:
                group_iatas |= set(route.split("-")) - {base_iata}

            if group_iatas & cv_set:
                suffix += "_CV"
            if group_iatas & kef_set:
                suffix += "_KF"
            if group_iatas & cy_set:
                suffix += "_CY"

            for j, route in enumerate(routes):
                visited = set(route.split("-")) - {base_iata}
                for i, reg in enumerate(aircraft_in_group):
                    if visited & cv_set and reg not in all_iris:
                        forbidden_pairs.add((i, j))
                    if visited & cy_set and reg in cy_ban:
                        forbidden_pairs.add((i, j))
                    if visited & kef_set and aircrafttype_val not in neo_types and reg not in auto_set:
                        forbidden_pairs.add((i, j))

        if suffix:
            group_index = f"{group_index}{suffix}"

        result = optimize_group(group, group_index, forbidden_pairs)

        if result:
            group_results, current_cost, optimal_cost, savings, was_skipped = result
            all_results.extend(group_results)
            total_current_cost += current_cost
            total_optimal_cost += optimal_cost
            total_savings      += savings
            groups_processed   += 1

            if was_skipped:
                skipped_groups.append({
                    "group_index": group_index,
                    "reason": "infeasible constraints — route has no eligible aircraft",
                })
            elif EPS_VALUES:
                eps_sweep = optimize_group_eps(
                    group, group_index, forbidden_pairs, EPS_VALUES, optimal_cost
                )
                _group_s1_fuel = sum(r["total_trip_fuel"] for r in group_results)
                for eps, result_rows in eps_sweep:
                    all_eps_rows.extend(result_rows)
                    s = eps_group_summary[eps]
                    s["total_opti_cost"]    += sum(r["opti_cost"]      for r in result_rows)
                    s["total_fuel_cost_kg"] += sum(r["opti_fuel_used"] for r in result_rows)
                    s["baseline_fuel_kg"]   += _group_s1_fuel
                    s["n_changed"]          += sum(1 for r in result_rows if r["changed"])
                    s["n_groups"]           += 1

        if groups_processed % 10 == 0 and groups_processed:
            print(f"  Processed {groups_processed}/{len(partitions)} groups...")

    print(f"\nCompleted {groups_processed} groups")
    print(f"  Total current cost : ${total_current_cost:,.2f}")
    print(f"  Total optimal cost : ${total_optimal_cost:,.2f}")
    print(f"  Total savings      : ${total_savings:,.2f}")

    if skipped_groups:
        names = [g["group_index"] for g in skipped_groups]
        print(f"WARNING: {len(skipped_groups)} group(s) skipped (infeasible): {names}")

    results_df = pl.DataFrame(all_results)

    out_path = f"{INTERMEDIATE_DIRECTORY}/{OUTPUT_FILE}"
    results_df.write_csv(out_path)
    print(f"\nResults written : {out_path}")

    skipped_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_skipped_groups.json"
    with open(skipped_path, "w") as f:
        json.dump(skipped_groups, f, indent=2)

    # ── ε-sweep outputs ───────────────────────────────────────────────────────
    if EPS_VALUES and all_eps_rows:
        eps_df = pl.DataFrame(all_eps_rows)
        eps_csv_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_results.csv"
        eps_df.write_csv(eps_csv_path)
        print(f"eps-sweep results : {eps_csv_path}")

        sweep_summary = {}
        for eps in EPS_VALUES:
            s = eps_group_summary.get(eps, {})
            _baseline_kg = s.get("baseline_fuel_kg", 0.0)
            sweep_summary[str(eps)] = {
                "total_opti_cost":     round(s.get("total_opti_cost", 0.0), 2),
                "total_fuel_cost_kg":  round(s.get("total_fuel_cost_kg", 0.0), 2),
                "total_fuel_cost_usd": round(s.get("total_fuel_cost_kg", 0.0) * FUEL_PRICE, 2),
                "n_changed":           s.get("n_changed", 0),
                "n_groups":            s.get("n_groups", 0),
                "baseline_fuel_kg":    round(_baseline_kg, 2),
                "baseline_fuel_usd":   round(_baseline_kg * FUEL_PRICE, 2),
            }

        summary_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(sweep_summary, f, indent=2)
        print(f"eps-sweep summary : {summary_path}")

        # ── Auto-select ε* ────────────────────────────────────────────────────
        eps_star = None
        auto_selected_eps = None
        applied_eps = None

        if AUTO_SELECT_EPS:
            _selectors = {
                "chord":     _chord_select_eps,
                "curvature": _curvature_select_eps,
            }
            _selector = _selectors.get(AUTO_SELECT_EPS_METHOD, _kneedle_select_eps)
            eps_star, auto_selected_eps = _selector(sweep_summary)
            if auto_selected_eps is not None:
                applied_eps = auto_selected_eps
                print(
                    f"{AUTO_SELECT_EPS_METHOD} eps*={eps_star:.6f} ({eps_star * 100:.4f}%) "
                    f"-> nearest grid eps={auto_selected_eps} applied"
                )
        elif SELECTED_EPS is not None:
            if SELECTED_EPS not in EPS_VALUES:
                print(
                    f"WARNING: SELECTED_EPS={SELECTED_EPS} is not in EPS_VALUES "
                    "— keeping standard cost-optimal result"
                )
            else:
                applied_eps = SELECTED_EPS
                auto_selected_eps = SELECTED_EPS

        if applied_eps is not None:
            chosen_rows = [
                {k: v for k, v in r.items() if k != "epsilon"}
                for r in all_eps_rows
                if r["epsilon"] == applied_eps
            ]
            if chosen_rows:
                pl.DataFrame(chosen_rows).write_csv(out_path)
                print(f"Applied eps={applied_eps}: {OUTPUT_FILE} now reflects eps solution")

        for k in sweep_summary:
            sweep_summary[k]["eps_star"]           = eps_star
            sweep_summary[k]["kneedle_eps_star"]   = eps_star
            sweep_summary[k]["auto_selected_eps"]  = auto_selected_eps
            sweep_summary[k]["auto_select_method"] = AUTO_SELECT_EPS_METHOD if AUTO_SELECT_EPS else None
        with open(summary_path, "w") as f:
            json.dump(sweep_summary, f, indent=2)
        print(f"eps-sweep summary updated: eps*={eps_star}, applied={auto_selected_eps}")

    print(f"\nColumns : {results_df.columns}")
    print("\nStage 5 Assignment Optimisation complete.")


if __name__ == "__main__":
    main()
