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
- Optional global ε-constraint sweep: minimise daily fuel s.t. daily cost ≤ C*(1+ε)
- Lexicographic tie-breaks: minimum fuel, then cost, then reassignment count
- Auto-select an actual solved point from the validated cost/fuel concave hull
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
    ConstraintList,
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
    EPS_MAX_COST_PER_FUEL_KG,
    EPS_KNEE_MIN_PROMINENCE,
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

from src.eps_frontier import annotate_summary, select_frontier_point

INPUT_FILE  = f"{DATE_PREFIX}_eligibility_filter.csv"
OUTPUT_FILE = f"{DATE_PREFIX}_assignment_optimisation.csv"


# ── Solvers ──────────────────────────────────────────────────────────────────

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


# ── Global ε-constraint solver ───────────────────────────────────────────────

LEX_FUEL_TOL_KG = 1e-6
LEX_COST_TOL_USD = 1e-5


def _solve_pyomo_model(model):
    solver = SolverFactory("appsi_highs")
    solver.options["output_flag"] = False
    solver.options["mip_rel_gap"] = 0
    solver.options["time_limit"] = 300
    result = solver.solve(model)
    if (result.solver.status != SolverStatus.ok or
            result.solver.termination_condition != TerminationCondition.optimal):
        raise RuntimeError(
            f"Global epsilon solver failed: status={result.solver.status}, "
            f"termination={result.solver.termination_condition}"
        )


def pyomo_global_fuel_constrained_assignment(group_problems, variable_cost_budget):
    """Lexicographically minimise fuel, cost, then swaps under one daily budget."""
    arcs = [
        (g, i, j)
        for g, problem in enumerate(group_problems)
        for i in range(problem["n"])
        for j in range(problem["n"])
        if (i, j) not in problem["forbidden_pairs"]
    ]

    model = ConcreteModel()
    model.arcs = Set(dimen=3, initialize=arcs)
    model.x = Var(model.arcs, domain=Binary)
    model.assignments = ConstraintList()
    for g, problem in enumerate(group_problems):
        n = problem["n"]
        forbidden = problem["forbidden_pairs"]
        for i in range(n):
            model.assignments.add(
                sum(model.x[g, i, j] for j in range(n) if (i, j) not in forbidden) == 1
            )
        for j in range(n):
            model.assignments.add(
                sum(model.x[g, i, j] for i in range(n) if (i, j) not in forbidden) == 1
            )

    fuel_expr = sum(
        float(group_problems[g]["fuel_matrix_kg"][i, j]) * model.x[g, i, j]
        for g, i, j in arcs
    )
    cost_expr = sum(
        float(group_problems[g]["cost_matrix"][i, j]) * model.x[g, i, j]
        for g, i, j in arcs
    )
    swap_expr = sum(model.x[g, i, j] for g, i, j in arcs if i != j)

    model.budget = Constraint(expr=cost_expr <= variable_cost_budget + LEX_COST_TOL_USD)
    model.objective = Objective(expr=fuel_expr, sense=minimize)
    _solve_pyomo_model(model)
    fuel_star = value(fuel_expr)

    model.objective.deactivate()
    model.fuel_tie = Constraint(expr=fuel_expr <= fuel_star + LEX_FUEL_TOL_KG)
    model.cost_objective = Objective(expr=cost_expr, sense=minimize)
    _solve_pyomo_model(model)
    cost_star = value(cost_expr)

    model.cost_objective.deactivate()
    model.cost_tie = Constraint(expr=cost_expr <= cost_star + LEX_COST_TOL_USD)
    model.swap_objective = Objective(expr=swap_expr, sense=minimize)
    _solve_pyomo_model(model)

    assignments = {}
    for g, problem in enumerate(group_problems):
        selected = np.empty(problem["n"], dtype=int)
        for i in range(problem["n"]):
            matches = [
                j for j in range(problem["n"])
                if (g, i, j) in model.x and value(model.x[g, i, j]) > 0.5
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Global solver returned invalid assignment for group {g}, row {i}")
            selected[i] = matches[0]
        assignments[g] = selected
    return assignments


def gurobi_cm_global_fuel_constrained_assignment(group_problems, variable_cost_budget):
    """Gurobi Cluster Manager equivalent of the global lexicographic solve."""
    import gurobipy as gp
    from gurobipy import GRB

    env = _gurobi_cm_env()
    with gp.Model(env=env) as model:
        model.setParam("MIPGap", 0)
        model.setParam("TimeLimit", 300)
        arcs = [
            (g, i, j)
            for g, problem in enumerate(group_problems)
            for i in range(problem["n"])
            for j in range(problem["n"])
            if (i, j) not in problem["forbidden_pairs"]
        ]
        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        for g, problem in enumerate(group_problems):
            n = problem["n"]
            forbidden = problem["forbidden_pairs"]
            for i in range(n):
                model.addConstr(gp.quicksum(x[g, i, j] for j in range(n) if (i, j) not in forbidden) == 1)
            for j in range(n):
                model.addConstr(gp.quicksum(x[g, i, j] for i in range(n) if (i, j) not in forbidden) == 1)

        fuel_expr = gp.quicksum(
            float(group_problems[g]["fuel_matrix_kg"][i, j]) * x[g, i, j]
            for g, i, j in arcs
        )
        cost_expr = gp.quicksum(
            float(group_problems[g]["cost_matrix"][i, j]) * x[g, i, j]
            for g, i, j in arcs
        )
        swap_expr = gp.quicksum(x[g, i, j] for g, i, j in arcs if i != j)
        model.addConstr(cost_expr <= variable_cost_budget + LEX_COST_TOL_USD)

        model.setObjective(fuel_expr, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Global Gurobi fuel solve failed: status={model.Status}")
        fuel_star = fuel_expr.getValue()

        model.addConstr(fuel_expr <= fuel_star + LEX_FUEL_TOL_KG)
        model.setObjective(cost_expr, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Global Gurobi cost tie-break failed: status={model.Status}")
        cost_star = cost_expr.getValue()

        model.addConstr(cost_expr <= cost_star + LEX_COST_TOL_USD)
        model.setObjective(swap_expr, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Global Gurobi swap tie-break failed: status={model.Status}")

        assignments = {}
        for g, problem in enumerate(group_problems):
            selected = np.empty(problem["n"], dtype=int)
            for i in range(problem["n"]):
                matches = [j for j in range(problem["n"]) if (g, i, j) in x and x[g, i, j].X > 0.5]
                if len(matches) != 1:
                    raise RuntimeError(f"Global Gurobi returned invalid assignment for group {g}, row {i}")
                selected[i] = matches[0]
            assignments[g] = selected
        return assignments


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


def _build_group_problem(group_df, group_index, forbidden_pairs):
    aircraft_perf_corr = group_df["avg_perf_corr"].to_numpy()
    aircraft_fh_rate = group_df["total_fh_rate"].to_numpy()
    aircraft_cycle_rate = group_df["total_cycle_rate"].to_numpy()
    route_fuel = group_df["total_baseline_fuel"].to_numpy()
    route_hours = group_df["total_pred_act_hours"].to_numpy()
    route_sectors = group_df["total_sectors"].to_numpy()
    fuel_matrix_kg = np.outer(aircraft_perf_corr, route_fuel)
    cost_matrix = (
        fuel_matrix_kg * FUEL_PRICE
        + np.outer(aircraft_fh_rate, route_hours)
        + np.outer(aircraft_cycle_rate, route_sectors)
    )
    return {
        "group_df": group_df,
        "group_index": group_index,
        "n": len(group_df),
        "forbidden_pairs": set(forbidden_pairs),
        "fuel_matrix_kg": fuel_matrix_kg,
        "cost_matrix": cost_matrix,
    }


def _rows_for_assignment(problem, selected_routes, epsilon):
    """Convert an aircraft->route permutation into the established row schema."""
    group_df = problem["group_df"]
    n = problem["n"]
    reverse_assignment = np.empty(n, dtype=int)
    reverse_assignment[selected_routes] = np.arange(n)

    aircraftreg = group_df["aircraftreg"].to_list()
    base = group_df["base"].to_list()
    perf_type = group_df["perf_type"].to_list()
    mtow = group_df["mtow"].to_list()
    perf_corr = group_df["avg_perf_corr"].to_list()
    seat_config = group_df["seat_config"].to_list()
    trip_fuel = group_df["total_trip_fuel"].to_list()
    deg_burn = group_df["total_deg_burn"].to_list()
    flt_date = group_df["flt_date"].to_list()
    route_fuel = group_df["total_baseline_fuel"].to_list()
    route_hours = group_df["total_pred_act_hours"].to_list()
    route_sectors = group_df["total_sectors"].to_list()
    fh_rate = group_df["total_fh_rate"].to_list()
    cycle_rate = group_df["total_cycle_rate"].to_list()
    route_in_df = "Route" in group_df.columns
    routes = group_df["Route"].to_list() if route_in_df else None
    flight_numbers = group_df["flt_numbers"].to_list() if route_in_df else None
    cost_matrix = problem["cost_matrix"]

    rows = []
    for route_idx in range(n):
        aircraft_idx = int(reverse_assignment[route_idx])
        opti_fuel_used = route_fuel[route_idx] * perf_corr[aircraft_idx]
        row = {
            "epsilon": epsilon,
            "group_index": problem["group_index"],
            "flt_date": flt_date[route_idx],
            "aircraftreg": aircraftreg[route_idx],
            "base": base[route_idx],
            "perf_type": perf_type[route_idx],
            "mtow": mtow[route_idx],
            "avg_perf_corr": perf_corr[route_idx],
            "seat_config": seat_config[route_idx],
            "total_trip_fuel": trip_fuel[route_idx],
            "total_deg_burn": deg_burn[route_idx],
            "total_baseline_fuel": route_fuel[route_idx],
            "total_pred_act_hours": route_hours[route_idx],
            "total_sectors": route_sectors[route_idx],
            "total_fh_rate": fh_rate[route_idx],
            "total_cycle_rate": cycle_rate[route_idx],
            "opti_aircraftreg": aircraftreg[aircraft_idx],
            "opti_aircraftreg_avg_perf_corr": perf_corr[aircraft_idx],
            "opti_seat_config": seat_config[aircraft_idx],
            "opti_total_baseline_fuel": route_fuel[route_idx],
            "opti_fh_rate": fh_rate[aircraft_idx],
            "opti_cycle_rate": cycle_rate[aircraft_idx],
            "opti_fuel_used": opti_fuel_used,
            "fuel_delta": trip_fuel[route_idx] - opti_fuel_used,
            "current_cost": float(cost_matrix[route_idx, route_idx]),
            "opti_cost": float(cost_matrix[aircraft_idx, route_idx]),
            "savings": float(cost_matrix[route_idx, route_idx] - cost_matrix[aircraft_idx, route_idx]),
            "changed": route_idx != aircraft_idx,
        }
        if route_in_df:
            row["Route"] = routes[route_idx]
            row["flt_numbers"] = flight_numbers[route_idx]
        rows.append(row)
    return rows


def optimize_global_eps(group_problems, fixed_rows, eps_values, total_cost_star):
    """Solve every epsilon with one global daily cost budget."""
    fixed_cost = sum(row["opti_cost"] for row in fixed_rows)
    results = []
    for eps in sorted(set(float(e) for e in eps_values)):
        total_budget = float(total_cost_star) * (1.0 + eps)
        variable_budget = total_budget - fixed_cost
        if SOLVER_BACKEND == "gurobi":
            assignments = _gurobi_call_with_retry(
                gurobi_cm_global_fuel_constrained_assignment,
                group_problems,
                variable_budget,
            )
        else:
            assignments = pyomo_global_fuel_constrained_assignment(
                group_problems,
                variable_budget,
            )

        rows = []
        for group_number, problem in enumerate(group_problems):
            rows.extend(_rows_for_assignment(problem, assignments[group_number], eps))
        rows.extend({**row, "epsilon": eps} for row in fixed_rows)
        actual_cost = sum(row["opti_cost"] for row in rows)
        if actual_cost > total_budget + 0.02:
            raise RuntimeError(
                f"epsilon={eps} violates global budget: ${actual_cost:.6f} > ${total_budget:.6f}"
            )
        results.append((eps, rows, total_budget))
        print(
            f"  global eps={eps:.6f}: cost=${actual_cost:,.2f} / "
            f"budget=${total_budget:,.2f}",
            flush=True,
        )
    return results


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
    epsilon_budgets = {}
    group_problems = []
    fixed_rows = []
    skipped_groups = []
    total_current_cost  = 0.0
    total_optimal_cost  = 0.0
    total_savings       = 0.0
    groups_processed    = 0
    base_group_counters = {}
    if any(float(eps) < 0 for eps in EPS_VALUES):
        raise ValueError("EPS_VALUES must be non-negative")
    # A true cost-optimal reference is required for budgets, deltas and
    # dominance checks even if the user omitted 0 from the configured grid.
    sweep_eps_values = sorted({0.0, *(float(eps) for eps in EPS_VALUES)}) if EPS_VALUES else []

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
                fixed_rows.extend(group_results)
                skipped_groups.append({
                    "group_index": group_index,
                    "reason": "infeasible constraints — route has no eligible aircraft",
                })
            elif sweep_eps_values:
                group_problems.append(
                    _build_group_problem(group, group_index, forbidden_pairs)
                )

        if groups_processed % 10 == 0 and groups_processed:
            print(f"  Processed {groups_processed}/{len(partitions)} groups...")

    print(f"\nCompleted {groups_processed} groups")
    print(f"  Total current cost : ${total_current_cost:,.2f}")
    print(f"  Total optimal cost : ${total_optimal_cost:,.2f}")
    print(f"  Total savings      : ${total_savings:,.2f}")

    if skipped_groups:
        names = [g["group_index"] for g in skipped_groups]
        print(f"WARNING: {len(skipped_groups)} group(s) skipped (infeasible): {names}")

    if sweep_eps_values:
        if group_problems:
            print(
                f"\nSolving global epsilon sweep across {len(group_problems)} variable "
                f"groups ({len(fixed_rows)} fixed rows)..."
            )
            global_sweep = optimize_global_eps(
                group_problems,
                fixed_rows,
                sweep_eps_values,
                total_optimal_cost,
            )
        else:
            global_sweep = [
                (eps, [{**row, "epsilon": eps} for row in fixed_rows], total_optimal_cost * (1 + eps))
                for eps in sweep_eps_values
            ]
        for eps, rows, total_budget in global_sweep:
            all_eps_rows.extend(rows)
            epsilon_budgets[eps] = total_budget

    results_df = pl.DataFrame(all_results)

    out_path = f"{INTERMEDIATE_DIRECTORY}/{OUTPUT_FILE}"
    results_df.write_csv(out_path)
    print(f"\nResults written : {out_path}")

    skipped_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_skipped_groups.json"
    with open(skipped_path, "w") as f:
        json.dump(skipped_groups, f, indent=2)

    # ── ε-sweep outputs ───────────────────────────────────────────────────────
    if sweep_eps_values and all_eps_rows:
        eps_df = pl.DataFrame(all_eps_rows)
        eps_csv_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_results.csv"
        eps_df.write_csv(eps_csv_path)
        print(f"eps-sweep results : {eps_csv_path}")

        rows_by_eps = {
            eps: [row for row in all_eps_rows if abs(float(row["epsilon"]) - eps) < 1e-12]
            for eps in sweep_eps_values
        }
        expected_groups = {row["group_index"] for row in all_results}
        expected_row_count = len(all_results)
        for eps, rows in rows_by_eps.items():
            if len(rows) != expected_row_count or {row["group_index"] for row in rows} != expected_groups:
                raise RuntimeError(
                    f"Invalid epsilon sweep coverage at eps={eps}: expected "
                    f"{expected_row_count} rows/{len(expected_groups)} groups, got "
                    f"{len(rows)} rows/{len({row['group_index'] for row in rows})} groups"
                )
        exact_points = []
        sweep_summary = {}
        for eps, rows in rows_by_eps.items():
            total_cost = sum(row["opti_cost"] for row in rows)
            total_fuel = sum(row["opti_fuel_used"] for row in rows)
            baseline_kg = sum(row["total_trip_fuel"] for row in rows)
            exact_points.append((eps, total_cost, total_fuel))
            sweep_summary[str(eps)] = {
                "total_opti_cost": round(total_cost, 2),
                "total_fuel_cost_kg": round(total_fuel, 2),
                "total_fuel_cost_usd": round(total_fuel * FUEL_PRICE, 2),
                "n_changed": sum(1 for row in rows if row["changed"]),
                "n_groups": len({row["group_index"] for row in rows}),
                "baseline_fuel_kg": round(baseline_kg, 2),
                "baseline_fuel_usd": round(baseline_kg * FUEL_PRICE, 2),
                "global_cost_budget": round(epsilon_budgets[eps], 2),
                "budget_slack_usd": round(epsilon_budgets[eps] - total_cost, 2),
                "epsilon_formulation": "global_daily_budget_v2",
                "lexicographic_objectives": ["fuel_kg", "total_cost_usd", "changed_assignments"],
            }

        # Nested global budgets imply non-increasing optimum fuel.  With the
        # cost tie-break, realised cost is non-decreasing as well.  A violation
        # means the solver result is not a trustworthy frontier sample.
        for previous, current in zip(exact_points, exact_points[1:]):
            if current[2] > previous[2] + 0.01:
                raise RuntimeError(
                    f"Invalid epsilon sweep: fuel increased from eps={previous[0]} to eps={current[0]}"
                )
            if current[1] < previous[1] - 0.01:
                raise RuntimeError(
                    f"Invalid epsilon sweep: realised cost decreased from eps={previous[0]} to eps={current[0]}"
                )

        # ── Select only from validated, actually solved objective points ─────
        selection = select_frontier_point(
            sweep_summary,
            max_cost_per_fuel_kg=EPS_MAX_COST_PER_FUEL_KG,
            min_prominence=EPS_KNEE_MIN_PROMINENCE,
        )
        applied_eps = None
        if AUTO_SELECT_EPS:
            if AUTO_SELECT_EPS_METHOD != "frontier":
                print(
                    f"WARNING: legacy AUTO_SELECT_EPS_METHOD={AUTO_SELECT_EPS_METHOD!r} "
                    "is retired; using validated actual-cost frontier selection"
                )
            applied_eps = selection["selected_epsilon"]
            if applied_eps is None:
                print(
                    "No defensible interior epsilon point found; keeping the cost-optimal assignment. "
                    f"Reason: {selection['selection_reason']}"
                )
            else:
                print(
                    f"Validated frontier selected eps={applied_eps:.6f} "
                    f"({applied_eps * 100:.4f}%): {selection['selection_reason']}"
                )
        elif SELECTED_EPS is not None:
            matching = [eps for eps in rows_by_eps if abs(eps - float(SELECTED_EPS)) < 1e-12]
            if not matching:
                print(
                    f"WARNING: SELECTED_EPS={SELECTED_EPS} is not in EPS_VALUES "
                    "— keeping standard cost-optimal result"
                )
                selection["selected_epsilon"] = None
                selection["selection_rule"] = "cost_optimal"
                selection["selection_reason"] = "manual epsilon was not present in the solved grid"
            else:
                applied_eps = matching[0]
                selection["selected_epsilon"] = applied_eps
                selection["selection_rule"] = "manual"
                point_status = selection["frontier_status"][applied_eps]
                if point_status["is_dominated"] or point_status["is_duplicate"]:
                    selection["selection_reason"] = (
                        "manual epsilon override; warning: selected objective point is dominated or duplicate"
                    )
                    print(f"WARNING: manual eps={applied_eps} is dominated or duplicate")
                else:
                    selection["selection_reason"] = "manual epsilon override"
        else:
            selection["selected_epsilon"] = None
            selection["selection_rule"] = "cost_optimal"
            selection["selection_reason"] = "automatic selection disabled; no manual epsilon supplied"

        annotate_summary(sweep_summary, selection)

        summary_path = f"{INTERMEDIATE_DIRECTORY}/{DATE_PREFIX}_eps_sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(sweep_summary, f, indent=2)
        print(f"eps-sweep summary : {summary_path}")

        if applied_eps is not None:
            chosen_rows = [
                {k: v for k, v in r.items() if k != "epsilon"}
                for r in all_eps_rows
                if r["epsilon"] == applied_eps
            ]
            if chosen_rows:
                pl.DataFrame(chosen_rows).write_csv(out_path)
                print(f"Applied eps={applied_eps}: {OUTPUT_FILE} now reflects eps solution")

        print(
            f"eps-sweep selection updated: method={selection['selection_rule']}, "
            f"applied={selection['selected_epsilon']}"
        )

    print(f"\nColumns : {results_df.columns}")
    print("\nStage 5 Assignment Optimisation complete.")


if __name__ == "__main__":
    main()
