"""Validation and automatic selection for a discrete cost/fuel frontier.

The selector deliberately works on *realised total cost*, not epsilon.  Epsilon
is a budget parameter; it is not the cost paid by the selected assignment.
Only solved MIP points can be returned.
"""

from __future__ import annotations

from typing import Any


COST_TOL = 0.01
FUEL_TOL = 0.01
DEFAULT_MIN_PROMINENCE = 0.03


def _points(sweep_summary: dict[str, dict[str, Any]]) -> list[dict[str, float]]:
    if not sweep_summary:
        return []

    ordered = sorted(sweep_summary.items(), key=lambda item: float(item[0]))
    zero_key, zero = min(ordered, key=lambda item: abs(float(item[0])))
    del zero_key
    c0 = float(zero["total_opti_cost"])
    f0 = float(zero["total_fuel_cost_kg"])

    return [
        {
            "epsilon": float(key),
            "cost": float(row["total_opti_cost"]),
            "fuel_kg": float(row["total_fuel_cost_kg"]),
            "cost_delta": float(row["total_opti_cost"]) - c0,
            "fuel_saving_kg": f0 - float(row["total_fuel_cost_kg"]),
        }
        for key, row in ordered
    ]


def classify_frontier(
    sweep_summary: dict[str, dict[str, Any]],
    *,
    cost_tol: float = COST_TOL,
    fuel_tol: float = FUEL_TOL,
) -> tuple[list[dict[str, float]], dict[float, dict[str, bool]]]:
    """Return nondominated objective points and a status for every epsilon.

    Duplicate objective pairs are represented by their lowest-epsilon member.
    A point is dominated iff another solved point is no more costly and burns no
    more fuel, with a strict improvement beyond tolerance in at least one axis.
    """
    points = _points(sweep_summary)
    status: dict[float, dict[str, bool]] = {}
    representatives: list[dict[str, float]] = []

    for point in points:
        duplicate = any(
            abs(other["cost"] - point["cost"]) <= cost_tol
            and abs(other["fuel_kg"] - point["fuel_kg"]) <= fuel_tol
            for other in representatives
        )
        if not duplicate:
            representatives.append(point)
        status[point["epsilon"]] = {
            "is_duplicate": duplicate,
            "is_dominated": False,
            "is_pareto": False,
            "is_concave_hull": False,
        }

    for point in representatives:
        dominated = any(
            other is not point
            and other["cost"] <= point["cost"] + cost_tol
            and other["fuel_kg"] <= point["fuel_kg"] + fuel_tol
            and (
                other["cost"] < point["cost"] - cost_tol
                or other["fuel_kg"] < point["fuel_kg"] - fuel_tol
            )
            for other in representatives
        )
        status[point["epsilon"]]["is_dominated"] = dominated
        status[point["epsilon"]]["is_pareto"] = not dominated

    pareto = [
        point
        for point in representatives
        if status[point["epsilon"]]["is_pareto"]
    ]
    pareto.sort(key=lambda p: (p["cost"], p["fuel_kg"], p["epsilon"]))
    return pareto, status


def upper_concave_hull(points: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return the upper concave hull in (cost delta, fuel saving) coordinates."""
    if len(points) <= 2:
        return list(points)

    # For equal realised cost retain the point with the greatest fuel saving.
    by_x: list[dict[str, float]] = []
    for point in sorted(points, key=lambda p: (p["cost_delta"], -p["fuel_saving_kg"])):
        if by_x and abs(point["cost_delta"] - by_x[-1]["cost_delta"]) <= COST_TOL:
            if point["fuel_saving_kg"] > by_x[-1]["fuel_saving_kg"]:
                by_x[-1] = point
        else:
            by_x.append(point)

    hull: list[dict[str, float]] = []
    for point in by_x:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            dx1 = b["cost_delta"] - a["cost_delta"]
            dx2 = point["cost_delta"] - b["cost_delta"]
            dy1 = b["fuel_saving_kg"] - a["fuel_saving_kg"]
            dy2 = point["fuel_saving_kg"] - b["fuel_saving_kg"]
            # Concavity requires decreasing slopes.  Collinear points are not
            # vertices and are removed, avoiding a fabricated "knee".
            scale = max(1.0, abs(dy2 * dx1), abs(dy1 * dx2))
            if dy2 * dx1 >= dy1 * dx2 - 1e-12 * scale:
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def select_frontier_point(
    sweep_summary: dict[str, dict[str, Any]],
    *,
    max_cost_per_fuel_kg: float | None = None,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
) -> dict[str, Any]:
    """Select an actual solved point, or return no selection when no knee exists.

    If ``max_cost_per_fuel_kg`` is supplied, the rule is economically explicit:
    walk along the concave hull while the incremental dollars per kg saved do
    not exceed the limit.  Otherwise, use the maximum normalised above-chord
    distance on the *actual-cost* concave hull, subject to a prominence test.
    """
    pareto, status = classify_frontier(sweep_summary)
    hull = upper_concave_hull(pareto)
    for point in hull:
        status[point["epsilon"]]["is_concave_hull"] = True

    base_result: dict[str, Any] = {
        "selected_epsilon": None,
        "selection_rule": "marginal_cost_limit" if max_cost_per_fuel_kg is not None else "actual_cost_concave_hull",
        "selection_reason": "",
        "frontier_point_count": len(pareto),
        "concave_hull_point_count": len(hull),
        "frontier_status": status,
        "prominence": None,
        "incremental_cost_per_kg": None,
    }

    if not hull:
        base_result["selection_reason"] = "no solved frontier points"
        return base_result

    if max_cost_per_fuel_kg is not None:
        if max_cost_per_fuel_kg < 0:
            raise ValueError("max_cost_per_fuel_kg must be non-negative or None")
        selected = hull[0]
        selected_ratio = None
        for left, right in zip(hull, hull[1:]):
            fuel_gain = right["fuel_saving_kg"] - left["fuel_saving_kg"]
            cost_gain = right["cost_delta"] - left["cost_delta"]
            if fuel_gain <= FUEL_TOL:
                continue
            ratio = cost_gain / fuel_gain
            if ratio > max_cost_per_fuel_kg:
                break
            selected = right
            selected_ratio = ratio
        base_result.update(
            selected_epsilon=selected["epsilon"],
            selection_reason=(
                f"last concave-hull point whose incremental cost is at most "
                f"${max_cost_per_fuel_kg:g}/kg"
            ),
            incremental_cost_per_kg=selected_ratio,
        )
        return base_result

    if len(hull) < 3:
        base_result["selection_reason"] = (
            "no defensible interior knee: frontier is flat, linear, convex, or has fewer than three hull vertices"
        )
        return base_result

    x0, x1 = hull[0]["cost_delta"], hull[-1]["cost_delta"]
    y0, y1 = hull[0]["fuel_saving_kg"], hull[-1]["fuel_saving_kg"]
    if x1 - x0 <= COST_TOL or y1 - y0 <= FUEL_TOL:
        base_result["selection_reason"] = "no material cost/fuel range"
        return base_result

    candidates = []
    for point in hull[1:-1]:
        x_norm = (point["cost_delta"] - x0) / (x1 - x0)
        y_norm = (point["fuel_saving_kg"] - y0) / (y1 - y0)
        candidates.append((y_norm - x_norm, point))
    prominence, selected = max(candidates, key=lambda item: item[0])
    base_result["prominence"] = prominence
    if prominence < min_prominence:
        base_result["selection_reason"] = (
            f"no prominent knee: maximum normalised above-chord distance "
            f"{prominence:.4f} is below {min_prominence:.4f}"
        )
        return base_result

    base_result.update(
        selected_epsilon=selected["epsilon"],
        selection_reason="most prominent solved vertex on the actual-cost upper concave hull",
    )
    return base_result


def annotate_summary(
    sweep_summary: dict[str, dict[str, Any]], selection: dict[str, Any]
) -> None:
    """Add objective deltas, frontier flags, and selection metadata in place."""
    points = {point["epsilon"]: point for point in _points(sweep_summary)}
    status = selection["frontier_status"]
    for key, row in sweep_summary.items():
        eps = float(key)
        point = points[eps]
        row.update(
            actual_cost_delta=round(point["cost_delta"], 2),
            fuel_saving_vs_eps0_kg=round(point["fuel_saving_kg"], 2),
            **status[eps],
            auto_selected_eps=selection["selected_epsilon"],
            eps_star=selection["selected_epsilon"],
            auto_select_method=selection["selection_rule"],
            selection_reason=selection["selection_reason"],
            frontier_point_count=selection["frontier_point_count"],
            concave_hull_point_count=selection["concave_hull_point_count"],
            selection_prominence=(
                round(selection["prominence"], 6)
                if selection["prominence"] is not None
                else None
            ),
            selection_incremental_cost_per_kg=(
                round(selection["incremental_cost_per_kg"], 4)
                if selection["incremental_cost_per_kg"] is not None
                else None
            ),
        )
