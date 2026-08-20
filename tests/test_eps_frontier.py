import unittest

from src.eps_frontier import classify_frontier, select_frontier_point


def summary(points):
    """Build sweep-summary rows from (epsilon, cost delta, fuel saving) tuples."""
    return {
        str(epsilon): {
            "total_opti_cost": 100.0 + cost_delta,
            "total_fuel_cost_kg": 1000.0 - fuel_saving,
        }
        for epsilon, cost_delta, fuel_saving in points
    }


class FrontierSelectionTests(unittest.TestCase):
    def test_dominated_and_duplicate_points_are_not_candidates(self):
        data = summary([
            (0.0, 0.0, 0.0),
            (0.1, 2.0, 2.0),
            (0.2, 1.5, 3.0),
            (0.3, 1.5, 3.0),
            (0.4, 3.0, 4.0),
        ])
        pareto, status = classify_frontier(data)
        self.assertTrue(status[0.1]["is_dominated"])
        self.assertTrue(status[0.3]["is_duplicate"])
        self.assertEqual([point["epsilon"] for point in pareto], [0.0, 0.2, 0.4])

    def test_linear_frontier_returns_no_knee(self):
        result = select_frontier_point(summary([
            (0.0, 0.0, 0.0),
            (0.1, 1.0, 10.0),
            (0.2, 2.0, 20.0),
            (0.3, 3.0, 30.0),
        ]))
        self.assertIsNone(result["selected_epsilon"])

    def test_convex_frontier_returns_no_knee(self):
        result = select_frontier_point(summary([
            (0.0, 0.0, 0.0),
            (0.1, 1.0, 5.0),
            (0.2, 2.0, 20.0),
            (0.3, 3.0, 50.0),
        ]))
        self.assertIsNone(result["selected_epsilon"])

    def test_concave_frontier_selects_an_actual_interior_vertex(self):
        data = summary([
            (0.0, 0.0, 0.0),
            (0.1, 1.0, 50.0),
            (0.2, 2.0, 80.0),
            (0.3, 3.0, 100.0),
        ])
        result = select_frontier_point(data)
        self.assertEqual(result["selected_epsilon"], 0.1)
        self.assertIn(str(result["selected_epsilon"]), data)

    def test_marginal_cost_limit_stops_before_expensive_tail(self):
        result = select_frontier_point(
            summary([
                (0.0, 0.0, 0.0),
                (0.1, 10.0, 10.0),
                (0.2, 30.0, 15.0),
            ]),
            max_cost_per_fuel_kg=2.0,
        )
        self.assertEqual(result["selected_epsilon"], 0.1)


if __name__ == "__main__":
    unittest.main()
