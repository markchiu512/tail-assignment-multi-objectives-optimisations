# Changelog

All notable public-repository changes are documented here.

## [2.1.0] - 2026-08-21

### Added

- Discrete cost/fuel frontier validation and selection in src/eps_frontier.py.
- Public configuration for a conservative frontier knee or an explicit marginal-cost-per-fuel-saved limit.

### Changed

- The ε sweep now uses one global daily cost budget across compatible assignment groups.
- Tied frontier solutions are resolved lexicographically: fuel, realised cost, then reassignment count.
- The Streamlit views now show realised cost/fuel trade-offs, Pareto status, and concave-hull status.

### Fixed

- Automatic ε selection now returns only an actually solved, non-dominated decision point, or retains the cost-optimal assignment when no defensible knee exists.

## [2.0.0] - 2026-08-18

### Added

- Initial sanitised portfolio edition of the airline tail-assignment optimisation system.
