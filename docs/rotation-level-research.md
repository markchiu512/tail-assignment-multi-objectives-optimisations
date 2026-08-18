# Rotation-level assignment research

The production optimiser assigns an aircraft tail to a complete Line of Flight.
The research branch explores a finer decision unit: a return rotation or, where
necessary, a fixed single leg.

It runs after leg-level fuel and airborne-hour prediction and does not alter
production assignment files.

## Additional feasibility model

For each candidate tail and rotation, the model requires:

- sufficient MTOW capability for every leg in the rotation;
- presence at the rotation origin with an available time window;
- a post-rotation turnaround buffer; and
- no temporal overlap with fixed events or another assigned rotation.

The model minimises the same three cost components as the production system:
fuel, flight-hour cost, and cycle cost. It is formulated as a binary MIP, with
one selected tail per rotation and no-overlap constraints for shared eligible
tails.

## Validation approach

The research scripts include four deliberate checks: forced status-quo
feasibility, LP relaxation, eligibility-exclusion attribution, and a final
integer-solution feasibility check. This makes the research useful as a
planning-verified shadow model rather than simply an unconstrained cost sweep.

The public repository intentionally contains no real rotations, registrations,
routes, schedules, or cost results. See `rotation_mip.py` and
`build_rotation_preopt.py` for the implementation.
