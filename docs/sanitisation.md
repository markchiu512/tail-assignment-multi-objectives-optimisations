# Public portfolio scope

This repository is a sanitised technical portfolio version of an aviation
decision-support system. It retains the implementation approach, optimisation
formulation, and software architecture while excluding operational assets.

## Included

- Streamlit operational interface and six-stage pipeline structure.
- Pooled-model inference contract, constrained assignment model, and solver
  integration patterns.
- Generic warehouse-query templates and environment-driven configuration.
- Rotation-level research formulation and source code.

## Deliberately excluded

- Production schedules, aircraft registrations, cost rates, ground-event data,
  approved swap lists, and actual operating outcomes.
- Trained model binaries, credentials, workspace and warehouse identifiers,
  deployment exports, user details, and internal reports.
- Real rotation-research inputs, schedules, reports, and visualisations.

To run a deployment, provide approved data, model artefacts, and environment
values through the interfaces documented in `.env.example`.
