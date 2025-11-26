# KMRL – AI-Driven Nightly Train Induction Planner (MVP)

This project automatically assigns Metro trainsets to Service, Standby, or IBL roles using:

- CSV ingestion
- Constraint validation (certificates, blocks, cleaning)
- Linear cost model (readiness, mileage, shunt)
- Hungarian Algorithm (minimum-cost assignment)
- Detailed CSV output + visualizer frontend

## How to run

1. Add CSVs inside `/data/`

2. Run the full pipeline:

