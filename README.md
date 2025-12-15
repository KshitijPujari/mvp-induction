# KMRL – AI-Driven Nightly Train Induction Planner (MVP)

This project automatically assigns Metro trainsets to Service, Standby, or IBL roles using:

- CSV ingestion
- Constraint validation (certificates, blocks, cleaning)
- Linear cost model (readiness, mileage, shunt)
- Hungarian Algorithm (minimum-cost assignment)
- Detailed CSV output + visualizer frontend

## How to run

### Backend Pipeline (Command Line)

1. Add CSVs inside `/data/`

2. Run the full pipeline:
   ```bash
   python -m src.solver
   ```

### Interactive Frontend

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask backend server:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

4. In the interactive frontend:
   - Select trains/metros using the checkboxes
   - Click "Find Optimal Path" to get optimal assignments
   - View the visual path diagram and detailed assignment table
   - See cost breakdowns and feasibility information

