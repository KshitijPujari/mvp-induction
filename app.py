"""
Flask API backend for KMRL Induction Planner
Exposes solver logic for interactive frontend
"""

import os
import sys
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src import ingestion, solver, cost_model, constraints

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(ROOT, "data")


@app.route('/')
def index():
    """Serve the main frontend"""
    return send_from_directory('frontend', 'index.html')


@app.route('/api/trains', methods=['GET'])
def get_trains():
    """Get list of all available trains with their details"""
    try:
        data = ingestion.load_all()
        trains = []
        
        for train_id in data["job_card"]["train_id"].unique():
            job_row = data["job_card"].loc[data["job_card"]["train_id"] == train_id].iloc[0]
            cert_row = data["certificate_validity"].loc[data["certificate_validity"]["train_id"] == train_id].iloc[0]
            clean_row = data["cleaning_roster"].loc[data["cleaning_roster"]["train_id"] == train_id].iloc[0]
            mileage_row = data["mileage"].loc[data["mileage"]["train_id"] == train_id].iloc[0]
            
            trains.append({
                "train_id": train_id,
                "blocked": bool(job_row.get("blocked", 0) == 1),
                "certificates_valid": all(cert_row.get(c, 0) == 1 for c in ["brake_test_valid", "pto_valid", "atp_valid"]),
                "cleaning_done": bool(clean_row.get("cleaning_done", 1) == 1),
                "mileage_km": float(mileage_row.get("mileage_km", 0)),
                "readiness_score": cost_model.readiness_risk(train_id, data["job_card"], data["certificate_validity"], data["cleaning_roster"]),
                "mileage_score": cost_model.mileage_penalty(train_id, data["mileage"])
            })
        
        return jsonify({"trains": trains})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/slots', methods=['GET'])
def get_slots():
    """Get list of all available induction slots"""
    try:
        data = ingestion.load_all()
        slots = []
        
        for _, row in data["induction_slots"].iterrows():
            slots.append({
                "slot_id": str(row["slot_id"]),
                "role": row.get("role", ""),
                "shunt_distance": float(row.get("shunt_distance", 0))
            })
        
        return jsonify({"slots": slots})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/solve', methods=['POST'])
def solve_optimal():
    """Solve optimal assignment for selected trains"""
    try:
        req_data = request.get_json()
        selected_trains = req_data.get("train_ids", [])
        weights = req_data.get("weights", None)
        
        if not selected_trains:
            return jsonify({"error": "No trains selected"}), 400
        
        # Load data
        data = ingestion.load_all()
        ingestion.validate_basic_schema(data)
        
        # Filter to selected trains only
        original_job_card = data["job_card"]
        data["job_card"] = original_job_card[original_job_card["train_id"].isin(selected_trains)]
        
        if data["job_card"].empty:
            return jsonify({"error": "No valid trains found"}), 400
        
        # Solve
        assignments = solver.solve_and_extract(data, weights=weights, debug=False)
        
        # Build detailed response
        results = []
        for _, row in assignments.iterrows():
            train_id = row["train_id"]
            assigned_slot = row["assigned_slot"]
            role = row["role"]
            cost = row["cost"]
            
            r_score = cost_model.readiness_risk(train_id, data["job_card"], data["certificate_validity"], data["cleaning_roster"])
            m_score = cost_model.mileage_penalty(train_id, data["mileage"])
            s_score = 0.0
            
            slot_info = None
            if assigned_slot is not None and pd.notna(assigned_slot):
                slot_row = data["induction_slots"].loc[data["induction_slots"]["slot_id"] == assigned_slot].iloc[0]
                s_score = cost_model.shunt_proxy(slot_row)
                slot_info = {
                    "slot_id": str(assigned_slot),
                    "role": role,
                    "shunt_distance": float(slot_row.get("shunt_distance", 0))
                }
            
            w = weights if weights else cost_model.DEFAULT_WEIGHTS
            total_cost = w["w_readiness"] * r_score + w["w_mileage"] * m_score + w["w_shunt"] * s_score
            
            # Get infeasibility reasons if unassigned
            infeas_reasons = []
            if assigned_slot is None:
                from src.explain import get_infeasibility_reasons
                for _, srow in data["induction_slots"].iterrows():
                    if not constraints.is_pair_feasible(train_id, data["job_card"], data["certificate_validity"], 
                                                       data["cleaning_roster"], srow):
                        reasons_list = get_infeasibility_reasons(train_id, srow, data)
                        if reasons_list:
                            infeas_reasons.extend([f"{r} ({srow['slot_id']})" for r in reasons_list])
                # Deduplicate
                infeas_reasons = list(dict.fromkeys(infeas_reasons))
            
            results.append({
                "train_id": train_id,
                "assigned_slot": assigned_slot,
                "role": role,
                "slot_info": slot_info,
                "total_cost": round(total_cost, 2) if cost else None,
                "readiness_score": r_score,
                "mileage_score": round(m_score, 2),
                "shunt_score": round(s_score, 2),
                "explanation": row.get("explanation", ""),
                "infeasibility_reasons": infeas_reasons,
                "is_assigned": assigned_slot is not None and pd.notna(assigned_slot)
            })
        
        # Calculate summary stats
        assigned_count = sum(1 for r in results if r["is_assigned"])
        total_cost_sum = sum(r["total_cost"] for r in results if r["total_cost"] is not None)
        
        return jsonify({
            "assignments": results,
            "summary": {
                "total_trains": len(results),
                "assigned": assigned_count,
                "unassigned": len(results) - assigned_count,
                "total_cost": round(total_cost_sum, 2),
                "avg_cost": round(total_cost_sum / assigned_count, 2) if assigned_count > 0 else 0
            }
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/cost-matrix', methods=['POST'])
def get_cost_matrix():
    """Get cost matrix for selected trains"""
    try:
        req_data = request.get_json()
        selected_trains = req_data.get("train_ids", [])
        weights = req_data.get("weights", None)
        
        if not selected_trains:
            return jsonify({"error": "No trains selected"}), 400
        
        # Load and filter data
        data = ingestion.load_all()
        original_job_card = data["job_card"]
        data["job_card"] = original_job_card[original_job_card["train_id"].isin(selected_trains)]
        
        # Build cost matrix
        cost_df = cost_model.build_cost_matrix(data, weights=weights)
        
        # Convert to JSON-friendly format
        matrix = {
            "trains": list(cost_df.index),
            "slots": list(cost_df.columns),
            "costs": {}
        }
        
        for train_id in cost_df.index:
            matrix["costs"][train_id] = {}
            for slot_id in cost_df.columns:
                cost_val = float(cost_df.at[train_id, slot_id])
                matrix["costs"][train_id][slot_id] = cost_val if cost_val < cost_model.INF / 10 else None
        
        return jsonify(matrix)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

