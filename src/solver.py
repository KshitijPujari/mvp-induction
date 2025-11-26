"""
Solver for KMRL Induction MVP (updated)


- Uses Hungarian algorithm (linear_sum_assignment) to minimize total cost.
- Pads the cost matrix to square with INF so solver can always run.
- Produces two CSV outputs:
    1) simple final_plan_YYYYMMDD_HHMMSS.csv (train_id, assigned_slot, role, cost, explanation)
    2) detailed final_plan_detailed_YYYYMMDD_HHMMSS.csv (includes readiness/mileage/shunt breakdown & infeasibility reasons)
- Relies on src.explain.explain_all() to produce the detailed breakdown (no duplicate logic).
"""


import os
import datetime
import numpy as np
import pandas as pd


try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from e


# local imports with fallback
try:
    from src import ingestion, cost_model, constraints, explain
except Exception:
    import ingestion, cost_model, constraints, explain


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)



def pad_to_square(matrix_df, pad_value=cost_model.INF):
    rows = list(matrix_df.index.astype(str))
    cols = list(matrix_df.columns.astype(str))
    n = max(len(rows), len(cols))
    square = np.full((n, n), fill_value=pad_value, dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            square[i, j] = float(matrix_df.at[r, c])
    return square, rows, cols



def solve_and_extract(data, weights=None, debug=False):
    """
    Build cost matrix, pad, solve, and return a DataFrame of assignments.
    The returned DataFrame has columns:
      train_id, assigned_slot, role, cost, explanation
    """
    cost_df = cost_model.build_cost_matrix(data, weights=weights)
    if debug:
        print("Cost DataFrame:")
        print(cost_df)

    square, trains, slots = pad_to_square(cost_df, pad_value=cost_model.INF)
    row_ind, col_ind = linear_sum_assignment(square)

    assignments = []
    n_trains = len(trains)
    n_slots = len(slots)

    for r, c in zip(row_ind, col_ind):
        if r < n_trains and c < n_slots:
            train_id = trains[r]
            slot_id = slots[c]
            cost_val = float(square[r, c])
            if cost_val >= cost_model.INF / 10.0:
                assignments.append({
                    "train_id": train_id,
                    "assigned_slot": None,
                    "role": None,
                    "cost": None,
                    "explanation": "No feasible slot"
                })
            else:
                slot_row = data["induction_slots"].loc[data["induction_slots"]["slot_id"] == slot_id].iloc[0]
                explanation = cost_model.explain_reason(train_id, slot_row, data)
                assignments.append({
                    "train_id": train_id,
                    "assigned_slot": slot_id,
                    "role": slot_row.get("role"),
                    "cost": cost_val,
                    "explanation": explanation
                })
        elif r < n_trains and c >= n_slots:
            train_id = trains[r]
            assignments.append({
                "train_id": train_id,
                "assigned_slot": None,
                "role": None,
                "cost": None,
                "explanation": "Assigned to dummy (no available real slot)"
            })
        # ignore rows >= n_trains (dummy trains)

    out_df = pd.DataFrame(assignments)
    # ensure all trains appear in original job_card order
    out_df = out_df.set_index("train_id").reindex(list(data["job_card"]["train_id"])).reset_index()
    return out_df



def write_csv_simple(df, ts):
    fname = f"final_plan_{ts}.csv"
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False)
    return path



def write_csv_detailed(detailed_df, ts):
    fname = f"final_plan_detailed_{ts}.csv"
    path = os.path.join(OUTPUT_DIR, fname)
    detailed_df.to_csv(path, index=False)
    return path



def run_pipeline(weights=None, debug=False):
    """
    Load data, run solver, produce both simple and detailed outputs.
    """
    data = ingestion.load_all()
    ingestion.validate_basic_schema(data)

    # 1) Solve (simple assignments)
    assignments = solve_and_extract(data, weights=weights, debug=debug)

    # 2) Produce detailed breakdown using src.explain
    # explain.explain_all returns a DataFrame and can optionally save; we call it with save_csv=False
    try:
        detailed_df = explain.explain_all(weights=weights, save_csv=False)
    except Exception:
        # Fallback: build a minimal detailed DF from assignments + cost components
        detailed_rows = []
        for _, row in assignments.iterrows():
            train_id = row["train_id"]
            assigned_slot = row["assigned_slot"]
            role = row["role"]
            r_score = cost_model.readiness_risk(train_id, data["job_card"], data["certificate_validity"], data["cleaning_roster"])
            m_score = cost_model.mileage_penalty(train_id, data["mileage"])
            s_score = 0.0
            if assigned_slot is not None and pd.notna(assigned_slot):
                slot_row = data["induction_slots"].loc[data["induction_slots"]["slot_id"] == assigned_slot].iloc[0]
                s_score = cost_model.shunt_proxy(slot_row)
            total_cost = (weights or cost_model.DEFAULT_WEIGHTS)["w_readiness"] * r_score + \
                         (weights or cost_model.DEFAULT_WEIGHTS)["w_mileage"] * m_score + \
                         (weights or cost_model.DEFAULT_WEIGHTS)["w_shunt"] * s_score
            detailed_rows.append({
                "train_id": train_id,
                "assigned_slot": assigned_slot,
                "role": role,
                "total_cost_recomputed": round(total_cost, 4),
                "readiness_score": r_score,
                "mileage_score": round(m_score, 4),
                "shunt_score": round(s_score, 4),
                "reasons": cost_model.explain_reason(train_id, data["induction_slots"].iloc[0], data)
            })
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df = detailed_df.set_index("train_id").reindex(list(data["job_card"]["train_id"])).reset_index()

    # 3) Save both outputs with same timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    simple_path = write_csv_simple(assignments, ts)
    detailed_path = write_csv_detailed(detailed_df, ts)

    # 4) Print short summaries
    print("\nFinal simple assignments (first 20 rows):")
    print(assignments.head(20).to_string(index=False))

    print(f"\nSaved simple plan to: {simple_path}")
    print(f"Saved detailed plan to: {detailed_path}")

    return assignments, detailed_df, simple_path, detailed_path




if __name__ == "__main__":
    run_pipeline(debug=False)
