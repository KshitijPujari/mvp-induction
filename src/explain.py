"""
Explain module for KMRL Induction MVP

Run as:
    python -m src.explain

What it does:
- Loads data (via ingestion)
- Runs solver.solve_and_extract to get assignments
- For each train (assigned slot or unassigned), computes:
    - readiness_score (0..5)
    - mileage_score (float)
    - shunt_score (float)
    - recomputed total_cost using cost_model.DEFAULT_WEIGHTS
    - structured ''reasons'' (top contributors)
    - ''infeasibility_reasons'' when no feasible slot
- Saves a detailed CSV to outputs/
- Prints a detailed summary
"""


import os
import datetime
import pandas as pd


try:
    from src import ingestion, cost_model, constraints, solver
except Exception:
    import ingestion, cost_model, constraints, solver


ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)



def get_infeasibility_reasons(train_id, slot_row, data):
    reasons = []

    job_df = data["job_card"]
    cert_df = data["certificate_validity"]
    clean_df = data["cleaning_roster"]

    job_row = job_df.loc[job_df["train_id"] == train_id]
    cert_row = cert_df.loc[cert_df["train_id"] == train_id]
    clean_row = clean_df.loc[clean_df["train_id"] == train_id]

    if job_row.empty:
        reasons.append("missing job-card data")
    if cert_row.empty:
        reasons.append("missing certificate data")
    if clean_row.empty:
        reasons.append("missing cleaning data")

    if slot_row is not None and not slot_row.empty:
        if not cert_row.empty:
            cert = cert_row.iloc[0]
            if not all(cert.get(c, 0) == 1 for c in ["brake_test_valid", "pto_valid", "atp_valid"]):
                reasons.append("certificate invalid")

        if not job_row.empty:
            job = job_row.iloc[0]
            if job.get("blocked", 0) == 1 and slot_row.get("role") == "Service":
                reasons.append("blocked train cannot go to Service")

        if not clean_row.empty:
            clean = clean_row.iloc[0]
            if clean.get("cleaning_done", 1) == 0 and slot_row.get("role") == "Service":
                reasons.append("cleaning pending for Service")

    return list(dict.fromkeys(reasons))



def top_reasons_from_breakdown(readiness, mileage, shunt):
    reasons = []

    if readiness >= 4:
        reasons.append("high readiness risk")
    elif readiness >= 1:
        reasons.append("readiness concern")

    if mileage >= 3:
        reasons.append("very high mileage")
    elif mileage >= 1:
        reasons.append("high mileage")

    if shunt >= 4:
        reasons.append("long shunt")

    if not reasons:
        reasons = ["good fit"]

    return reasons[:3]



def explain_all(weights=None, save_csv=True):
    data = ingestion.load_all()
    ingestion.validate_basic_schema(data)

    assignments = solver.solve_and_extract(data, weights=weights, debug=False)

    rows = []
    default_weights = weights if weights is not None else cost_model.DEFAULT_WEIGHTS

    for _, row in assignments.iterrows():
        train_id = row["train_id"]
        assigned_slot = row["assigned_slot"]
        role = row["role"]

        r_score = cost_model.readiness_risk(train_id, data["job_card"], data["certificate_validity"], data["cleaning_roster"])
        m_score = cost_model.mileage_penalty(train_id, data["mileage"])

        slot_row = None
        if assigned_slot is not None and not pd.isna(assigned_slot):
            slot_row = data["induction_slots"].loc[data["induction_slots"]["slot_id"] == assigned_slot].iloc[0]
            s_score = cost_model.shunt_proxy(slot_row)
        else:
            s_score = 0.0

        total_cost = (
            default_weights["w_readiness"] * r_score
            + default_weights["w_mileage"] * m_score
            + default_weights["w_shunt"] * s_score
        )

        reasons = top_reasons_from_breakdown(r_score, m_score, s_score)

        infeas_reasons = []
        if assigned_slot is None:
            slots = data["induction_slots"]
            reason_counts = {}
            for _, srow in slots.iterrows():
                feasible = constraints.is_pair_feasible(
                    train_id,
                    data["job_card"],
                    data["certificate_validity"],
                    data["cleaning_roster"],
                    srow
                )
                if not feasible:
                    fail_list = get_infeasibility_reasons(train_id, srow, data)
                    for r in fail_list:
                        reason_counts[r] = reason_counts.get(r, 0) + 1

            sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
            infeas_reasons = [f"{r} ({c} slots)" for r, c in sorted_reasons] or ["no feasible slot"]

        rows.append({
            "train_id": train_id,
            "assigned_slot": assigned_slot,
            "role": role,
            "total_cost_recomputed": round(total_cost, 4),
            "readiness_score": r_score,
            "mileage_score": round(m_score, 4),
            "shunt_score": round(s_score, 4),
            "reasons": "; ".join(reasons),
            "infeasibility_reasons": "; ".join(infeas_reasons)
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.set_index("train_id").reindex(list(data["job_card"]["train_id"])).reset_index()

    if save_csv:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"final_plan_detailed_{ts}.csv"
        path = os.path.join(OUTPUT_DIR, fname)
        out_df.to_csv(path, index=False)
        print(f"Saved detailed plan to: {path}")

    print("\nDetailed assignment summary:")
    print(out_df.to_string(index=False))

    return out_df



if __name__ == "__main__":
    explain_all()
