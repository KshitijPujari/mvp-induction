"""
Cost model for KMRL Induction MVP
- Linear cost: w_readiness*readiness_risk + w_mileage*mileage_penalty + w_shunt*shunt_proxy
- Produces a cost matrix (trains x slots) with INF for infeasible pairs.
- Includes a simple explanation helper.
"""


import os
import math
import pandas as pd


try:
    from src import constraints
except Exception:
    import constraints  # fallback


# Default weights (tune later)
DEFAULT_WEIGHTS = {
    "w_readiness": 10.0,
    "w_mileage": 1.0,
    "w_shunt": 0.5,
}


INF = 1_000_000.0



def readiness_risk(train_id, job_df, cert_df, clean_df):
    """
    Simple readiness risk score:
    - blocked -> 5
    - missing certificates -> 4
    - cleaning not done -> 1
    - else -> 0
    """
    job_row = job_df.loc[job_df["train_id"] == train_id]
    cert_row = cert_df.loc[cert_df["train_id"] == train_id]
    clean_row = clean_df.loc[clean_df["train_id"] == train_id]

    if job_row.empty or cert_row.empty or clean_row.empty:
        return 5

    job_row = job_row.iloc[0]
    cert_row = cert_row.iloc[0]
    clean_row = clean_row.iloc[0]

    if job_row.get("blocked", 0) == 1:
        return 5

    if not all(cert_row.get(c, 0) == 1 for c in ["brake_test_valid", "pto_valid", "atp_valid"]):
        return 4

    if clean_row.get("cleaning_done", 1) == 0:
        return 1

    return 0



def mileage_penalty(train_id, mileage_df, threshold_km=200000.0, scale=10000.0):
    """
    penalty = max(0, (mileage_km - threshold_km) / scale)
    """
    row = mileage_df.loc[mileage_df["train_id"] == train_id]
    if row.empty:
        return 2.0

    mileage = float(row.iloc[0].get("mileage_km", 0))
    penalty = max(0.0, (mileage - threshold_km) / scale)
    return penalty



def shunt_proxy(slot_row):
    """
    Convert shunt_distance to a small cost.
    """
    if "shunt_distance" in slot_row:
        try:
            return float(slot_row["shunt_distance"]) / 100.0
        except Exception:
            pass

    role = str(slot_row.get("role", "")).lower()
    mapping = {"service": 1.0, "standby": 0.5, "ibl": 1.5}
    return mapping.get(role, 1.0)



def compute_pair_cost(train_id, slot_row, data, weights=None):
    """
    Return the cost for a train-slot pair.
    If infeasible, return INF.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    feasible = constraints.is_pair_feasible(
        train_id,
        job_df=data["job_card"],
        cert_df=data["certificate_validity"],
        clean_df=data["cleaning_roster"],
        slot_row=slot_row,
    )

    if not feasible:
        return INF

    r_risk = readiness_risk(train_id, data["job_card"], data["certificate_validity"], data["cleaning_roster"])
    m_pen = mileage_penalty(train_id, data["mileage"])
    s_proxy = shunt_proxy(slot_row)

    cost = (
        weights["w_readiness"] * r_risk
        + weights["w_mileage"] * m_pen
        + weights["w_shunt"] * s_proxy
    )

    if not math.isfinite(cost):
        return INF

    return float(cost)



def build_cost_matrix(data, weights=None, inf=INF):
    """
    Return a DataFrame cost matrix.
    Rows = train_id
    Cols = slot_id
    """
    job = data["job_card"]
    slots = data["induction_slots"]

    trains = list(job["train_id"].unique())
    slot_ids = list(slots["slot_id"].astype(str))

    matrix = pd.DataFrame(index=trains, columns=slot_ids, dtype=float)

    for _, slot_row in slots.iterrows():
        sid = str(slot_row["slot_id"])
        for t in trains:
            matrix.at[t, sid] = compute_pair_cost(t, slot_row, data, weights=weights)

    return matrix



def explain_reason(train_id, slot_row, data):
    """
    Simple human-readable explanation.
    """
    reasons = []

    r = readiness_risk(train_id, data["job_card"], data["certificate_validity"], data["cleaning_roster"])
    if r >= 5:
        reasons.append("blocked / high readiness risk")
    elif r >= 4:
        reasons.append("certificate issue")
    elif r >= 1:
        reasons.append("cleaning pending")

    m = mileage_penalty(train_id, data["mileage"])
    if m >= 3:
        reasons.append("very high mileage")
    elif m >= 1:
        reasons.append("high mileage")

    s = shunt_proxy(slot_row)
    if s >= 4:
        reasons.append("long shunt movement")

    if not reasons:
        return "good fit"

    return "; ".join(reasons[:2])



if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(ROOT, "data")

    data = {
        "job_card": pd.read_csv(os.path.join(DATA_DIR, "job_card.csv")),
        "certificate_validity": pd.read_csv(os.path.join(DATA_DIR, "certificate_validity.csv")),
        "cleaning_roster": pd.read_csv(os.path.join(DATA_DIR, "cleaning_roster.csv")),
        "mileage": pd.read_csv(os.path.join(DATA_DIR, "mileage.csv")),
        "induction_slots": pd.read_csv(os.path.join(DATA_DIR, "induction_slots.csv")),
    }

    print("Building cost matrix...")
    cm = build_cost_matrix(data)
    print(cm)

    print("\nSample explanations (feasible pairs only):")
    for _, slot_row in data["induction_slots"].iterrows():
        for t in data["job_card"]["train_id"].unique():
            if constraints.is_pair_feasible(t, data["job_card"], data["certificate_validity"], data["cleaning_roster"], slot_row):
                print(f"{t} -> {slot_row['slot_id']}: {explain_reason(t, slot_row, data)}")
