"""
Constraint Layer for KMRL Train Induction MVP
---------------------------------------------
This file defines simple binary feasibility checks for each train–slot pair.

Rules (MVP Level):
1. Certificate validity → all required certificates must be 1.
2. Job-card block → if train is blocked (blocked == 1), it cannot be assigned to Service slots.
3. Cleaning availability → if cleaning_done == 0 AND slot role is Service, treat it as risky/infeasible.
4. (Optional MVP) A train_id must exist in all datasets (no missing info).
"""


import pandas as pd


def is_cert_valid(cert_row):
    """
    Certificates required: brake, PTO, ATP.
    Returns True if all are valid (==1), else False.
    """
    required_cols = ["brake_test_valid", "pto_valid", "atp_valid"]
    return all(cert_row[col] == 1 for col in required_cols)



def is_jobcard_allowed(job_row, slot_row):
    """
    If blocked == 1, train CANNOT be placed in Service slots.
    But it CAN go to Standby or IBL.
    """
    if job_row["blocked"] == 1 and slot_row["role"] == "Service":
        return False
    return True



def is_cleaning_ok(clean_row, slot_row):
    """
    If cleaning is NOT done and slot is Service → infeasible.
    For Standby or IBL, allow it.
    """
    if clean_row["cleaning_done"] == 0 and slot_row["role"] == "Service":
        return False
    return True



def is_pair_feasible(train_id, job_df, cert_df, clean_df, slot_row):
    """
    Updated MVP feasibility rules:

    SERVICE:
        - Train must have all certificates valid.
        - Train must have cleaning_done == 1.
        - Blocked trains (blocked == 1) cannot be assigned.

    STANDBY / IBL:
        - Certificates are NOT required.
        - Cleaning is NOT required.
        - Blocked trains ARE allowed.

    Applies to each train-slot pair.
    Returns True if feasible, False otherwise.
    """

    job_row  = job_df.loc[job_df["train_id"] == train_id]
    cert_row = cert_df.loc[cert_df["train_id"] == train_id]
    clean_row = clean_df.loc[clean_df["train_id"] == train_id]

    # if any of these rows are missing entirely, we cannot evaluate safely
    if job_row.empty or cert_row.empty or clean_row.empty:
        return False

    job_row = job_row.iloc[0]
    cert_row = cert_row.iloc[0]
    clean_row = clean_row.iloc[0]

    role = str(slot_row.get("role", "")).strip().lower()

    # 1. Blocked trains cannot go to Service
    if role == "service" and job_row.get("blocked", 0) == 1:
        return False

    # 2. SERVICE slot must have all certificates valid
    if role == "service":
        cert_ok = all(cert_row.get(c, 0) == 1 for c in ["brake_test_valid", "pto_valid", "atp_valid"])
        if not cert_ok:
            return False

        # SERVICE must have cleaning done
        if clean_row.get("cleaning_done", 1) == 0:
            return False

        return True

    # 3. STANDBY / IBL (roles other than "service"):
    #    Certificates NOT required, cleaning NOT required.
    return True



def build_feasibility_map(data):
    """
    Creates a dict: feasible[(train_id, slot_id)] = True/False
    """
    job = data["job_card"]
    cert = data["certificate_validity"]
    clean = data["cleaning_roster"]
    slots = data["induction_slots"]

    feasible = {}

    for _, slot_row in slots.iterrows():
        slot_id = slot_row["slot_id"]
        for train_id in job["train_id"].unique():
            feasible[(train_id, slot_id)] = is_pair_feasible(
                train_id=train_id,
                job_df=job,
                cert_df=cert,
                clean_df=clean,
                slot_row=slot_row
            )

    return feasible



if __name__ == "__main__":
    # Mini test (only works if ingestion.py already loaded CSVs)
    import os
    import pandas as pd

    ROOT = os.path.dirname(os.path.dirname(__file__))
    DATA = os.path.join(ROOT, "data")

    job = pd.read_csv(os.path.join(DATA, "job_card.csv"))
    cert = pd.read_csv(os.path.join(DATA, "certificate_validity.csv"))
    clean = pd.read_csv(os.path.join(DATA, "cleaning_roster.csv"))
    slots = pd.read_csv(os.path.join(DATA, "induction_slots.csv"))

    print("Feasibility sample checks:\n")
    for _, slot_row in slots.iterrows():
        for train_id in job["train_id"].unique():
            print(
                train_id,
                slot_row["slot_id"],
                is_pair_feasible(train_id, job, cert, clean, slot_row)
            )
        print()
