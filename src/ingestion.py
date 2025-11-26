import os
import pandas as pd


# Path to the data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


REQUIRED_FILES = {
    "job_card": "job_card.csv",
    "certificate_validity": "certificate_validity.csv",
    "cleaning_roster": "cleaning_roster.csv",
    "mileage": "mileage.csv",
    "induction_slots": "induction_slots.csv",
}



def _read_csv(name, filename):
    """
    Read a CSV file and return a pandas DataFrame.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expected data file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df



def load_all():
    """
    Load all required CSVs from the /data folder.
    Returns a dictionary of DataFrames.
    """
    data = {}
    for key, fname in REQUIRED_FILES.items():
        data[key] = _read_csv(key, fname)
    return data



def validate_basic_schema(data):
    """
    Basic validation: checks for essential columns.
    """
    print("\nPerforming basic schema validation...")

    required_columns = {
        "job_card": ["train_id", "blocked"],
        "certificate_validity": ["train_id", "brake_test_valid"],
        "cleaning_roster": ["train_id", "cleaning_done"],
        "mileage": ["train_id", "mileage_km"],
        "induction_slots": ["slot_id", "role", "shunt_distance"],
    }

    for name, required in required_columns.items():
        df = data[name]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    print("Basic schema validation passed.")



if __name__ == "__main__":
    print("Loading all CSV data...")
    data = load_all()
    validate_basic_schema(data)

    print("\nPreview: job_card.csv")
    print(data["job_card"].head().to_string(index=False))

    print("\nPreview: induction_slots.csv")
    print(data["induction_slots"].head().to_string(index=False))
