# tools/analyse_output.py
import pathlib
import pandas as pd


ROOT = pathlib.Path(".").resolve()
OUT = ROOT / "outputs"


def find_latest_detailed():
    files = sorted(OUT.glob("final_plan_detailed_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No detailed plan CSV found in outputs/")
    return files[0]


def summarize(df):
    total = len(df)
    assigned = df["assigned_slot"].notna().sum()
    unassigned = total - assigned
    avg_cost = df["total_cost_recomputed"].dropna().mean()
    median_cost = df["total_cost_recomputed"].dropna().median()

    high_cost = df[df["total_cost_recomputed"].notna()].sort_values("total_cost_recomputed", ascending=False).head(10)
    slot_usage = df["assigned_slot"].value_counts(dropna=True)
    role_usage = df["role"].value_counts(dropna=True)

    infeas = df[df["assigned_slot"].isna()]["infeasibility_reasons"].dropna().astype(str)
    reasons_series = infeas.str.split(";").explode().str.strip().value_counts()

    return {
        "total_trains": total,
        "assigned": assigned,
        "unassigned": unassigned,
        "avg_cost": float(avg_cost) if pd.notna(avg_cost) else None,
        "median_cost": float(median_cost) if pd.notna(median_cost) else None,
        "slot_usage": slot_usage,
        "role_usage": role_usage,
        "top_high_cost": high_cost,
        "infeasibility_reason_counts": reasons_series
    }


def print_summary(s):
    print(f"Total trains: {s['total_trains']}")
    print(f"Assigned: {s['assigned']}  Unassigned: {s['unassigned']}")
    print(f"Average cost (assigned only): {s['avg_cost']:.3f}" if s['avg_cost'] is not None else "Average cost: N/A")
    print(f"Median cost: {s['median_cost']:.3f}" if s['median_cost'] is not None else "Median cost: N/A")

    print("\nSlot usage (most -> least):")
    print(s['slot_usage'].to_string())

    print("\nRole usage:")
    print(s['role_usage'].to_string())

    print("\nTop infeasibility reasons (unassigned trains):")
    print(s['infeasibility_reason_counts'].head(10).to_string())

    print("\nTop high-cost assignments (worst fits):")
    if not s['top_high_cost'].empty:
        print(s['top_high_cost'][["train_id","assigned_slot","role","total_cost_recomputed","reasons"]].to_string(index=False))
    else:
        print("None")


def interactive_inspect(df):
    print("\nYou can now inspect subsets. Examples:")
    print("1) df[df['assigned_slot'].isna()]  # unassigned trains")
    print("2) df[df['role']=='Service']        # service-assigned trains")
    print("3) df[df['total_cost_recomputed'] > 20]  # high cost assignments")
    print("\nPrinting first 20 rows:")
    print(df.head(20).to_string(index=False))


def main():
    path = find_latest_detailed()
    print(f"Loading latest detailed file: {path}")
    df = pd.read_csv(path)
    summary = summarize(df)
    print_summary(summary)
    interactive_inspect(df)


if __name__ == '__main__':
    main()
