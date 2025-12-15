# tools/copy_latest_for_frontend.py

"""
Copy the latest final_plan_detailed_*.csv from outputs/ to outputs/final_plan_detailed.csv
so the static frontend can fetch a stable filename.

Run this after running the solver: python tools/copy_latest_for_frontend.py
"""
from pathlib import Path
import shutil


ROOT = Path('.').resolve()
OUT = ROOT / 'outputs'


files = sorted(OUT.glob('final_plan_detailed_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
if not files:
    print('No final_plan_detailed_*.csv files found in outputs/')
    raise SystemExit(1)

latest = files[0]
dest = OUT / 'final_plan_detailed.csv'
shutil.copy2(latest, dest)
print('Copied', latest.name, '->', dest)

