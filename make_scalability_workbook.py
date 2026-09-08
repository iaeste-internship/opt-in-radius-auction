"""Rebuild the archived scalability workbook from the fresh CSV outputs.

`ORA_scalability.xlsx` is the checked, consolidated copy of Experiment D. It is
written from the CSVs produced by run_scalability_benchmark.py, so it must be
rebuilt whenever the benchmark is rerun with a different seed count; otherwise
the archived workbook reports a different experiment from the manuscript.

Run from the project root, after run_scalability_benchmark.py:

    python make_scalability_workbook.py

Inputs:

    results/scalability_raw.csv
    results/scalability_summary.csv
    results/scalability_scaling_fits.csv

Output:

    ORA_scalability.xlsx    (sheets: _README, raw, summary, scaling_fits)
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
OUTPUT_FILE = Path("ORA_scalability.xlsx")

# Sheet name -> source CSV, in the order the workbook stores them.
SHEETS = {
    "raw": "scalability_raw.csv",
    "summary": "scalability_summary.csv",
    "scaling_fits": "scalability_scaling_fits.csv",
}


def load_sheets() -> dict:
    frames = {}
    missing = []
    for sheet, filename in SHEETS.items():
        path = RESULTS_DIR / filename
        if path.exists():
            frames[sheet] = pd.read_csv(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing scalability CSVs: "
            + ", ".join(missing)
            + ". Run run_scalability_benchmark.py first."
        )
    return frames


def provenance_sheet() -> pd.DataFrame:
    rows = [
        {
            "sheet": "_README",
            "kind": "provenance / source-file map",
            "source_file": None,
        }
    ]
    rows += [
        {"sheet": sheet, "kind": "from CSV", "source_file": filename}
        for sheet, filename in SHEETS.items()
    ]
    return pd.DataFrame(rows)


def report(frames: dict) -> None:
    """Print what the rebuilt workbook contains.

    The seed count is echoed because it is the value quoted in the manuscript
    methodology, and a silent mismatch there is the failure this script exists
    to prevent.
    """

    raw = frames["raw"]
    summary = frames["summary"]
    seeds = sorted(set(summary["n_seeds"].astype(int)))
    sizes = sorted(summary["n_passengers"].astype(int))

    print(f"Wrote {OUTPUT_FILE}")
    print(f"  market sizes : {sizes}")
    print(f"  seeds/size   : {seeds if len(seeds) > 1 else seeds[0]}")
    print(f"  raw rows     : {len(raw)}")
    if len(seeds) > 1:
        print("  WARNING: seed count varies across market sizes.")


def main() -> None:
    frames = load_sheets()
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        provenance_sheet().to_excel(writer, sheet_name="_README", index=False)
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    report(frames)


if __name__ == "__main__":
    main()
