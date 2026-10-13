
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "src/data_preparation/validate_raw_data.py",
    "src/data_preparation/prepare_sales_data.py",
    "src/data_preparation/merge_calendar_data.py",
]


def main():
    print("STARTING WALMART DEMAND FORECASTING PIPELINE")

    for step in STEPS:
        print(f"\nRunning: {step}")

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / step)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:
            print(f"PIPELINE FAILED at {step}")
            sys.exit(result.returncode)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
