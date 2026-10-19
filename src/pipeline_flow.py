
from pathlib import Path
import subprocess
import sys

from prefect import flow, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "src/data_preparation/validate_raw_data.py",
    "src/data_preparation/prepare_sales_data.py",
    "src/data_preparation/merge_calendar_data.py",
    "src/data_preparation/validate_prepared_data.py",
    "src/data_preparation/anonymize_data.py",
]


@task
def run_pipeline_step(step):
    print(f"\nRunning pipeline step: {step}", flush=True)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / step)],
        cwd=PROJECT_ROOT,
        check=True,
    )


@flow(name="Walmart Demand Forecasting Pipeline")
def walmart_demand_forecasting_pipeline():
    for step in STEPS:
        run_pipeline_step(step)

    print("\nPIPELINE COMPLETED SUCCESSFULLY", flush=True)


if __name__ == "__main__":
    walmart_demand_forecasting_pipeline()
