
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def merge_calendar_data():
    sales_file = PROCESSED_DIR / "sales_sample_long.csv"
    calendar_file = RAW_DIR / "calendar.csv"
    output_file = PROCESSED_DIR / "sales_calendar_sample.csv"

    print("Loading prepared sales and calendar data...")

    sales = pd.read_csv(sales_file)
    calendar = pd.read_csv(calendar_file)

    # Keep the calendar fields needed for analysis
    calendar = calendar[
        ["d", "date", "wm_yr_wk", "weekday", "month", "year",
         "event_name_1", "event_type_1"]
    ].copy()

    # Match the sales day labels (d_1, d_2, ...) to calendar labels
    sales["d"] = sales["day"]

    merged = sales.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one"
    )

    if merged["date"].isna().any():
        raise ValueError("Some sales observations did not match a calendar date.")

    if len(merged) != len(sales):
        raise ValueError("The calendar merge changed the number of sales rows.")

    merged["date"] = pd.to_datetime(merged["date"])

    merged.to_csv(output_file, index=False)

    print("\nCALENDAR MERGE REPORT")
    print("=" * 45)
    print(f"Sales observations: {len(sales)}")
    print(f"Merged observations: {len(merged)}")
    print(f"Calendar dates matched: {merged['date'].notna().sum()}")
    print(f"Output file: {output_file}")
    print("Status: PASS")


if __name__ == "__main__":
    merge_calendar_data()
