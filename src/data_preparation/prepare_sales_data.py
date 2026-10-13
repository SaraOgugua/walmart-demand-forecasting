
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_sales_data():
    """Convert the sales dataset from wide format to long format."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    input_file = RAW_DIR / "sales_train_validation.csv"
    output_file = PROCESSED_DIR / "sales_sample_long.csv"

    print("Loading sales data...")

    # Load the source data
    sales = pd.read_csv(input_file)

    # Select a manageable sample for initial pipeline testing
    sample = sales.head(100).copy()

    # Identify daily sales columns
    day_columns = [
        column for column in sample.columns
        if column.startswith("d_")
    ]

    print(f"Products selected: {len(sample)}")
    print(f"Daily columns found: {len(day_columns)}")

    # Reshape daily columns into rows
    long_data = sample.melt(
        id_vars=[
            "id", "item_id", "dept_id",
            "cat_id", "store_id", "state_id"
        ],
        value_vars=day_columns,
        var_name="day",
        value_name="units_sold"
    )

    # Ensure sales values are numeric
    long_data["units_sold"] = pd.to_numeric(
        long_data["units_sold"], errors="coerce"
    )

    # Basic quality checks
    if long_data.empty:
        raise ValueError("The prepared sales dataset is empty.")

    if long_data["units_sold"].isna().any():
        raise ValueError("Missing or non-numeric sales values were found.")

    if (long_data["units_sold"] < 0).any():
        raise ValueError("Negative sales values were found.")

    # Save the prepared sample
    long_data.to_csv(output_file, index=False)

    print("\nSALES DATA PREPARATION REPORT")
    print("=" * 45)
    print(f"Products processed: {len(sample)}")
    print(f"Daily observations: {len(long_data)}")
    print(f"Output file: {output_file}")
    print("Status: PASS")


if __name__ == "__main__":
    prepare_sales_data()
