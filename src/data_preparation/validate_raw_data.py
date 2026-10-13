
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

REQUIRED_COLUMNS = {
    "calendar.csv": {"date", "wm_yr_wk", "d"},
    "sell_prices.csv": {
        "store_id", "item_id", "wm_yr_wk", "sell_price"
    },
    "sales_train_validation.csv": {
        "id", "item_id", "dept_id", "cat_id",
        "store_id", "state_id"
    },
}


def validate_source_file(filename, required_columns):
    """Check source-file existence, schema, and basic integrity."""
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing_columns = set(required_columns) - set(columns)

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing_columns)}"
        )

    sample = pd.read_csv(path, nrows=100)

    if sample.empty:
        raise ValueError(f"{filename} contains no data rows.")

    if sample.columns.duplicated().any():
        raise ValueError(f"{filename} has duplicate column names.")

    return {
        "file": filename,
        "status": "PASS",
        "columns": len(columns),
        "sample_rows_checked": len(sample),
        "size_mb": round(path.stat().st_size / (1024**2), 2),
    }


def main():
    results = []

    for filename, required_columns in REQUIRED_COLUMNS.items():
        result = validate_source_file(filename, required_columns)
        results.append(result)

    print("\nRAW DATA VALIDATION REPORT")
    print("=" * 60)

    for result in results:
        print(
            f"{result['status']} | {result['file']} | "
            f"{result['columns']} columns | "
            f"{result['sample_rows_checked']} sample rows | "
            f"{result['size_mb']} MB"
        )

    print("\nAll source-file schema and sample checks passed.")


if __name__ == "__main__":
    main()
