
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_sales_sample_has_expected_columns():
    file_path = PROCESSED_DIR / "sales_sample_long.csv"
    data = pd.read_csv(file_path, nrows=5)

    expected = {
        "id", "item_id", "dept_id", "cat_id",
        "store_id", "state_id", "day", "units_sold"
    }

    assert expected.issubset(data.columns)


def test_sales_sample_has_no_missing_sales_values():
    file_path = PROCESSED_DIR / "sales_sample_long.csv"
    data = pd.read_csv(file_path, usecols=["units_sold"])

    assert data["units_sold"].notna().all()


def test_calendar_merge_preserves_row_count():
    file_path = PROCESSED_DIR / "sales_calendar_sample.csv"
    data = pd.read_csv(file_path)

    assert len(data) == 191300


def test_calendar_merge_has_no_missing_dates():
    file_path = PROCESSED_DIR / "sales_calendar_sample.csv"
    data = pd.read_csv(file_path, usecols=["date"])

    assert data["date"].notna().all()
