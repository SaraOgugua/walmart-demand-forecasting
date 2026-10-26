
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MERGED_FILE = PROCESSED_DIR / "sales_calendar_sample.csv"


def load_merged_data():
    assert MERGED_FILE.exists(), (
        f"Required dataset not found: {MERGED_FILE}"
    )
    return pd.read_csv(MERGED_FILE)


def test_calendar_merge_has_expected_columns():
    data = pd.read_csv(MERGED_FILE, nrows=5)

    expected = {"id", "day", "units_sold", "date"}
    assert expected.issubset(data.columns)


def test_calendar_merge_has_no_missing_sales_values():
    data = pd.read_csv(MERGED_FILE, usecols=["units_sold"])

    assert data["units_sold"].notna().all()


def test_calendar_merge_preserves_row_count():
    data = pd.read_csv(MERGED_FILE)

    assert len(data) == 191300


def test_calendar_merge_has_no_missing_dates():
    data = pd.read_csv(MERGED_FILE, usecols=["date"])

    assert data["date"].notna().all()

