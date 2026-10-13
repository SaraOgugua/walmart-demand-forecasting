
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def anonymize_data():
    input_file = PROCESSED_DIR / "sales_calendar_sample.csv"
    output_file = PROCESSED_DIR / "sales_calendar_anonymized_sample.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {input_file}"
        )

    data = pd.read_csv(input_file)

    # Remove direct product and store identifiers from the analysis copy.
    identifiers_to_remove = [
        "id",
        "item_id",
        "store_id",
        "state_id",
    ]

    columns_to_remove = [
        column for column in identifiers_to_remove
        if column in data.columns
    ]

    anonymized_data = data.drop(columns=columns_to_remove)
    anonymized_data.to_csv(output_file, index=False)

    print("PRIVACY AND IDENTIFIER-REMOVAL REPORT")
    print("=" * 45)
    print(f"Input rows: {len(data):,}")
    print(f"Output rows: {len(anonymized_data):,}")
    print(f"Identifiers removed: {columns_to_remove}")
    print(f"Output file: {output_file.name}")
    print("Original prepared data remains unchanged.")
    print(
        "Note: Identifier removal alone does not guarantee "
        "full anonymization."
    )


if __name__ == "__main__":
    anonymize_data()
