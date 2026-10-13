
from pathlib import Path
import pandas as pd
import great_expectations as gx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def validate_prepared_data():
    file_path = PROCESSED_DIR / "sales_calendar_sample.csv"

    print("Loading prepared data for quality checks...")
    data = pd.read_csv(file_path)

    # Create a Great Expectations Data Context
    context = gx.get_context(mode="ephemeral")

    # Register a Pandas datasource and create a data asset
    datasource = context.data_sources.add_pandas(
        name="walmart_datasource"
    )

    asset = datasource.add_dataframe_asset(
        name="prepared_sales"
    )

    batch_definition = asset.add_batch_definition_whole_dataframe(
        name="prepared_sales_batch"
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": data}
    )

    checks = [
        gx.expectations.ExpectColumnToExist(
            column="units_sold"
        ),
        gx.expectations.ExpectColumnToExist(
            column="date"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="units_sold"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="units_sold",
            min_value=0
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="date"
        ),
    ]

    print("\nGREAT EXPECTATIONS DATA QUALITY REPORT")
    print("=" * 50)

    results = []

    for expectation in checks:
        result = batch.validate(expectation)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"{status} | {result.expectation_config.type}")

    passed = sum(result.success for result in results)
    print(f"\nExpectations passed: {passed}/{len(results)}")

    if passed != len(results):
        raise ValueError("One or more data-quality checks failed.")

    print("Overall status: PASS")


if __name__ == "__main__":
    validate_prepared_data()


