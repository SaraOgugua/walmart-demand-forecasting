
# Walmart Demand Forecasting — Data Pipeline Architecture

## 1. Purpose

This document describes the data pipeline developed for the BAN6800 business analytics project. The pipeline prepares a sample of the M5 Forecasting – Accuracy dataset for future demand forecasting analysis.

## 2. Pipeline Workflow

1. **Raw data ingestion:** Read the sales, calendar, and selling-price CSV files from `data/raw/`.
2. **Source validation:** Check that required files, columns, and sample records are available.
3. **Data preparation:** Select 100 products and reshape daily sales from wide to long format.
4. **Calendar integration:** Join daily sales observations with calendar information using the day identifier.
5. **Data-quality validation:** Check required columns, missing values, non-negative sales, and date completeness.
6. **Privacy processing:** Create a separate analysis copy with direct product and store identifiers removed.
7. **Output storage:** Save prepared datasets in `data/processed/` for subsequent analysis.

## 3. Main Components

| Component | Purpose |
|---|---|
| `validate_raw_data.py` | Checks the source files and required columns |
| `prepare_sales_data.py` | Converts the sales sample into long format |
| `merge_calendar_data.py` | Integrates calendar information |
| `validate_prepared_data.py` | Runs Great Expectations quality checks |
| `anonymize_data.py` | Removes selected identifiers from an analysis copy |
| `run_pipeline.py` | Orchestrates the main validation, preparation, and merge steps |
| `test_data_pipeline.py` | Runs automated pipeline tests |

## 4. Current Validation Results

- Source validation: PASS
- Products processed: 100
- Daily sales observations: 191,300
- Calendar observations matched: 191,300
- Automated tests: 4 of 4 passed
- Great Expectations checks: 5 of 5 passed
- Main pipeline execution: successful

## 5. Privacy and Limitations

The raw dataset and original prepared data remain unchanged by the identifier-removal script. The separate analysis copy removes the `id`, `item_id`, `store_id`, and `state_id` columns where present.

Removing these identifiers does not guarantee complete anonymization. The dataset is a public retail forecasting dataset, and this project does not claim to process personal customer records.

The pipeline currently processes a 100-product sample rather than the full dataset. Forecast modelling, time-based model evaluation, and production deployment are outside the scope of the current pipeline implementation.

## 6. Future Improvements

- Add automated validation of selling-price data.
- Introduce structured logging and clearer error reports.
- Add a formal orchestration workflow with dependency tracking.
- Expand testing to cover invalid input files and unexpected data changes.
- Evaluate forecasting models using time-aware validation and operational error metrics.
