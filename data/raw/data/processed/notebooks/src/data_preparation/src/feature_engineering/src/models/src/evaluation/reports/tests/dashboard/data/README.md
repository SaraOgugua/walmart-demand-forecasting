# Data Management

## Data Source

This project uses the public Kaggle M5 Forecasting – Accuracy dataset:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

The dataset contains historical Walmart sales information, calendar events, selling prices, and product and store identifiers.

## Folder Organization

- `raw/` — original dataset files, retained separately from processed data.
- `processed/` — cleaned and transformed data prepared for analysis and modelling.

## Data Handling Principles

1. Preserve source data and document any transformations.
2. Validate data types, missing values, duplicates, and date coverage.
3. Integrate sales, calendar, and pricing data using appropriate identifiers.
4. Avoid data leakage when creating forecasting features and evaluating models.
5. Do not upload large raw datasets or confidential credentials to the repository.
6. Record relevant assumptions and limitations in project documentation.

## Current Status

The folder structure is established. Dataset acquisition, data profiling, cleaning, and transformation are planned activities and should be documented as they are completed.
