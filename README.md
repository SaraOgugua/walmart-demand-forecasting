# walmart-demand-forecasting
BAN6800 business analytics project for explainable retail demand forecasting using the Walmart M5 dataset.
# Explainable Retail Demand Forecasting for Walmart Inventory and Demand Planning

## Project Overview

This project proposes a reproducible and explainable retail demand forecasting solution using the M5 Forecasting – Accuracy dataset. The goal is to support inventory and demand-planning decisions by analyzing historical sales, calendar events, and selling prices.

The project is being developed as part of the BAN6800 Business Analytics course at Nexford University. Forecasts are intended to support human decision-making, not automatically generate inventory orders.

## Business Problem

Retail demand varies across products, stores, seasons, calendar events, and prices. Inaccurate forecasts can contribute to stock shortages, excess inventory, and inefficient resource allocation. This project will evaluate forecasting approaches and examine whether their performance is sufficiently accurate, transparent, and consistent across operational groups.

## Dataset

The project uses the public [Kaggle M5 Forecasting – Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), which includes:

- Historical daily sales
- Calendar and event information
- Weekly selling prices
- Product, department, category, store, and state identifiers

Raw dataset files are not stored in this repository. Refer to the dataset source for access and applicable usage terms.

## Planned Workflow

1. Profile and validate the source data.
2. Clean and transform sales observations.
3. Integrate calendar and price information.
4. Engineer forecasting features.
5. Establish baseline forecasts and evaluate candidate models.
6. Compare forecast errors across operational groups.
7. Apply suitable explainability methods.
8. Prepare a Power BI dashboard for demand-planning decision support.

## Evaluation and Responsible AI

Planned evaluation measures include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Weighted Absolute Percentage Error (WAPE), where appropriate. The project will also examine group-level performance, temporal leakage, data quality, model limitations, and explainability.

Model selection will be based on observed results rather than assumed performance.

## Repository Structure

- `data/` — raw and processed data organization
- `notebooks/` — exploratory and modelling notebooks
- `src/` — reusable data preparation, feature engineering, modelling, and evaluation code
- `reports/` — technical documentation and analytical reports
- `tests/` — data and model evaluation tests
- `dashboard/` — Power BI dashboard artifacts
- `models/` — saved model artifacts
- `requirements.txt` — Python environment dependencies

## Current Status

**Planning and repository setup.** The project structure is being prepared. Data profiling, pipeline development, model evaluation, and dashboard implementation will follow in later project stages.

## References

Kaggle. (2020). *M5 forecasting – Accuracy*. https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
