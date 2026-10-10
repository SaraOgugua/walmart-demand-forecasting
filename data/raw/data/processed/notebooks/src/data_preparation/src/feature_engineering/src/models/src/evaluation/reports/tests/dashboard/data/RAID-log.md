# Project RAID Log

**Project:** Explainable Retail Demand Forecasting for Walmart Inventory and Demand Planning  
**Purpose:** Track risks, assumptions, issues, and dependencies affecting project delivery.

## 1. Risks

| ID | Risk | Potential impact | Planned mitigation |
|---|---|---|---|
| R1 | Large dataset files may exceed available memory or storage capacity. | Delays in data preparation and modelling. | Profile file sizes, use efficient data types, and process data in manageable sections. |
| R2 | Data leakage may lead to unrealistically strong forecast results. | Incorrect model selection and unreliable business recommendations. | Use time-aware validation and ensure features contain only information available at forecast time. |
| R3 | Forecast accuracy may vary across products, stores, or categories. | Some operational groups may receive less reliable forecasts. | Report performance by relevant groups and document limitations. |
| R4 | Explainability outputs may be misinterpreted as causal evidence. | Managers may make unsupported decisions. | Explain that feature importance indicates model relationships, not necessarily causation. |

## 2. Assumptions

| ID | Assumption | Validation approach |
|---|---|---|
| A1 | The public M5 dataset can be accessed for the project. | Confirm dataset access and applicable terms. |
| A2 | Historical sales, calendar, and price information are sufficient to build meaningful baseline forecasts. | Examine data coverage and missing values. |
| A3 | Python and the planned libraries can support the required analysis. | Test the environment and document compatible versions. |
| A4 | Forecasting insights can support human inventory-planning decisions. | Translate evaluation results into clearly qualified recommendations. |

## 3. Issues

| ID | Issue | Current status | Resolution |
|---|---|---|---|
| I1 | Data profiling and cleaning have not yet been completed. | Open — planned work. | Inspect data quality after acquiring the dataset. |
| I2 | No forecasting model has been evaluated yet. | Open — planned work. | Establish a baseline before comparing candidate models. |
| I3 | Dashboard implementation and validation are incomplete. | Open — planned work. | Develop the dashboard after producing verified analytical outputs. |

## 4. Dependencies

| ID | Dependency | Why it matters |
|---|---|---|
| D1 | Access to the M5 Forecasting – Accuracy dataset. | Required for data preparation and analysis. |
| D2 | A functioning Python environment and compatible libraries. | Required for modelling, evaluation, and explainability. |
| D3 | Completed data preparation and feature engineering. | Required before reliable model training and evaluation. |
| D4 | Verified model evaluation results. | Required before making evidence-based recommendations and developing the final dashboard. |

## Review and Maintenance

This log should be reviewed at each project milestone. Update statuses, add newly identified items, and record the actions taken. Items listed as planned or open must not be treated as completed project work.
