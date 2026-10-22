# Model Card: Explainable Retail Demand Forecasting

**Model version:** 1.0  
**Model type:** HistGradientBoostingRegressor  
**Project:** Explainable Retail Demand Forecasting and Decision-Support Platform  
**Status:** Experimental; not approved for autonomous operational decisions

## 1. Intended Use

The model estimates daily unit sales for sampled retail products using historical sales patterns and calendar features. Its intended users are demand planners, inventory and operations managers, business analysts, and decision-makers evaluating demand-planning approaches. Outputs are decision-support estimates and must be reviewed alongside business context and human judgment.

The model is a research prototype using public M5 Forecasting data. It is not a proprietary Walmart system, a live Walmart integration, or an automated replenishment solution.

## 2. Model Development

The selected model is a HistGradientBoostingRegressor trained on lagged sales, rolling averages, calendar variables, product indicators, and event features. Data were separated chronologically into training, validation, and test periods. Candidate models were compared using validation MAE, and the selected model was evaluated on a held-out test period.

The saved test evaluation covers 2,800 observations from 28 March to 24 April 2016. The prediction procedure is rolling one-step-ahead: lagged inputs use historical observations available before each target date. It should not be described as a fully recursive multi-day forecast.

## 3. Data Summary

The development sample contains 100 products from one store, department, and category, with daily observations spanning 29 January 2011 to 24 April 2016 before feature-based row removal. Calendar features include weekday, month, year, and event indicators.

This restricted sample is suitable for demonstrating a forecasting workflow but does not establish generalisability across all stores, departments, categories, or retail settings.

## 4. Performance

| Metric | Baseline | Selected model |
|---|---:|---:|
| MAE | 1.435 | 1.132 |
| RMSE | 3.777 | 2.595 |
| R² | Not reported | 0.398 |
| WAPE | 115.89% | 91.42% |

The selected model reduced test MAE by approximately 21.1% and RMSE by approximately 31.3% relative to the baseline. However, WAPE remained high, indicating substantial forecast error relative to total observed demand. These results support further experimentation, not a claim of production readiness.

## 5. Fairness and Segment Performance

Demographic fairness could not be assessed because the dataset does not contain relevant demographic attributes. Instead, performance was examined across low-, medium-, and high-demand product groups.

The model performed worse than the baseline on low-demand products, where approximately 85.7% of observed test values were zero. This indicates a material limitation for intermittent-demand items. The dataset also covers only one store, department, and category, preventing defensible claims about cross-store or cross-category fairness.

## 6. Explainability

SHAP explanations were generated using a permutation explainer after the TreeExplainer additivity check failed. The saved global results identified rolling sales averages among the most influential features. Local explanations illustrate how features contributed to individual predictions. A one-observation sensitivity analysis varied selected sales features by ±20%.

SHAP values and sensitivity results describe model behaviour under the tested conditions; they do not establish causation. The explanation sample was small, and further testing is required to establish explanation stability.

## 7. Limitations and Risks

- High WAPE indicates that forecasts can be materially inaccurate.
- Low-demand products are a specific weakness because demand is intermittent.
- The sample does not establish performance across stores, departments, or categories.
- The model may not respond to market changes, supply constraints, promotions, or structural demand shifts not represented in its features.
- Forecast errors could contribute to poor inventory decisions if outputs are used without review.
- Predictions may be fractional because the model estimates a continuous target; they should not be treated as exact unit counts.

## 8. Human Oversight and Monitoring

The model must remain decision-support only. Users should review forecasts against recent sales, business events, inventory context, and operational constraints. Before any operational adoption, evaluation should be repeated on broader data and later time periods. Monitoring should include MAE, RMSE, WAPE, residual patterns, prediction bias, and error by demand segment. Low-demand performance should be addressed before use for automated or high-impact replenishment decisions.

## 9. Governance and Versioning

The project uses version-controlled code and tracked experiments. Model artefacts should be loaded only from trusted project sources. Each model release should retain its training-data description, feature definitions, validation results, limitations, and approval status. Material changes to data, features, or algorithms require revalidation and an updated model card.