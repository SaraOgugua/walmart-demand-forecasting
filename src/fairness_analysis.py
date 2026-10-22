from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "test_predictions.csv"
OUT = ROOT / "reports"

df = pd.read_csv(INPUT, parse_dates=["date"])

required = {
    "item_id", "date", "units_sold",
    "baseline_prediction", "model_prediction"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df[list(required - {"item_id", "date"})].isna().any().any():
    raise ValueError("Missing values found in actual or prediction fields.")

if (df["units_sold"] < 0).any():
    raise ValueError("Negative actual sales found.")

def calculate_metrics(group, prediction_col):
    actual = group["units_sold"].to_numpy()
    predicted = group[prediction_col].to_numpy()
    errors = predicted - actual

    total_actual = actual.sum()
    wape = (
        np.abs(errors).sum() / total_actual * 100
        if total_actual > 0 else np.nan
    )

    return {
        "observations": len(group),
        "total_actual_sales": total_actual,
        "mean_actual_sales_per_day": actual.mean(),
        "MAE": np.abs(errors).mean(),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "WAPE_pct": wape,
        "zero_sales_share_pct": (actual == 0).mean() * 100,
        "mean_prediction_minus_actual": errors.mean(),
    }

# Product-level performance
product_rows = []
for item_id, group in df.groupby("item_id"):
    row = {"item_id": item_id}
    row.update({
        f"model_{k}": v
        for k, v in calculate_metrics(group, "model_prediction").items()
    })
    baseline = calculate_metrics(group, "baseline_prediction")
    row.update({f"baseline_{k}": v for k, v in baseline.items()})
    product_rows.append(row)

product_metrics = pd.DataFrame(product_rows)

# Relative demand bands based on each product's average actual test sales.
# Ranking before qcut allows products with tied averages to be assigned.
ranked_demand = product_metrics["model_mean_actual_sales_per_day"].rank(
    method="first"
)
product_metrics["demand_band"] = pd.qcut(
    ranked_demand,
    q=3,
    labels=["Low demand", "Medium demand", "High demand"]
)

product_metrics = product_metrics.sort_values(
    "model_MAE", ascending=False
)
product_metrics.to_csv(OUT / "product_error_analysis.csv", index=False)

# Compare model and baseline errors within each demand band.
fairness_rows = []
for band, products in product_metrics.groupby("demand_band", observed=True):
    item_ids = products["item_id"]
    group = df[df["item_id"].isin(item_ids)]

    for model_name, column in [
        ("Selected HistGradientBoosting", "model_prediction"),
        ("Seasonal-naive baseline", "baseline_prediction"),
    ]:
        row = {
            "demand_band": str(band),
            "model": model_name,
            "products": group["item_id"].nunique(),
        }
        row.update(calculate_metrics(group, column))
        fairness_rows.append(row)

fairness = pd.DataFrame(fairness_rows)
fairness.to_csv(OUT / "fairness_metrics_report.csv", index=False)

print("\nOPERATIONAL FAIRNESS / ERROR ANALYSIS")
print("------------------------------------")
print(f"Test observations: {len(df):,}")
print(f"Product series: {df['item_id'].nunique()}")
print("Demand bands are relative groups based on average actual test sales.")
print("\nModel and baseline performance by demand band:")
print(fairness.round(3).to_string(index=False))

print("\nFive products with the highest model MAE:")
print(
    product_metrics[
        ["item_id", "demand_band", "model_MAE",
         "model_WAPE_pct", "model_zero_sales_share_pct"]
    ].head(5).round(3).to_string(index=False)
)

print("\nSaved:")
print("reports/fairness_metrics_report.csv")
print("reports/product_error_analysis.csv")
print(
    "\nInterpretation limit: these are operational error comparisons, "
    "not demographic fairness metrics. The sample covers one store, "
    "one department and one category."
)
