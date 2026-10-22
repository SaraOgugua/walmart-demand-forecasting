
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sales_calendar_sample.csv"
BUNDLE = ROOT / "reports" / "best_demand_model.joblib"
PREDICTIONS = ROOT / "reports" / "test_predictions.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

print("Loading model bundle...")
bundle = joblib.load(BUNDLE)
model = bundle["model"]
feature_cols = bundle["feature_columns"]

print("Recreating training features...")
df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)
grouped = df.groupby("id", sort=False)["units_sold"]

df["lag_7"] = grouped.shift(7)
df["lag_14"] = grouped.shift(14)
df["lag_28"] = grouped.shift(28)
df["rolling_mean_7"] = grouped.transform(
    lambda s: s.shift(7).rolling(7).mean()
)
df["rolling_mean_28"] = grouped.transform(
    lambda s: s.shift(7).rolling(28).mean()
)
df["weekday"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["event_name_1"] = df["event_name_1"].fillna("NoEvent")
df["event_type_1"] = df["event_type_1"].fillna("NoEvent")

df = df.dropna(
    subset=[
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28"
    ]
).copy()

df = pd.get_dummies(
    df,
    columns=["item_id", "event_name_1", "event_type_1"],
    dtype=int,
)

missing = sorted(set(feature_cols) - set(df.columns))
if missing:
    raise ValueError(f"Missing trained features: {missing[:10]}")

X = df[feature_cols].astype(float)
test_start = pd.Timestamp(bundle["test_start"])
test_end = pd.Timestamp(bundle["test_end"])
mask = df["date"].between(test_start, test_end)

X_test = X.loc[mask]
meta = df.loc[mask, ["id", "date", "units_sold"]].copy()

if X_test.empty:
    raise ValueError("No test rows found for the saved test period.")

# Keep SHAP computation manageable on this computer.
X_explain = X_test.sample(
    n=min(20, len(X_test)), random_state=42
)
background = X.sample(n=min(10, len(X)), random_state=42)

print(f"Features: {len(feature_cols)}")
print(f"Test rows: {len(X_test):,}")
print("Calculating SHAP values; this may take a few minutes...")

try:
    explainer = shap.TreeExplainer(model, data=background)
    global_values = explainer(X_explain)
    local_values = explainer(X_test.iloc[[0]])
except Exception as exc:
    print(f"TreeExplainer unavailable ({exc}). Using permutation explainer.")
    explainer = shap.Explainer(
        model.predict,
        background,
        algorithm="permutation",
        feature_names=feature_cols,
    )
    max_evals = 2 * len(feature_cols) + 1
    global_values = explainer(X_explain, max_evals=max_evals)
    local_values = explainer(X_test.iloc[[0]], max_evals=max_evals)

importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_absolute_shap": np.abs(global_values.values).mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(REPORTS / "shap_global_importance.csv", index=False)

plt.figure()
shap.plots.bar(global_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIGURES / "shap_global_importance.png",
            dpi=160, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.beeswarm(global_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIGURES / "shap_summary_beeswarm.png",
            dpi=160, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.waterfall(local_values[0], max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIGURES / "shap_local_explanation.png",
            dpi=160, bbox_inches="tight")
plt.close()

local = pd.DataFrame({
    "feature": feature_cols,
    "feature_value": X_test.iloc[0].to_numpy(),
    "shap_value": local_values.values[0],
})
local["absolute_shap"] = local["shap_value"].abs()
local = local.sort_values("absolute_shap", ascending=False)
local.to_csv(REPORTS / "shap_local_explanation.csv", index=False)

print("\nTOP 15 GLOBAL FEATURES")
print(importance.head(15).round(5).to_string(index=False))

print("\nLOCAL EXPLANATION EXAMPLE")
print(f"Product series: {meta.iloc[0]['id']}")
print(f"Date: {meta.iloc[0]['date'].date()}")
print(f"Actual sales: {meta.iloc[0]['units_sold']}")
print(f"Model prediction: {model.predict(X_test.iloc[[0]])[0]:.4f}")
print(local.head(10)[
    ["feature", "feature_value", "shap_value"]
].round(5).to_string(index=False))

print("\nSaved SHAP reports and plots under reports/ and reports/figures/.")
print("SHAP values describe model contributions, not causal effects.")
