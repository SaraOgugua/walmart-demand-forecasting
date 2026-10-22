
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "reports").exists():
    ROOT = ROOT.parent

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Load the saved model and source data
bundle = joblib.load(REPORTS / "best_demand_model.joblib")
model = bundle["model"]
feature_columns = bundle["feature_columns"]

df = pd.read_csv(ROOT / "data/processed/sales_calendar_sample.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id", "date"]).copy()

# Reconstruct the model's lag and rolling features
grouped = df.groupby("id")["units_sold"]

for lag in [7, 14, 28]:
    df[f"lag_{lag}"] = grouped.shift(lag)

for window in [7, 28]:
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
    subset=["lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_28"]
)

# Match the saved model's exact feature schema
X = pd.get_dummies(
    df,
    columns=["item_id", "event_name_1", "event_type_1"],
    dtype=int
)
X = X.reindex(columns=feature_columns, fill_value=0)

# Select a real observation from the saved test predictions
predictions = pd.read_csv(REPORTS / "test_predictions.csv")
predictions["date"] = pd.to_datetime(predictions["date"])
test_row = predictions.sort_values(["date", "id"]).iloc[0]

match = (
    (df["id"] == test_row["id"]) &
    (df["date"] == test_row["date"])
)
positions = np.flatnonzero(match.to_numpy())

if len(positions) != 1:
    raise ValueError("Could not uniquely match the selected test observation.")

base_features = X.iloc[[positions[0]]].copy()
base_prediction = float(model.predict(base_features)[0])

# One-feature-at-a-time sensitivity scenarios
results = [{
    "feature_changed": "None (reference)",
    "scenario": "Original",
    "feature_value": np.nan,
    "predicted_units": base_prediction,
    "change_from_reference": 0.0,
    "change_percent": 0.0
}]

for feature in ["lag_7", "rolling_mean_7", "rolling_mean_28"]:
    original_value = float(base_features.iloc[0][feature])

    for factor, label in [(0.8, "-20%"), (1.2, "+20%")]:
        scenario = base_features.copy()
        changed_value = max(0.0, original_value * factor)
        scenario.loc[:, feature] = changed_value
        prediction = float(model.predict(scenario)[0])

        results.append({
            "feature_changed": feature,
            "scenario": label,
            "feature_value": changed_value,
            "predicted_units": prediction,
            "change_from_reference": prediction - base_prediction,
            "change_percent": (
                (prediction - base_prediction) / abs(base_prediction) * 100
                if base_prediction != 0 else np.nan
            )
        })

results_df = pd.DataFrame(results)
results_df.insert(0, "test_id", test_row["id"])
results_df.insert(1, "test_date", test_row["date"].date().isoformat())

output_path = REPORTS / "sensitivity_analysis.csv"
results_df.to_csv(output_path, index=False)

print("Selected test observation:", test_row["id"], test_row["date"].date())
print("Actual units sold:", test_row["units_sold"])
print("Saved-model prediction:", round(base_prediction, 3))
print("\nSensitivity results:")
print(results_df.to_string(index=False))
print("\nSaved:", output_path)

# Visualize scenario predictions
plot_df = results_df.iloc[1:].copy()
plot_df["label"] = (
    plot_df["feature_changed"] + " (" + plot_df["scenario"] + ")"
)

plt.figure(figsize=(10, 5))
plt.bar(plot_df["label"], plot_df["predicted_units"])
plt.axhline(base_prediction, linestyle="--",
            label="Original prediction")
plt.ylabel("Predicted units sold")
plt.title("One-Feature-at-a-Time Sensitivity Analysis")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()

figure_path = FIGURES / "sensitivity_analysis.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved figure:", figure_path)
