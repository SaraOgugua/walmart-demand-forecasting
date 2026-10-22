
from pathlib import Path
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PROJECT PATHS AND OUTPUT DIRECTORIES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/sales_calendar_sample.csv"
OUT = ROOT / "reports"
FIG = OUT / "figures"
TRACKING_DB = ROOT / "mlflow.db"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Required dataset not found: {DATA}")


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("Loading modelling dataset...", flush=True)

df = pd.read_csv(DATA, parse_dates=["date"])

required_columns = [
    "id",
    "item_id",
    "date",
    "units_sold",
    "event_name_1",
    "event_type_1",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df.duplicated(["id", "date"]).any():
    raise ValueError("Duplicate product-series/date records detected.")

if df["units_sold"].isna().any():
    raise ValueError("Missing target values detected.")

if (df["units_sold"] < 0).any():
    raise ValueError("Negative sales values detected.")

df = df.sort_values(["id", "date"]).reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(f"Product series: {df['id'].nunique():,}")
print(
    f"Date range: {df['date'].min().date()} to "
    f"{df['date'].max().date()}"
)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nCreating lag and calendar features...", flush=True)

grouped = df.groupby("id", sort=False)["units_sold"]

# Historical sales features: only information available before
# the target date is used.
df["lag_7"] = grouped.shift(7)
df["lag_14"] = grouped.shift(14)
df["lag_28"] = grouped.shift(28)

df["rolling_mean_7"] = grouped.transform(
    lambda series: series.shift(7).rolling(7).mean()
)

df["rolling_mean_28"] = grouped.transform(
    lambda series: series.shift(7).rolling(28).mean()
)

# Calendar variables are known for the target date.
df["weekday"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df["event_name_1"] = df["event_name_1"].fillna("NoEvent")
df["event_type_1"] = df["event_type_1"].fillna("NoEvent")

# Remove rows without enough prior sales history.
df = df.dropna(
    subset=[
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
    ]
).copy()

# Encode categorical variables into numeric indicator columns.
df = pd.get_dummies(
    df,
    columns=["item_id", "event_name_1", "event_type_1"],
    dtype=int,
)

categorical_features = [
    column for column in df.columns
    if column.startswith(
        ("item_id_", "event_name_1_", "event_type_1_")
    )
]

numeric_features = [
    "weekday",
    "month",
    "year",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

feature_cols = numeric_features + categorical_features

if not feature_cols:
    raise ValueError("No modelling features were created.")

print(f"Number of model features: {len(feature_cols)}")
print(f"Rows after feature preparation: {len(df):,}")


# ============================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# The last 28 calendar days are reserved for final testing.
# The preceding 28 days are reserved for model selection.
# Earlier observations are used for initial training.

dates = sorted(df["date"].unique())

if len(dates) < 100:
    raise ValueError("Insufficient dates for the planned time split.")

test_start = dates[-28]
validation_start = dates[-56]

train = df[df["date"] < validation_start].copy()

validation = df[
    (df["date"] >= validation_start)
    & (df["date"] < test_start)
].copy()

test = df[df["date"] >= test_start].copy()

train_validation = df[df["date"] < test_start].copy()

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("One or more chronological data splits are empty.")

X_train = train[feature_cols]
y_train = train["units_sold"]

X_validation = validation[feature_cols]
y_validation = validation["units_sold"]

X_train_validation = train_validation[feature_cols]
y_train_validation = train_validation["units_sold"]

X_test = test[feature_cols]
y_test = test["units_sold"]

print("\nCHRONOLOGICAL DATA SPLIT")
print(
    f"Training:   {train['date'].min().date()} to "
    f"{train['date'].max().date()} "
    f"({len(train):,} rows)"
)
print(
    f"Validation: {validation['date'].min().date()} to "
    f"{validation['date'].max().date()} "
    f"({len(validation):,} rows)"
)
print(
    f"Test:       {test['date'].min().date()} to "
    f"{test['date'].max().date()} "
    f"({len(test):,} rows)"
)


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    absolute_error = np.abs(actual - predicted)
    denominator = np.abs(actual).sum()

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "R2": float(r2_score(actual, predicted)),
        "WAPE_pct": float(
            100 * absolute_error.sum() / denominator
        ) if denominator > 0 else float("nan"),
    }


# ============================================================
# 6. CONFIGURE DATABASE-BACKED MLFLOW
# ============================================================

print("\nConfiguring MLflow...", flush=True)

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
mlflow.set_experiment("Walmart Retail Demand Forecasting")


# ============================================================
# 7. SEASONAL-NAIVE BASELINE
# ============================================================

# Predict each date using sales from the same product series
# seven days earlier. This is a rolling one-step-ahead baseline.

baseline_validation_predictions = np.maximum(
    validation["lag_7"].to_numpy(), 0
)

baseline_validation_metrics = calculate_metrics(
    y_validation,
    baseline_validation_predictions,
)

print("\nSEASONAL-NAIVE VALIDATION RESULTS")
print(baseline_validation_metrics)

with mlflow.start_run(run_name="Seasonal_Naive_Baseline"):
    mlflow.log_param("algorithm", "SeasonalNaive")
    mlflow.log_param("lag_days", 7)
    mlflow.log_param("evaluation_protocol", "chronological_holdout")
    mlflow.log_metrics({
        f"validation_{key}": value
        for key, value in baseline_validation_metrics.items()
        if np.isfinite(value)
    })


# ============================================================
# 8. MODEL CANDIDATES AND HYPERPARAMETER TUNING
# ============================================================

candidates = [
    (
        "RandomForest",
        {
            "n_estimators": 60,
            "max_depth": 10,
            "min_samples_leaf": 3,
        },
    ),
    (
        "RandomForest",
        {
            "n_estimators": 60,
            "max_depth": 16,
            "min_samples_leaf": 3,
        },
    ),
    (
        "HistGradientBoosting",
        {
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
        },
    ),
    (
        "HistGradientBoosting",
        {
            "max_iter": 150,
            "learning_rate": 0.08,
            "max_leaf_nodes": 15,
        },
    ),
]

results = []

for index, (algorithm, parameters) in enumerate(candidates, start=1):

    print(
        f"\nCandidate {index}/{len(candidates)}: "
        f"{algorithm} | {parameters}",
        flush=True,
    )

    if algorithm == "RandomForest":
        model = RandomForestRegressor(
            **parameters,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            **parameters,
            random_state=42,
        )

    with mlflow.start_run(
        run_name=f"{algorithm}_candidate_{index}"
    ):
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_params(parameters)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("forecast_horizon_days", 7)
        mlflow.log_param("lag_reference_days", 7)
        mlflow.log_param(
            "validation_start",
            str(validation["date"].min().date()),
        )
        mlflow.log_param(
            "validation_end",
            str(validation["date"].max().date()),
        )

        model.fit(X_train, y_train)

        validation_predictions = np.maximum(
            model.predict(X_validation), 0
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        mlflow.log_metrics({
            f"validation_{key}": value
            for key, value in validation_metrics.items()
            if np.isfinite(value)
        })

        # Use pickle serialization explicitly to avoid MLflow's
        # skops trusted-type audit for these locally trained models.
        # Only load model artifacts from sources you trust.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle",
        )

        results.append({
            "algorithm": algorithm,
            "parameters": parameters.copy(),
            "validation_metrics": validation_metrics,
        })

        print("Validation metrics:", validation_metrics)


# ============================================================
# 9. SELECT MODEL USING VALIDATION MAE
# ============================================================

best = min(
    results,
    key=lambda result: result["validation_metrics"]["MAE"],
)

print("\nSELECTED MODEL")
print("Algorithm:", best["algorithm"])
print("Parameters:", best["parameters"])
print("Validation metrics:", best["validation_metrics"])

if best["algorithm"] == "RandomForest":
    final_model = RandomForestRegressor(
        **best["parameters"],
        random_state=42,
        n_jobs=-1,
    )
else:
    final_model = HistGradientBoostingRegressor(
        **best["parameters"],
        random_state=42,
    )


# ============================================================
# 10. REFIT AND EVALUATE ON THE UNTOUCHED TEST PERIOD
# ============================================================

print("\nRefitting selected model on training + validation data...")

final_model.fit(X_train_validation, y_train_validation)

test_predictions = np.maximum(
    final_model.predict(X_test), 0
)

baseline_test_predictions = np.maximum(
    test["lag_7"].to_numpy(), 0
)

baseline_test_metrics = calculate_metrics(
    y_test,
    baseline_test_predictions,
)

model_test_metrics = calculate_metrics(
    y_test,
    test_predictions,
)

comparison = pd.DataFrame([
    {
        "model": "Seasonal-naive baseline",
        **baseline_test_metrics,
    },
    {
        "model": f"Selected {best['algorithm']}",
        **model_test_metrics,
    },
])

comparison.to_csv(
    OUT / "model_comparison.csv",
    index=False,
)

print("\nTEST-SET COMPARISON")
print(comparison.to_string(index=False))


# ============================================================
# 11. SAVE TEST PREDICTIONS AND RESIDUALS
# ============================================================

predictions = test[
    ["id", "store_id", "dept_id", "cat_id", "date", "units_sold"]
].copy()

# Recover item IDs from the original dataset using the stable series ID.
item_lookup = (
    pd.read_csv(DATA, usecols=["id", "item_id"])
    .drop_duplicates("id")
)

predictions = predictions.merge(
    item_lookup,
    on="id",
    how="left",
    validate="many_to_one",
)

predictions["baseline_prediction"] = baseline_test_predictions
predictions["model_prediction"] = test_predictions
predictions["residual"] = (
    predictions["units_sold"] - predictions["model_prediction"]
)

predictions.to_csv(
    OUT / "test_predictions.csv",
    index=False,
)

# Residual plot for model diagnostics.
plt.figure(figsize=(8, 5))
plt.scatter(
    test_predictions,
    predictions["residual"],
    alpha=0.25,
    s=10,
)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted units sold")
plt.ylabel("Residual (actual - predicted)")
plt.title("Selected Model: Test-Set Residuals")
plt.tight_layout()
plt.savefig(
    FIG / "residual_plot.png",
    dpi=160,
)
plt.close()

# Actual vs predicted plot.
plt.figure(figsize=(6, 6))
plt.scatter(
    y_test,
    test_predictions,
    alpha=0.25,
    s=10,
)
upper = max(float(np.max(y_test)), float(np.max(test_predictions)), 1)
plt.plot([0, upper], [0, upper], linestyle="--")
plt.xlabel("Actual units sold")
plt.ylabel("Predicted units sold")
plt.title("Actual vs Predicted Demand: Test Set")
plt.tight_layout()
plt.savefig(
    FIG / "actual_vs_predicted.png",
    dpi=160,
)
plt.close()


# ============================================================
# 12. SERIALIZE FINAL MODEL AND METADATA
# ============================================================

model_bundle = {
    "model": final_model,
    "feature_columns": feature_cols,
    "algorithm": best["algorithm"],
    "parameters": best["parameters"],
    "forecast_horizon_days": 7,
    "evaluation_protocol": (
        "Rolling one-step-ahead predictions using seven-day lags"
    ),
    "training_end": str(
        train_validation["date"].max().date()
    ),
    "test_start": str(test["date"].min().date()),
    "test_end": str(test["date"].max().date()),
}

model_path = OUT / "best_demand_model.joblib"
joblib.dump(model_bundle, model_path)


# ============================================================
# 13. TRACK FINAL MODEL AND TEST METRICS IN MLFLOW
# ============================================================

print("\nLogging final evaluation to MLflow...", flush=True)

with mlflow.start_run(
    run_name="Selected_Model_Final_Test"
) as final_run:

    mlflow.log_param("selected_algorithm", best["algorithm"])
    mlflow.log_params({
        f"selected_{key}": value
        for key, value in best["parameters"].items()
    })
    mlflow.log_param("random_state", 42)
    mlflow.log_param("forecast_horizon_days", 7)
    mlflow.log_param(
        "evaluation_protocol",
        "Rolling one-step-ahead with seven-day lags",
    )
    mlflow.log_param(
        "test_start",
        str(test["date"].min().date()),
    )
    mlflow.log_param(
        "test_end",
        str(test["date"].max().date()),
    )

    mlflow.log_metrics({
        f"test_{key}": value
        for key, value in model_test_metrics.items()
        if np.isfinite(value)
    })

    mlflow.log_metrics({
        f"baseline_test_{key}": value
        for key, value in baseline_test_metrics.items()
        if np.isfinite(value)
    })

    mlflow.log_artifact(
        str(OUT / "model_comparison.csv")
    )
    mlflow.log_artifact(
        str(OUT / "test_predictions.csv")
    )
    mlflow.log_artifact(
        str(FIG / "residual_plot.png")
    )
    mlflow.log_artifact(
        str(FIG / "actual_vs_predicted.png")
    )
    mlflow.log_artifact(str(model_path))

    # Explicit pickle format avoids skops trusted-type errors.
    final_model_info = mlflow.sklearn.log_model(
        sk_model=final_model,
        name="model",
        serialization_format="pickle",
    )

    final_model_uri = final_model_info.model_uri
    final_run_id = final_run.info.run_id


# ============================================================
# 14. ATTEMPT MODEL REGISTRY REGISTRATION
# ============================================================

print("\nAttempting MLflow model registration...", flush=True)

try:
    registered_model = mlflow.register_model(
        model_uri=final_model_uri,
        name="WalmartRetailDemandForecast",
    )

    print(
        "Registered model version:",
        registered_model.version,
    )

    # Stages are retained here for compatibility with registry
    # workflows that support the legacy Staging stage.
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="WalmartRetailDemandForecast",
            version=registered_model.version,
            stage="Staging",
            archive_existing_versions=True,
        )
        print("Registered model stage: Staging")
    except Exception as stage_error:
        print(
            "Model registered, but stage transition was not completed:",
            stage_error,
        )

except Exception as registry_error:
    print(
        "MODEL REGISTRY WARNING:",
        registry_error,
    )
    print(
        "The model, test metrics, and experiment artifacts "
        "were saved separately."
    )


# ============================================================
# 15. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL DEVELOPMENT RUN FINISHED")
print("=" * 60)

print("\nTest-set comparison:")
print(comparison.to_string(index=False))

print("\nCreated files:")
print("Model bundle:       ", model_path)
print("Model comparison:   ", OUT / "model_comparison.csv")
print("Test predictions:   ", OUT / "test_predictions.csv")
print("Residual plot:      ", FIG / "residual_plot.png")
print("Actual vs predicted:", FIG / "actual_vs_predicted.png")
print("MLflow database:    ", TRACKING_DB)
print("Final MLflow run ID:", final_run_id)

print("\nReview the metrics before drawing conclusions.")
