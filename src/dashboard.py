
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# 1. Paths and page configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA_PATH = ROOT / "data" / "processed" / "sales_calendar_sample.csv"
MODEL_PATH = REPORTS / "best_demand_model.joblib"
COMPARISON_PATH = REPORTS / "model_comparison.csv"
PREDICTIONS_PATH = REPORTS / "test_predictions.csv"

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
)

st.title("Retail Demand Forecasting")
st.caption(
    "Stakeholder decision-support dashboard | "
    "Walmart business context | Public M5 sample data"
)

st.warning(
    "Experimental prototype. Predictions are estimates, not guarantees. "
    "This dashboard is not a production inventory system and does not "
    "automatically recommend or place orders."
)

# --------------------------------------------------
# 2. Load existing project artifacts
# --------------------------------------------------

@st.cache_data
def load_reports():
    comparison = pd.read_csv(COMPARISON_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    return comparison, predictions


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


required_paths = [
    COMPARISON_PATH,
    PREDICTIONS_PATH,
    MODEL_PATH,
]

missing = [str(p.relative_to(ROOT)) for p in required_paths if not p.exists()]

if missing:
    st.error("Required project files are missing: " + ", ".join(missing))
    st.stop()

comparison, predictions = load_reports()
bundle = load_model()
model = bundle["model"]
feature_columns = bundle["feature_columns"]

predictions = predictions.copy()
predictions["date"] = pd.to_datetime(predictions["date"])
predictions = predictions.sort_values("date")

# --------------------------------------------------
# 3. Model performance
# --------------------------------------------------

st.header("1. Model performance")

baseline_rows = comparison[
    comparison["model"].str.contains("baseline", case=False, na=False)
]
selected_rows = comparison[
    comparison["model"].str.contains("Selected", case=False, na=False)
]

if baseline_rows.empty or selected_rows.empty:
    st.error(
        "Could not identify the baseline and selected model in "
        "model_comparison.csv. Check the model labels."
    )
    st.stop()

baseline = baseline_rows.iloc[0]
selected = selected_rows.iloc[0]

mae_improvement = (
    (baseline["MAE"] - selected["MAE"]) / baseline["MAE"] * 100
)
rmse_improvement = (
    (baseline["RMSE"] - selected["RMSE"]) / baseline["RMSE"] * 100
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model MAE", f"{selected['MAE']:.3f}")
c2.metric("Selected model RMSE", f"{selected['RMSE']:.3f}")
c3.metric("MAE improvement", f"{mae_improvement:.1f}%")
c4.metric("Selected model WAPE", f"{selected['WAPE_pct']:.2f}%")

st.markdown(
    f"The selected model ({selected['model']}) reduced test MAE by "
    f"**{mae_improvement:.1f}%** and RMSE by **{rmse_improvement:.1f}%** "
    "relative to the seasonal-naive baseline. However, WAPE remains high, "
    "so prediction errors are still material."
)

metric_data = comparison.melt(
    id_vars="model",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Error",
)

fig_metrics = px.bar(
    metric_data,
    x="Metric",
    y="Error",
    color="model",
    barmode="group",
    title="Test-set error comparison (lower is better)",
    labels={"model": "Model"},
)
st.plotly_chart(fig_metrics, use_container_width=True)

with st.expander("View all evaluation metrics"):
    st.dataframe(comparison, use_container_width=True)

# --------------------------------------------------
# 4. Actual versus predicted sales
# --------------------------------------------------

st.header("2. Actual versus predicted demand")

left, right = st.columns(2)

with left:
    fig_scatter = px.scatter(
        predictions,
        x="units_sold",
        y="model_prediction",
        hover_data=["item_id", "store_id", "date"],
        labels={
            "units_sold": "Actual units sold",
            "model_prediction": "Predicted units sold",
        },
        title="Actual and predicted values",
        opacity=0.55,
    )

    upper = max(
        float(predictions["units_sold"].max()),
        float(predictions["model_prediction"].max()),
        1,
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, upper],
            y=[0, upper],
            mode="lines",
            name="Perfect prediction reference",
            line={"dash": "dash"},
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with right:
    daily = (
        predictions.groupby("date", as_index=False)
        .agg(
            Actual=("units_sold", "sum"),
            Predicted=("model_prediction", "sum"),
            Baseline=("baseline_prediction", "sum"),
        )
    )

    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["Actual"],
            mode="lines",
            name="Actual",
        )
    )
    fig_daily.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["Predicted"],
            mode="lines",
            name="Model prediction",
        )
    )
    fig_daily.update_layout(
        title="Daily total across the evaluation sample",
        xaxis_title="Date",
        yaxis_title="Units sold",
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.caption(
    "These charts describe the held-out test period in the saved report; "
    "they are not evidence of future live forecasting performance."
)

# --------------------------------------------------
# 5. Interactive example predictions
# --------------------------------------------------

st.header("3. Explore example predictions")

items = sorted(predictions["item_id"].dropna().unique().tolist())

if not items:
    st.info("No product IDs were found in the prediction report.")
else:
    chosen_item = st.selectbox("Select a product", items)

    item_data = predictions[
        predictions["item_id"] == chosen_item
    ].sort_values("date")

    chosen_date = st.selectbox(
        "Select a test date",
        item_data["date"].dt.strftime("%Y-%m-%d").tolist(),
    )

    example = item_data[
        item_data["date"].dt.strftime("%Y-%m-%d") == chosen_date
    ].iloc[0]

    p1, p2, p3 = st.columns(3)
    p1.metric("Actual units sold", f"{example['units_sold']:.0f}")
    p2.metric("Model estimate", f"{example['model_prediction']:.3f}")
    p3.metric("Baseline estimate", f"{example['baseline_prediction']:.3f}")

    st.write(
        f"For product **{chosen_item}** on **{chosen_date}**, "
        f"the actual sales were {example['units_sold']:.0f} units, "
        f"while the selected model estimated "
        f"{example['model_prediction']:.2f} units."
    )

    st.caption(
        "A prediction is an estimate, not a guarantee. "
        "Review it alongside recent sales and relevant business context."
    )

# --------------------------------------------------
# 6. Model-based what-if sensitivity analysis
# --------------------------------------------------

st.header("4. What-if analysis: sensitivity to recent demand")

st.write(
    "Select a product and test date, then vary its 28-day rolling mean. "
    "The model is rerun with the selected feature changed while all other "
    "features are held constant. This tests model sensitivity; it is not "
    "a causal estimate or a guaranteed business outcome."
)

if DATA_PATH.exists():
    @st.cache_data
    def prepare_features(data_path, expected_features):
        df = pd.read_csv(data_path, parse_dates=["date"])
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
                "lag_7",
                "lag_14",
                "lag_28",
                "rolling_mean_7",
                "rolling_mean_28",
            ]
        ).copy()

        df = pd.get_dummies(
            df,
            columns=["item_id", "event_name_1", "event_type_1"],
            dtype=int,
        )

        # Match the model's training schema exactly.
        X = df.reindex(columns=expected_features, fill_value=0)
        return df, X

    try:
        feature_df, X_all = prepare_features(
            str(DATA_PATH), feature_columns
        )

        # Match the product/date from the held-out prediction report
        # back to the prepared source data.
        lookup = feature_df[
            ["id", "date", "units_sold"]
        ].copy()

        item_date_rows = predictions[
            predictions["item_id"] == chosen_item
        ].copy()

        matched = item_date_rows[
            item_date_rows["date"] == pd.Timestamp(chosen_date)
        ]

        if not matched.empty:
            target_id = matched.iloc[0]["id"]
            row_indices = feature_df.index[
                (feature_df["id"] == target_id)
                & (feature_df["date"] == pd.Timestamp(chosen_date))
            ]

            if len(row_indices) == 1:
                source_idx = row_indices[0]
                baseline_features = X_all.loc[[source_idx]].copy()
                original_value = float(
                    baseline_features.iloc[0]["rolling_mean_28"]
                )

                change_pct = st.slider(
                    "Change in 28-day rolling mean",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5,
                    format="%d%%",
                )

                scenario_features = baseline_features.copy()
                scenario_features.loc[
                    scenario_features.index[0], "rolling_mean_28"
                ] = max(0.0, original_value * (1 + change_pct / 100))

                original_prediction = max(
                    0.0,
                    float(model.predict(baseline_features)[0]),
                )
                scenario_prediction = max(
                    0.0,
                    float(model.predict(scenario_features)[0]),
                )

                w1, w2, w3 = st.columns(3)
                w1.metric(
                    "Original 28-day mean",
                    f"{original_value:.3f}",
                )
                w2.metric(
                    "Original model estimate",
                    f"{original_prediction:.3f}",
                )
                w3.metric(
                    "What-if estimate",
                    f"{scenario_prediction:.3f}",
                    delta=f"{scenario_prediction - original_prediction:+.3f}",
                )

                st.info(
                    "Interpret this as a controlled sensitivity test of one "
                    "model input. Changing one feature independently may "
                    "create an input combination that does not reflect real "
                    "sales history."
                )
            else:
                st.warning(
                    "Could not uniquely match the selected product/date "
                    "to the prepared source data."
                )
        else:
            st.warning("No matching product/date row was found.")
    except Exception as exc:
        st.error(
            "The what-if analysis could not be prepared. "
            "Check the processed dataset and model feature schema."
        )
        st.caption(str(exc))
else:
    st.warning(
        "The processed source dataset is missing. The dashboard can still "
        "display saved test results, but model-based what-if analysis "
        "requires data/processed/sales_calendar_sample.csv."
    )

# --------------------------------------------------
# 7. Operational performance differences
# --------------------------------------------------

st.header("5. Differences across demand levels")

st.write(
    "The following exploratory grouping compares prediction errors for "
    "products with lower, medium, and higher average actual sales in the "
    "test period. These are operational demand segments, not demographic "
    "groups or a formal demographic fairness test."
)

segment_data = predictions.copy()
product_means = (
    segment_data.groupby("item_id")["units_sold"]
    .mean()
    .rename("mean_actual_demand")
)
segment_data = segment_data.merge(
    product_means,
    on="item_id",
    how="left",
)

try:
    segment_data["demand_segment"] = pd.qcut(
        segment_data["mean_actual_demand"],
        q=3,
        labels=["Lower", "Middle", "Higher"],
        duplicates="drop",
    )

    segment_summary = (
        segment_data.groupby("demand_segment", observed=True)
        .apply(
            lambda g: pd.Series({
                "Product count": g["item_id"].nunique(),
                "MAE": np.mean(
                    np.abs(g["units_sold"] - g["model_prediction"])
                ),
                "WAPE (%)": (
                    100 * np.abs(
                        g["units_sold"] - g["model_prediction"]
                    ).sum() / g["units_sold"].abs().sum()
                    if g["units_sold"].abs().sum() > 0
                    else np.nan
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    st.dataframe(segment_summary, use_container_width=True)

    segment_chart = px.bar(
        segment_summary,
        x="demand_segment",
        y="MAE",
        title="Model MAE by exploratory demand segment",
        labels={
            "demand_segment": "Demand segment",
            "MAE": "Mean absolute error (units)",
        },
    )
    st.plotly_chart(segment_chart, use_container_width=True)

except ValueError:
    st.info(
        "There were not enough distinct product demand levels to form "
        "three exploratory groups."
    )

st.warning(
    "This is not evidence of demographic fairness. The dataset does not "
    "provide suitable person-level protected attributes for testing "
    "demographic parity, equalized odds, or disparate impact. "
    "Operational segment metrics reveal where error differs, not why."
)

# --------------------------------------------------
# 8. Transparency, limitations, and data context
# --------------------------------------------------

st.header("6. Transparency and responsible use")

with st.expander("How the model works"):
    st.write(
        "The selected HistGradientBoosting model learns relationships "
        "between historical sales, rolling averages, calendar variables, "
        "product indicators, and event indicators. It estimates sales for "
        "a target date using historical information available before that "
        "date."
    )

with st.expander("Evaluation protocol"):
    st.write(
        "Results come from a chronological held-out test period. "
        "The reported task is a rolling one-step-ahead estimate using "
        "seven-day lag features; it should not be described as a direct "
        "seven-day-ahead forecast."
    )
    st.write(
        f"Test dates in the saved report: "
        f"{predictions['date'].min().date()} to "
        f"{predictions['date'].max().date()}."
    )

with st.expander("Known limitations"):
    st.markdown(
        """
        - The evaluation uses a restricted public-data sample and does not
          represent all Walmart stores or products.
        - The selected model's WAPE is still high, indicating material
          error relative to total actual sales.
        - Intermittent and low-volume products can be difficult to predict.
        - Operational demand-segment comparisons are not demographic
          fairness assessments.
        - Sensitivity analysis does not establish causal relationships.
        - The prototype has not been established as production-ready.
        """
    )

with st.expander("Data and intended use"):
    st.write(
        "Organizational context: Walmart. Dataset: public M5 "
        "Forecasting–Accuracy data. This is an academic decision-support "
        "prototype, not a Walmart-operated or Walmart-deployed system."
    )
    st.write(
        "A human decision-maker must review estimates alongside current "
        "inventory, promotions, supplier constraints, and local business "
        "conditions before making operational decisions."
    )

st.divider()
st.caption(
    "BAN6800 Capstone | Stakeholder decision-support prototype | "
    "Metrics are based on the saved model evaluation artifacts."
)
