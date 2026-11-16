
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Electricity Demand Prediction",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Weather-Based Electricity Demand Prediction")

st.write(
    "Evaluating Support Vector Regression (SVR) "
    "for one-hour-ahead household electricity "
    "demand forecasting."
)

# ------------------------------------------
# 2. FILE PATHS
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "processed_hourly.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "svr_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "data" / "test_predictions.csv"

FEATURES = [
    "temperature",
    "humidity",
    "windSpeed",
    "pressure",
    "hour",
    "day_of_week"
]

# ------------------------------------------
# 3. LOAD MODEL AND DATA
# ------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime"]
    )

    results = pd.read_csv(
        RESULTS_PATH,
        parse_dates=["datetime"]
    )

    return data, results


model = load_model()
data, results = load_data()

# ------------------------------------------
# 4. HISTORICAL OBSERVATION SELECTION
# ------------------------------------------

st.header("1. Select a Historical Observation")

st.write(
    "Select an observation from the held-out "
    "test period to examine the model's "
    "one-hour-ahead prediction."
)

# Only show observations from the test period
test_data = data[
    data["datetime"].isin(results["datetime"])
].copy()

test_data = test_data.sort_values("datetime")
test_data = test_data.reset_index(drop=True)

selected_index = st.slider(
    "Select observation",
    min_value=0,
    max_value=len(test_data) - 1,
    value=0
)

selected_row = test_data.iloc[selected_index]

selected_time = selected_row["datetime"]

st.write("Selected observation:", selected_time)

# ------------------------------------------
# 5. DISPLAY WEATHER CONDITIONS
# ------------------------------------------

st.header("2. Weather Conditions")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Temperature",
    f"{selected_row['temperature']:.1f} °F"
)

col2.metric(
    "Humidity",
    f"{selected_row['humidity'] * 100:.1f}%"
)

col3.metric(
    "Wind Speed",
    f"{selected_row['windSpeed']:.2f}"
)

col4.metric(
    "Atmospheric Pressure",
    f"{selected_row['pressure']:.2f}"
)

# ------------------------------------------
# 6. GENERATE SVR PREDICTION
# ------------------------------------------

st.header("3. Electricity Demand Prediction")

input_features = pd.DataFrame(
    [selected_row[FEATURES].to_dict()],
    columns=FEATURES
)

prediction = model.predict(input_features)[0]

actual = selected_row["target_next_hour_kw"]

absolute_error = abs(actual - prediction)

col1, col2, col3 = st.columns(3)

col1.metric(
    "SVR Prediction",
    f"{prediction:.3f} kW"
)

col2.metric(
    "Actual Demand",
    f"{actual:.3f} kW"
)

col3.metric(
    "Absolute Error",
    f"{absolute_error:.3f} kW"
)

st.info(
    "The prediction represents the average "
    "electricity demand during the hour "
    "following the selected observation."
)

# ------------------------------------------
# 7. MODEL PERFORMANCE
# ------------------------------------------

st.header("4. Model Performance")

mae = np.mean(
    np.abs(
        results["actual_kw"] -
        results["predicted_kw"]
    )
)

baseline_mae = np.mean(
    np.abs(
        results["actual_kw"] -
        results["baseline_kw"]
    )
)

col1, col2 = st.columns(2)

col1.metric("SVR Test MAE", f"{mae:.4f} kW")
col2.metric(
    "Baseline Test MAE",
    f"{baseline_mae:.4f} kW"
)

st.subheader("Actual vs. Predicted Demand")

chart_data = results.set_index("datetime")[
    ["actual_kw", "predicted_kw", "baseline_kw"]
]

st.line_chart(chart_data)

# ------------------------------------------
# 8. PROJECT INTERPRETATION
# ------------------------------------------

st.header("5. Interpretation")

st.write(
    "This project investigates whether "
    "Support Vector Regression can predict "
    "residential electricity demand using "
    "weather and time-related features."
)

st.write(
    "The model is evaluated against a "
    "training-mean baseline to determine "
    "whether SVR provides meaningful "
    "predictive improvement."
)

st.warning(
    "This model was trained using approximately "
    "six days of historical winter data. "
    "Its predictions should be interpreted "
    "as a limited research demonstration, "
    "not as reliable real-world forecasts."
)
