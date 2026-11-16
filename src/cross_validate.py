
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# 1. FILE PATHS
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed_hourly.csv"

RESULTS_PATH = (
    PROJECT_ROOT / "documents" / "cross_validation_results.csv"
)

# ------------------------------------------
# 2. LOAD DATA
# ------------------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

features = [
    "temperature",
    "humidity",
    "windSpeed",
    "pressure",
    "hour",
    "day_of_week"
]

target = "target_next_hour_kw"

# Reserve the same final test period as before.
test_start = int(len(df) * 0.85)

development_df = df.iloc[:test_start].copy()

X = development_df[features]
y = development_df[target]

print("Development observations:", len(X))
print("Reserved test observations:", len(df) - test_start)

# ------------------------------------------
# 3. TIME-SERIES CROSS-VALIDATION
# ------------------------------------------

tscv = TimeSeriesSplit(
    n_splits=3,
    test_size=20,
    gap=1
)

results = []

# ------------------------------------------
# 4. TRAIN AND EVALUATE EACH FOLD
# ------------------------------------------

for fold, (train_idx, val_idx) in enumerate(
    tscv.split(X), start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Mean baseline
    baseline_predictions = np.full(
        len(y_val),
        y_train.mean()
    )

    # SVR pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(
            kernel="rbf",
            C=0.1,
            epsilon=0.05,
            gamma="scale"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    baseline_mae = mean_absolute_error(
        y_val, baseline_predictions
    )

    svr_mae = mean_absolute_error(
        y_val, predictions
    )

    svr_rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    svr_r2 = r2_score(y_val, predictions)

    results.append({
        "fold": fold,
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "baseline_mae": baseline_mae,
        "svr_mae": svr_mae,
        "svr_rmse": svr_rmse,
        "svr_r2": svr_r2
    })

    print(f"\n--- FOLD {fold} ---")
    print("Training observations:", len(X_train))
    print("Validation observations:", len(X_val))
    print(f"Baseline MAE: {baseline_mae:.4f} kW")
    print(f"SVR MAE:      {svr_mae:.4f} kW")
    print(f"SVR RMSE:     {svr_rmse:.4f} kW")
    print(f"SVR R2:       {svr_r2:.4f}")

# ------------------------------------------
# 5. SUMMARIZE RESULTS
# ------------------------------------------

results_df = pd.DataFrame(results)

print("\n--- CROSS-VALIDATION SUMMARY ---")
print(
    results_df[
        ["baseline_mae", "svr_mae", "svr_rmse", "svr_r2"]
    ].mean().round(4)
)

print("\n--- STANDARD DEVIATION ---")
print(
    results_df[
        ["baseline_mae", "svr_mae", "svr_rmse", "svr_r2"]
    ].std().round(4)
)

# ------------------------------------------
# 6. SAVE RESULTS
# ------------------------------------------

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(RESULTS_PATH, index=False)

print("\nResults saved to:", RESULTS_PATH)
