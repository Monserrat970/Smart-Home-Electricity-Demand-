
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -------------------------------------------------
# 1. FILE PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed_hourly.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "svr_model.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "test_predictions.csv"

# -------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------

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

X = df[features]
y = df[target]

# -------------------------------------------------
# 3. CHRONOLOGICAL SPLIT
# -------------------------------------------------

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

# Leave one observation between training and
# validation, and between validation and testing.
# Each target represents the following hour.

X_train = X.iloc[:train_end - 1]
y_train = y.iloc[:train_end - 1]

X_val = X.iloc[train_end:val_end - 1]
y_val = y.iloc[train_end:val_end - 1]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\n--- DATASET SPLIT ---")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# -------------------------------------------------
# 4. EVALUATION FUNCTION
# -------------------------------------------------

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"R2:   {r2:.4f}")

    return mae, rmse, r2


# -------------------------------------------------
# 5. BASELINE MODEL
# -------------------------------------------------

baseline_value = y_train.mean()

baseline_predictions = np.full(
    len(y_val),
    baseline_value
)

print("\n--- VALIDATION RESULTS ---")

evaluate_model(
    "Training-mean baseline",
    y_val,
    baseline_predictions
)

# -------------------------------------------------
# 6. DEFAULT SVR
# -------------------------------------------------

def create_model(C=1.0, epsilon=0.1, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(
            kernel="rbf",
            C=C,
            epsilon=epsilon,
            gamma=gamma
        ))
    ])


default_model = create_model()
default_model.fit(X_train, y_train)

default_predictions = default_model.predict(X_val)

evaluate_model(
    "Default SVR",
    y_val,
    default_predictions
)

# -------------------------------------------------
# 7. SIMPLE HYPERPARAMETER TUNING
# -------------------------------------------------

best_mae = float("inf")
best_params = None

for C in [0.1, 1, 10]:
    for epsilon in [0.05, 0.1, 0.2]:

        model = create_model(C=C, epsilon=epsilon)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)

        if mae < best_mae:
            best_mae = mae
            best_params = {
                "C": C,
                "epsilon": epsilon
            }

print("\n--- BEST HYPERPARAMETERS ---")
print(best_params)
print(f"Best validation MAE: {best_mae:.4f} kW")

# -------------------------------------------------
# 8. FINAL MODEL TRAINING
# -------------------------------------------------

# Refit using training + validation history.
# Exclude the boundary row whose target falls
# in the test period.

X_final_train = X.iloc[:val_end - 1]
y_final_train = y.iloc[:val_end - 1]

final_model = create_model(**best_params)
final_model.fit(X_final_train, y_final_train)

# -------------------------------------------------
# 9. FINAL TEST EVALUATION
# -------------------------------------------------

test_predictions = final_model.predict(X_test)

final_baseline_value = y_final_train.mean()
test_baseline_predictions = np.full(
    len(y_test),
    final_baseline_value
)

print("\n--- FINAL TEST RESULTS ---")

evaluate_model(
    "Training-mean baseline",
    y_test,
    test_baseline_predictions
)

evaluate_model(
    "Tuned SVR",
    y_test,
    test_predictions
)

# -------------------------------------------------
# 10. SAVE MODEL AND PREDICTIONS
# -------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(final_model, MODEL_PATH)

results = pd.DataFrame({
    "datetime": df.loc[X_test.index, "datetime"],
    "actual_kw": y_test.to_numpy(),
    "predicted_kw": test_predictions,
    "baseline_kw": test_baseline_predictions
})

results.to_csv(PREDICTIONS_PATH, index=False)

print("\nModel saved to:", MODEL_PATH)
print("Predictions saved to:", PREDICTIONS_PATH)
print("\nTraining and evaluation complete!")
