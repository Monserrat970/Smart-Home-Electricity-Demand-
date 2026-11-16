
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "test_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "documents" / "svr_predictions.png"

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

plt.figure(figsize=(12, 6))

plt.plot(
    df["datetime"],
    df["actual_kw"],
    label="Actual demand",
    marker="o",
    linewidth=2
)

plt.plot(
    df["datetime"],
    df["predicted_kw"],
    label="SVR prediction",
    marker="s",
    linestyle="--"
)

plt.plot(
    df["datetime"],
    df["baseline_kw"],
    label="Mean baseline",
    linestyle=":"
)

plt.title("Actual vs Predicted Household Electricity Demand")
plt.xlabel("Time (UTC)")
plt.ylabel("Average electricity demand (kW)")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print("Graph saved to:", OUTPUT_PATH)
