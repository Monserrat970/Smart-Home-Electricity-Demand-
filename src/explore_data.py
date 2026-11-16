
from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "HomeC.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

# Display basic information
print("\n--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Convert Unix timestamps into readable dates
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df = df.dropna(subset=["time"])

df["datetime"] = pd.to_datetime(
    df["time"], unit="s", utc=True
)

# Sort observations chronologically
df = df.sort_values("datetime")

print("\n--- DATE RANGE (UTC) ---")
print("Start:", df["datetime"].min())
print("End:", df["datetime"].max())

print("\n--- HOUSEHOLD DEMAND STATISTICS ---")
print(df["House overall [kW]"].describe())

# Inspect timestamp spacing
print("\n--- SAMPLING INTERVALS ---")
print(df["datetime"].diff().value_counts().head())

print("\nDataset exploration complete!")
