
from pathlib import Path
import pandas as pd

# 1. Define file locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "HomeC.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed_hourly.csv"

# 2. Load only the columns needed for our project
columns = [
    "time",
    "House overall [kW]",
    "temperature",
    "humidity",
    "windSpeed",
    "pressure"
]

df = pd.read_csv(
    INPUT_PATH,
    usecols=columns,
    low_memory=False
)

print("Original dataset shape:", df.shape)

# 3. Convert timestamps and numerical columns
df["time"] = pd.to_numeric(df["time"], errors="coerce")

for column in columns[1:]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows without a valid timestamp or target
df = df.dropna(subset=["time", "House overall [kW]"])

df["datetime"] = pd.to_datetime(
    df["time"],
    unit="s",
    utc=True
)

df = df.sort_values("datetime")
df = df.set_index("datetime")

# 4. Convert second-level measurements to hourly averages
measurement_columns = columns[1:]

hourly = df[measurement_columns].resample("1h").mean()

# Count valid power readings per hour
hourly["reading_count"] = (
    df["House overall [kW]"]
    .resample("1h")
    .count()
)

# Keep hours with at least 3500 valid readings
hourly = hourly[hourly["reading_count"] >= 3500].copy()
hourly = hourly.drop(columns=["reading_count"])

# Remove hours with missing required features
hourly = hourly.dropna()

# 5. Create time-related features
hourly["hour"] = hourly.index.hour
hourly["day_of_week"] = hourly.index.dayofweek

# 6. Create the next-hour prediction target
hourly["target_next_hour_kw"] = (
    hourly["House overall [kW]"].shift(-1)
)

# Remove rows without a consecutive following hour
next_timestamp = hourly.index.to_series().shift(-1)

consecutive = (
    next_timestamp - hourly.index.to_series()
    == pd.Timedelta(hours=1)
)

hourly = hourly.loc[consecutive].copy()

# Remove the current-hour power column from model inputs
# This first experiment uses weather and time features only.
hourly = hourly.drop(columns=["House overall [kW]"])

# 7. Save the processed dataset
hourly.to_csv(OUTPUT_PATH, index_label="datetime")

print("\n--- PROCESSED DATASET ---")
print("Shape:", hourly.shape)

print("\n--- FIRST FIVE ROWS ---")
print(hourly.head())

print("\n--- MISSING VALUES ---")
print(hourly.isnull().sum())

print("\n--- DATE RANGE ---")
print("Start:", hourly.index.min())
print("End:", hourly.index.max())

print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)
