# %%
import pandas as pd
from glob import glob


# -----------------------------------------
# 1. Function to read and preprocess sensor data
# -----------------------------------------
def read_data_from_files(files):
    data_path = "../../data/raw/MetaMotion/"

    acc_df = pd.DataFrame()  # DataFrame for accelerometer data
    gyr_df = pd.DataFrame()  # DataFrame for gyroscope data

    acc_set = 1  # Counter for accelerometer set
    gyr_set = 1  # Counter for gyroscope set

    for f in files:
        # Extract participant, label, and category info from filename
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)  # Load CSV file into DataFrame

        # Add metadata columns
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Append to appropriate DataFrame
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_df = pd.concat([acc_df, df])
            acc_set += 1

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_df = pd.concat([gyr_df, df])
            gyr_set += 1

    # Convert epoch time to datetime and set as index
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Drop unnecessary columns
    del acc_df["time (01:00)"]
    del acc_df["epoch (ms)"]
    del acc_df["elapsed (s)"]

    del gyr_df["time (01:00)"]
    del gyr_df["epoch (ms)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


# -----------------------------------------
# 2. Load all CSV files and extract data
# -----------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
acc_df, gyr_df = read_data_from_files(files)

# -----------------------------------------
# 3. Merge accelerometer and gyroscope data
# -----------------------------------------
df_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
df_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# Define which columns are numeric for aggregation
numeric_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# -----------------------------------------
# 4. Define sampling strategy for resampling
# -----------------------------------------
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

# NOTE: This line does nothing as the result is not saved
df_merged.resample(rule="200ms").apply(sampling)

# -----------------------------------------
# 5. Split merged data by day
# -----------------------------------------
days = [g for n, g in df_merged.groupby(pd.Grouper(freq="D"))]
# Now you have one DataFrame per day

# -----------------------------------------
# 6. Resample each day's data separately to 200ms windows
# -----------------------------------------
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

# Convert 'set' column back to integer (was float after aggregation)
data_resampled["set"] = data_resampled["set"].astype("int")

# -----------------------------------------
# 7. Save the final processed data
# -----------------------------------------
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
