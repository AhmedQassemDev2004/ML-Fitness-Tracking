# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

# --------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------
df = pd.read_pickle(r"../../data/interim/02_data_outliers_removed_chauvenet.pkl")
sensor_col = list(df.columns[:6])  # First 6 columns are sensor readings

# Configure plot style
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Handle missing values using interpolation
# --------------------------------------------------------------
df.info()
# Interpolation estimates missing values based on surrounding data points
df[df["set"] == 30]["acc_x"].plot()

for col in sensor_col:
    df[col] = df[col].interpolate()

df.info()
df[df["set"] == 30]["acc_x"].plot()

# --------------------------------------------------------------
# Calculate set duration
# --------------------------------------------------------------
# Each heavy set has 5 repetitions, medium set has 10.
# We measure duration per set to later estimate average repetition time.
for set_id in df["set"].unique():
    start = df[df["set"] == set_id].index[0]
    end = df[df["set"] == set_id].index[-1]
    duration = end - start
    df.loc[df["set"] == set_id, "duration"] = duration.seconds

duration_df = df.groupby("category")["duration"].mean()
duration_df[0] / 5  # ~2.9 seconds per rep (heavy set)
duration_df[1] / 10  # ~2.4 seconds per rep (medium set)

# --------------------------------------------------------------
# Apply Butterworth low-pass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()
sampling_frq = 1000 / 200  # Sampling frequency in Hz (5Hz for 200ms interval)
cutoff_frq = 1.3  # Lower cutoff → stronger smoothing

# Example: filtering acc_y
LowPass.low_pass_filter(df_lowpass, "acc_y", sampling_frq, cutoff_frq)
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"].iloc[0])

# Plot raw vs. filtered acc_y
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="Raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="Butterworth filter")
for a in ax:
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Filter all sensor columns
for col in sensor_col:
    df_lowpass = LowPass.low_pass_filter(
        df_lowpass, col, sampling_frq, cutoff_frq, order=5
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    df_lowpass.drop(columns=[col + "_lowpass"], inplace=True)

# --------------------------------------------------------------
# Principal Component Analysis (PCA)
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# Determine explained variance per component
pc_values = PCA.determine_pc_explained_variance(df_pca, sensor_col)
plt.plot(range(1, 7), pc_values)

# Reduce to 3 principal components
df_pca = PCA.apply_pca(df_pca, sensor_col, 3)
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Compute magnitude features (acc_r, gyr_r)
# --------------------------------------------------------------
# Magnitude = sqrt(x² + y² + z²) → independent of orientation
df_squares = df_pca.copy()
df_squares["acc_r"] = np.sqrt(
    df_squares["acc_x"] ** 2 + df_squares["acc_y"] ** 2 + df_squares["acc_z"] ** 2
)
df_squares["gyr_r"] = np.sqrt(
    df_squares["gyr_x"] ** 2 + df_squares["gyr_y"] ** 2 + df_squares["gyr_z"] ** 2
)

subset = df_squares[df_pca["set"] == 18]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction (rolling statistics)
# --------------------------------------------------------------
# Calculates moving averages & standard deviations within a window
df_temporal = df_squares.copy()
sensor_col = sensor_col + ["acc_r", "gyr_r"]
NumAbs = NumericalAbstraction()

df_temporal_list = []
for set_id in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == set_id].copy()
    subset = NumAbs.abstract_numerical(
        subset, sensor_col, window_size=5, aggregation_function="mean"
    )
    subset = NumAbs.abstract_numerical(
        subset, sensor_col, window_size=5, aggregation_function="std"
    )
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# Example plots
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency domain features (Fourier Transform)
# --------------------------------------------------------------
# Decomposes signals into frequency components for pattern detection
df_frq = df_temporal.copy().reset_index()
FreqAbd = FourierTransformation()
sampling_frq = int(1000 / 200)
window_size = int(2800 / 200)

# Example: single-column transformation
FreqAbd.abstract_frequency(df_frq, ["acc_y"], window_size, sampling_frq)

subset = df_frq[df_frq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_0.0_Hz_ws_14",
        "acc_y_freq_0.357_Hz_ws_14",
        "acc_y_freq_0.714_Hz_ws_14",
        "acc_y_freq_1.071_Hz_ws_14",
    ]
].plot()

# Apply Fourier transform to all sets
df_freq_list = []
for set_id in df_frq["set"].unique():
    print(f"Applying Fourier transformation to set {set_id}")
    subset = df_frq[df_frq["set"] == set_id].reset_index(drop=True).copy()
    subset = FreqAbd.abstract_frequency(subset, sensor_col, window_size, sampling_frq)
    df_freq_list.append(subset)

df_frq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
df_frq.drop("duration", axis=1, inplace=True)

# --------------------------------------------------------------
# Reduce correlation from overlapping windows
# --------------------------------------------------------------
# Skip every other row to reduce feature redundancy
df_frq.dropna(inplace=True)
df_frq = df_frq.iloc[::2]

# --------------------------------------------------------------
# Clustering (KMeans)
# --------------------------------------------------------------
from sklearn.cluster import KMeans

df_cluster = df_frq.copy()

cluster_col = ["acc_x", "acc_y", "acc_z"]

# Elbow method to determine optimal clusters
k_values = range(2, 10)
inertias = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    label = kmeans.fit_predict(df_cluster[cluster_col])
    inertias.append(kmeans.inertia_)
plt.plot(k_values, inertias, "--o")
# Observed elbow around k=5 or k=6

# Final model
kmeans = KMeans(n_clusters=6, n_init=20, random_state=0)
df_cluster["cluster"] = kmeans.fit_predict(df_cluster[cluster_col])

# Save clustering model
import joblib

joblib.dump(kmeans, "../../models/Clustering_model.pkl")

# 3D scatter plot by cluster
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# 3D scatter plot by label
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Save feature-engineered dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
