# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(r"../../data/interim/02_data_outliers_removed_chauvenet.pkl")

sensor_col = list(df.columns[:6])


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()
# Interpolation is a technique used in mathematics and statistics to estimate values between two known values. In the context of Pandas DataFrames, interpolation can be applied along rows or columns to estimate missing values based on neighboring data points.
# df['acc_x'].interpolate().isna().sum()

df[df["set"] == 30]["acc_x"].plot()

for col in sensor_col:
    df[col] = df[col].interpolate()

df.info()

df[df["set"] == 30]["acc_x"].plot()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

# We know that: the heavy set contains 5 repetitions, and medium set contains 10 repetitions for each exercise

# Now we need to know the duration for each set

for set in df["set"].unique():

    strart = df[df["set"] == set].index[0]
    end = df[df["set"] == set].index[-1]

    duration = end - strart

    df.loc[(df["set"] == set), "duration"] = duration.seconds


duration_df = df.groupby("category")["duration"].mean()

duration_df[0] / 5  # so each repetition take 2.9 sec in heavy set
duration_df[1] / 10  # so each repetition take 2.4 sec in medium set


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()

LowPass = LowPassFilter()

sampling_frq = (
    1000 / 200
)  # # because we are taking the record every 200 ms, so that line calculates number of repetition in 1 sec

cutoff_frq = 1.3  # the low cutoff frequency, meaning more smoothing in signal

LowPass.low_pass_filter(df_lowpass, "acc_y", sampling_frq, cutoff_frq)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)


for col in sensor_col:
    df_lowpass = LowPass.low_pass_filter(
        df_lowpass, col, sampling_frq, cutoff_frq, order=5
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pc_values = PCA.determine_pc_explained_variance(df_pca, sensor_col)
plt.plot(range(1, 7), pc_values)
df_pca = PCA.apply_pca(df_pca, sensor_col, 3)
subset = df_pca[df_pca["set"] == 35]

subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# To further exploit the data, the scalar magnitudes r of the accelerometer and gyroscope were calculated.
# r is the scalar magnitude of the three combined data points: x, y, and z.
# The advantage of using r versus any particular data direction is that it is impartial to device orientation and can handle dynamic re-orientations.
# r is calculated by: r_{magnitude} = sqrt{x^2 + y^2 + z^2}

df_squares = df_pca.copy()

acc_r = df_squares["acc_x"] ** 2 + df_squares["acc_y"] ** 2 + df_squares["acc_z"] ** 2
gyr_r = df_squares["gyr_x"] ** 2 + df_squares["gyr_y"] ** 2 + df_squares["gyr_z"] ** 2


df_squares["acc_r"] = np.sqrt(acc_r)
df_squares["gyr_r"] = np.sqrt(gyr_r)


df_squares
subset = df_squares[df_pca["set"] == 18]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

# Rolling averages are commonly used to smooth out short-term fluctuations or noise in time-series data and highlight longer-term trends or patterns.
# They are particularly useful for identifying patterns that might not be immediately apparent in raw data, especially when dealing with data that contains significant variability or noise.

df_temporal = df_squares.copy()
sensor_col = sensor_col + ["acc_r", "gyr_r"]
NumAbs = NumericalAbstraction()

# NumAbs.abstract_numerical(df_temporal , sensor_col , window_size=5 ,aggregation_function= 'mean' )
# we need to make moving average on each set because each set may containing different label (exercise)

df_temporal_list = []
for set in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == set].copy()

    for col in sensor_col:
        subset = NumAbs.abstract_numerical(
            subset, sensor_col, window_size=5, aggregation_function="mean"
        )
        subset = NumAbs.abstract_numerical(
            subset, sensor_col, window_size=5, aggregation_function="std"
        )

    df_temporal_list.append(subset)


df_temporal = pd.concat(df_temporal_list)

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

# The idea of a Fourier transformation is that any sequence of measurements we perform can be represented by a combination of sinusoid functionswith different frequencies

# DFT can provide insight into patterns and trends that would not otherwise be visible. Additionally, the DFT can be used to reduce noise, allowing for more accurate models.
df_frq = df_temporal.copy().reset_index()
# df_frq
FreqAbd = FourierTransformation()

sampling_frq = int(1000 / 200)
window_size = int(2800 / 200)
FreqAbd.abstract_frequency(df_frq, ["acc_y"], window_size, sampling_frq)
subset = df_frq[df_frq["set"] == 15]
subset[["acc_y"]].plot()
subset.columns
# Fourier transformation  abstracted the sign into its basic constituent elements

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

df_freq_list = []
for set in df_frq["set"].unique():
    print(f"Applying Fourier transformation to set {set}")
    subset = df_frq[df_frq["set"] == set].reset_index(drop=True).copy()

    subset = FreqAbd.abstract_frequency(subset, sensor_col, window_size, sampling_frq)

    df_freq_list.append(subset)
df_frq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
df_frq = df_frq.drop("duration", axis=1)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# All extra features are based on moving averages, so the value between the different rows are highly correlated #And this could cause overfitting in our model

# So we need dealing with that: we will take 50% from data by skipping one row in each step
df_frq
df_frq = df_frq.dropna()
df_frq = df_frq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

from sklearn.cluster import KMeans

df_cluster = df_frq.copy()

cluster_col = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:

    subset = df_cluster[cluster_col]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    label = kmeans.fit_predict(subset)

    inertias.append(kmeans.inertia_)

inertias
plt.plot(k_values, inertias, "--o")
# So the 5 or 6 is the optimal number


kmeans = KMeans(n_clusters=6, n_init=20, random_state=0)
subset = df_cluster[cluster_col]
df_cluster["cluster"] = kmeans.fit_predict(subset)
df_cluster

import joblib

joblib.dump(kmeans, "../../models/Clustering_model.pkl")

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
# Plot Labels
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
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
