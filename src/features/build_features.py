#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
df = pd.read_pickle("../../data/interim/02_data_outliers_removed_chauvenet.pkl")

# First 6 columns are predictors (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
predictor_cols = list(df.columns[:6])

# -----------------------------------------------------
# Plotting Settings
# -----------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

# -----------------------------------------------------
# Deal with Missing Values (Interpolation)
# -----------------------------------------------------
for col in predictor_cols:
    df[col] = df[col].interpolate()  # Fill missing values by interpolating

df.info()

# -----------------------------------------------------
# Calculate Set Duration
# -----------------------------------------------------
# Quick visualization for sets 25 and 50
df[df['set'] == 25]['acc_y'].plot()
df[df['set'] == 50]['acc_y'].plot()

# Compute duration for each set
for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    end = df[df['set'] == s].index[-1]
    duration = end - start
    df.loc[df['set'] == s, 'duration'] = duration

# Average duration per category
duration_df = df.groupby(['category'])['duration'].mean()

# -----------------------------------------------------
# Butterworth Low-Pass Filter
# -----------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200  # Sampling frequency (step size = 200ms → 5 Hz)
cutoff = 1.3     # Cutoff frequency; higher = less smoothing

# Example filter on 'acc_y'
df_lowpass = LowPass.low_pass_filter(df_lowpass, 'acc_y', fs, cutoff, order=5, phase_shift=True)

subset = df_lowpass[df_lowpass['set'] == 45]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

ax[0].plot(subset.index, subset['acc_y'], label='Original', color='red')
ax[0].set_title('Original Acceleration Data')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Acceleration (m/s²)')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(subset.index, subset['acc_y_lowpass'], label='Lowpass Filtered')
ax[1].set_title('Lowpass Filtered Acceleration Data')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('Acceleration (m/s²)')
ax[1].legend()
ax[1].grid(True)

fig.tight_layout()
plt.show()

# Apply low-pass filter to all predictor columns
for col in predictor_cols:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5, phase_shift=True)
    df_lowpass[col] = df_lowpass[col + '_lowpass']  # Replace original with filtered
    del df_lowpass[col + '_lowpass']

# -----------------------------------------------------
# Principal Component Analysis (PCA)
# -----------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# Determine explained variance for each component
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_cols)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_cols) + 1), pc_values)
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Apply PCA and keep 3 principal components
df_pca = PCA.apply_pca(df_pca, predictor_cols, number_comp=3)

# -----------------------------------------------------
# Sum of Squares Features (acc_r, gyr_r)
# -----------------------------------------------------
df_squared = df_pca.copy()
acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set'] == 14]
subset[['acc_r', 'gyr_r']].plot(subplots=True, figsize=(20, 10), title='Sum of Squares Attributes')

# -----------------------------------------------------
# Temporal Abstraction (Mean & Std)
# -----------------------------------------------------
df_temporal = df_squared.copy()
NumericalAbs = NumericalAbstraction()

window_size = 1000 / 200  # 5 seconds window (step size = 200ms)
predictor_cols += ['gyr_r', 'acc_r']

# Apply abstraction for each predictor
for col in predictor_cols:
    df_temporal = NumericalAbs.abstract_numerical(df_temporal, [col], window_size, 'mean')
    df_temporal = NumericalAbs.abstract_numerical(df_temporal, [col], window_size, 'std')

# Apply abstraction per set
df_temporal_list = []
for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set'] == s].copy()
    for col in predictor_cols:
        df_temporal = NumericalAbs.abstract_numerical(subset, [col], window_size, 'mean')
        df_temporal = NumericalAbs.abstract_numerical(subset, [col], window_size, 'std')
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# -----------------------------------------------------
# Frequency Abstraction (Fourier Transform)
# -----------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# Example on 'acc_y'
df_freq = FreqAbs.abstract_frequency(df_freq, ['acc_y'], window_size=int(2800/200), sampling_rate=int(1000/200))

subset = df_freq[df_freq['set'] == 14]
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# Apply frequency abstraction for all predictors
df_freq_list = []
for s in df_freq['set'].unique():
    subset = df_freq[df_freq['set'] == s].reset_index(drop=True)
    subset = FreqAbs.abstract_frequency(subset, predictor_cols, window_size=int(2800/200), sampling_rate=int(1000/200))
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# -----------------------------------------------------
# Reduce Overlap in Windows
# -----------------------------------------------------
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]  # Keep every second row to avoid overfitting

# -----------------------------------------------------
# Clustering (KMeans)
# -----------------------------------------------------
df_cluster = df_freq.copy()
cluster_cols = ['acc_y', 'acc_x', 'acc_z']

# Determine optimal k using Elbow Method
k_values = range(2, 10)
inertias = []
for k in k_values:
    subset = df_cluster[cluster_cols]
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Apply KMeans with chosen k
k = 5
kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
subset = df_cluster[cluster_cols]
df_cluster['cluster'] = kmeans.fit_predict(subset)

# -----------------------------------------------------
# 3D Visualization of Clusters
# -----------------------------------------------------
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df_cluster['acc_x'],
    df_cluster['acc_y'],
    df_cluster['acc_z'],
    c=df_cluster['cluster'],
    cmap='viridis',
    s=50
)

ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
ax.set_title('3D Cluster Visualization')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.show()

# -----------------------------------------------------
# 3D Visualization by Labels
# -----------------------------------------------------
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
for label in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == label]
    ax.scatter(
        subset['acc_x'],
        subset['acc_y'],
        subset['acc_z'],
        label=label,
        s=50
    )

ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
ax.set_title('3D Cluster Visualization by Label')
ax.legend()
plt.show()

# -----------------------------------------------------
# Save Final Processed Data
# -----------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features_extracted.pkl")
