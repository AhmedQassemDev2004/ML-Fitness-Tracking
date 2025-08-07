# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

DATA_PATH = "../../data/interim/01_data_processed.pkl"
OUTLIER_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle(DATA_PATH)
LABELS = df['label'].unique()

# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------
def plot_boxplots_by_label(dataframe, columns, title_prefix):
    dataframe[columns + ['label']].boxplot(by='label', figsize=(20, 10), layout=(1, len(columns)))
    plt.suptitle(f'{title_prefix} by label')
    plt.show()

def plot_binary_outliers(dataset, col, outlier_col, reset_index=True):
    df_plot = dataset.dropna(subset=[col, outlier_col]).copy()
    df_plot[outlier_col] = df_plot[outlier_col].astype(bool)
    if reset_index:
        df_plot = df_plot.reset_index()

    fig, ax = plt.subplots()
    ax.plot(df_plot.index[~df_plot[outlier_col]], df_plot[col][~df_plot[outlier_col]], "+")
    ax.plot(df_plot.index[df_plot[outlier_col]], df_plot[col][df_plot[outlier_col]], "r+")
    
    plt.xlabel("samples")
    plt.ylabel("value")
    plt.legend([f"no outlier {col}", f"outlier {col}"], loc="upper center", ncol=2, fancybox=True, shadow=True)
    plt.show()

def mark_outliers_iqr(dataset, col):
    df_copy = dataset.copy()
    Q1 = df_copy[col].quantile(0.25)
    Q3 = df_copy[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_copy[col + "_outlier"] = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
    return df_copy

def mark_outliers_chauvenet(dataset, col, C=2):
    df_copy = dataset.copy()
    mean = df_copy[col].mean()
    std = df_copy[col].std()
    N = len(df_copy)
    criterion = 1.0 / (C * N)

    deviation = abs(df_copy[col] - mean) / std
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)

    prob = [1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])) for i in range(N)]
    mask = [p < criterion for p in prob]
    df_copy[col + "_outlier"] = mask
    return df_copy

def mark_outliers_lof(dataset, columns, n_neighbors=20):
    df_copy = dataset.copy()
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    X = df_copy[columns].dropna()
    outlier_mask = lof.fit_predict(X) == -1
    df_copy['outlier_lof'] = False
    df_copy.loc[X.index, 'outlier_lof'] = outlier_mask
    return df_copy

# --------------------------------------------------------------
# 1. Visualize outliers using boxplots
# --------------------------------------------------------------
plot_boxplots_by_label(df, OUTLIER_COLS[:3], "Accelerometer")
plot_boxplots_by_label(df, OUTLIER_COLS[3:], "Gyroscope")

# --------------------------------------------------------------
# 2. Detect and plot IQR outliers (non-destructive)
# --------------------------------------------------------------
for col in OUTLIER_COLS:
    df_iqr = mark_outliers_iqr(df, col)
    plot_binary_outliers(df_iqr, col, col + "_outlier")

# --------------------------------------------------------------
# 3. Plot histograms for normality check
# --------------------------------------------------------------
df[OUTLIER_COLS[:3] + ['label']].plot.hist(by='label', figsize=(20, 20), layout=(3, 3))
df[OUTLIER_COLS[3:] + ['label']].plot.hist(by='label', figsize=(20, 20), layout=(3, 3))

# --------------------------------------------------------------
# 4. Detect and plot Chauvenet outliers (non-destructive)
# --------------------------------------------------------------
for col in OUTLIER_COLS:
    df_chauv = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(df_chauv, col, col + "_outlier")

# --------------------------------------------------------------
# 5. Detect and plot LOF outliers (non-destructive)
# --------------------------------------------------------------
df_lof = mark_outliers_lof(df, OUTLIER_COLS)
for col in OUTLIER_COLS:
    plot_binary_outliers(df_lof, col, "outlier_lof")

# --------------------------------------------------------------
# 6. Check outliers for a specific label (non-destructive)
# --------------------------------------------------------------
label_to_check = 'bench'
df_label = df[df['label'] == label_to_check].copy()

for col in OUTLIER_COLS:
    df_label_iqr = mark_outliers_iqr(df_label, col)
    plot_binary_outliers(df_label_iqr, col, col + "_outlier")

for col in OUTLIER_COLS:
    df_label_chauv = mark_outliers_chauvenet(df_label, col)
    plot_binary_outliers(df_label_chauv, col, col + "_outlier")

df_label_lof = mark_outliers_lof(df_label, OUTLIER_COLS)
for col in OUTLIER_COLS:
    plot_binary_outliers(df_label_lof, col, "outlier_lof")

# --------------------------------------------------------------
# 7. Remove outliers from a copy using Chauvenet method
# --------------------------------------------------------------
outliers_removed_df = df.copy()
for col in OUTLIER_COLS:
    for label in LABELS:
        subset = outliers_removed_df[outliers_removed_df['label'] == label].copy()
        subset = mark_outliers_chauvenet(subset, col)

        # replace outliers with NaN
        subset.loc[subset[col + "_outlier"], col] = np.nan

        # update only cleaned column back to main copy
        outliers_removed_df.loc[subset.index, col] = subset[col]

        n_outliers = subset[col + "_outlier"].sum()
        print(f"Removed {n_outliers} outliers from column {col} for label {label}")

# --------------------------------------------------------------
# 8. Save the cleaned DataFrame (outliers replaced with NaN)
# --------------------------------------------------------------
outliers_removed_df.to_pickle("../../data/interim/02_data_outliers_removed_chauvenet.pkl")
