import sys
import os
import math 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import scipy.special
import numpy as np
from src.features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.features.FrequencyAbstraction import FourierTransformation
from src.features.TemporalAbstraction import NumericalAbstraction
import joblib

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

class FitnessTrackerPredictor:
    def __init__(self, acc_path, gyr_path, model_path, cluster_model_path, feature_set=None):
        self.model_path = model_path
        self.acc_path = acc_path
        self.gyr_path = gyr_path
        self.cluster_model_path = cluster_model_path
        self.feature_set = feature_set if feature_set else np.load('models/feature_set_4.npy')
        self.model = joblib.load(model_path)

    def read_data(self):
        gyr_df = pd.read_csv(self.gyr_path)
        acc_df = pd.read_csv(self.acc_path)

        pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
        pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

        acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
        gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

        acc_df.drop(['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis=1, inplace=True)
        gyr_df.drop(['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis=1, inplace=True)

        data_merged = pd.concat([acc_df, gyr_df], axis=1)
        data_merged.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

        sampling = {
            "acc_x": "mean",
            "acc_y": "mean",
            "acc_z": "mean",
            "gyr_x": "mean",
            "gyr_y": "mean",
            "gyr_z": "mean",
        }

        # data_merged[:1000].resample(rule='200ms').apply(sampling) # that would result error so we will seperate by days then concat again

        days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
        data_resampled =  pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

        return data_resampled
    

    def remove_outliers(self):
        outliers_removed_df = self.read_data()
        for col in outliers_removed_df.columns:
            dataset = mark_outliers_chauvenet(outliers_removed_df, col)
            dataset.loc[dataset[col + "_outlier"], col] = np.nan
            outliers_removed_df[col] = dataset[col].interpolate()
        return outliers_removed_df
    
    def apply_feature_engineering(self):
        """Feature engineering pipeline: smoothing → PCA → feature extraction → clustering."""
        
        df = self.remove_outliers()
        sensor_cols = list(df.columns)

        # Low-pass filtering
        lp = LowPassFilter()
        sampling_freq = 1000 / 200
        cutoff_freq = 1.3
        for col in df.columns:
            df = lp.low_pass_filter(df, col, sampling_freq, cutoff_freq, order=5)
            df[col] = df[col + "_lowpass"]
            df.drop(columns=[col + "_lowpass"], inplace=True)
        print("After low-pass:", len(df))

        # PCA
        pca = PrincipalComponentAnalysis()
        df = pca.apply_pca(df, df.columns, 3)

        # Magnitude features
        df["acc_r"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        df["gyr_r"] = np.sqrt(df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2)
        print("After PCA + magnitude:", len(df))

        # Temporal aggregation
        num_abs = NumericalAbstraction()
        agg_cols = sensor_cols + ["acc_r", "gyr_r"]
        df = num_abs.abstract_numerical(df, agg_cols, window_size=5, aggregation_function="mean")
        df = num_abs.abstract_numerical(df, agg_cols, window_size=5, aggregation_function="std")
        print("After temporal:", len(df))

        # Frequency domain features
        freq_abs = FourierTransformation()
        sampling_freq_int = int(1000 / 200)
        window_size = int(2800 / 200)
        df = freq_abs.abstract_frequency(df.reset_index(drop=True), agg_cols, window_size, sampling_freq_int)
        # df.set_index("epoch (ms)", inplace=True, drop=True)

        # Downsampling
        df.dropna(inplace=True)
        df = df.iloc[::2]

        # Clustering
        kmeans = joblib.load(self.cluster_model_path)
        cluster_cols = ["acc_x", "acc_y", "acc_z"]
        cluster_preds = kmeans.predict(df[cluster_cols])
        df["cluster"] = pd.Series(cluster_preds).value_counts().idxmax()

        return df

    def predict_activity(self):
        df = self.apply_feature_engineering()
        sorted_features = [f for f in self.model.feature_names_in_ if f in self.feature_set]
        df = df[sorted_features]
        print(len(self.model.feature_names_in_))
        pred = self.model.predict(df)
        return pd.DataFrame(pred).mode()[0][0]


if __name__ == "__main__":
    # Example usage
    predictor = FitnessTrackerPredictor(
        acc_path="C:\\Users\\ahmed\\Desktop\\ML-Fitness-Tracking\\data\\raw\\MetaMotion\\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv",
        gyr_path="C:\\Users\\ahmed\\Desktop\\ML-Fitness-Tracking\\data\\raw\\MetaMotion\\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv",
        model_path="models/final_model.pkl",
        cluster_model_path="models/Clustering_model.pkl"
    )
    activity = predictor.predict_activity()
    print(f"Predicted Activity: {activity}")