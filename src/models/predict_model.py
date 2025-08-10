import pandas as pd
from src.features.remove_outliers import mark_outliers_chauvenet
import numpy as np
from src.features.build_features import LowPassFilter, PrincipalComponentAnalysis, NumericalAbstraction, FourierTransformation
import joblib



class FitnessTrackerPredictor:
    def __init__(self, acc_path, gyr_path, model_path, cluseter_model_path):
        self.model_path = model_path
        self.acc_path = acc_path
        self.gyr_path = gyr_path
        self.cluseter_model_path = cluseter_model_path

        self.model = joblib.load(model_path)

    def read_data(self):
        acc_df = pd.read_csv(self.acc_path, header=None, names=['acc_x', 'acc_y', 'acc_z'])
        gyr_df = pd.read_csv(self.gyr_path, header=None, names=['gyr_x', 'gyr_y', 'gyr_z'])

        pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
        pd.to_datetime(acc_df['time (01:00)'])

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
    

    def remove_outliers(self, data):
        outliers_removed_df = self.read_data()
        for col in outliers_removed_df.columns:
            dataset = mark_outliers_chauvenet(data, col)
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
        df.set_index("epoch (ms)", inplace=True, drop=True)

        # Downsampling
        df.dropna(inplace=True)
        df = df.iloc[::2]

        # Clustering
        kmeans = joblib.load(self.cluster_model_path)
        cluster_cols = ["acc_x", "acc_y", "acc_z"]
        cluster_preds = kmeans.predict(df[cluster_cols])
        df["cluster"] = pd.Series(cluster_preds).value_counts().idxmax()

        return df
