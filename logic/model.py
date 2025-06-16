from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from typing import List, Tuple
from logic.process import get_window_df, get_target_series
from sklearn.preprocessing import  StandardScaler

def iterwindows(x_data: pd.DataFrame, y_data: pd.DataFrame, days_window: int = 3, lookahead_days: int = 0, stride: int = 7):
    index = x_data.index
    y_data = y_data.copy()

    for i in range(0, len(index) - days_window * 7, stride):
        start_idx = index[i]

        window_df = get_window_df(x_data, start_idx, days_window=days_window)
        if window_df is None:
            continue

        target_series = get_target_series(y_data, start_idx, days_window, lookahead_days)
        if target_series is None:
            continue  # the following day is not a business day

        forecast_time = window_df.index[-1]        
        
        yield forecast_time.normalize(), window_df, target_series

def get_eigenvals(window_df: pd.DataFrame, num_eigens: int = 10) -> np.ndarray:
    cov_matrix = window_df.cov()
    eigenvals = np.linalg.eigvalsh(cov_matrix)
    sorted_eigenvals = np.sort(eigenvals)[::-1]
    return sorted_eigenvals[1:num_eigens+1]

def compute_training(x_data: pd.DataFrame, y_data: pd.DataFrame, num_eigens: int = 10, days_window: int = 3, lookahead_days: int = 0) -> List[Tuple[pd.Timestamp, pd.Series, pd.Series]]:
    results = []
    for day_idx, window_df, target_series in iterwindows(x_data, y_data, days_window, lookahead_days):
        lambda_vec = get_eigenvals(window_df, num_eigens)
        results.append((day_idx, pd.Series(lambda_vec), target_series))
    return results

def get_lambda_array(results: List[Tuple[pd.Timestamp, pd.Series, pd.Series]]) -> np.ndarray:
    return np.array([entry[1].values for entry in results])

def get_day_index_and_targets(results: List[Tuple[pd.Timestamp, pd.Series, pd.Series]]) -> Tuple[List[pd.Timestamp], List[pd.Series]]:
    day_indices = [entry[0] for entry in results]
    target_series_list = [entry[2] for entry in results]
    return day_indices, target_series_list

def process_data(X: np.ndarray, log: bool = True, scale: bool = False) -> np.ndarray:
    X_proc = X.copy()

    if log:
        if np.any(X_proc < -1):
            raise ValueError("Input contains values < -1, log1p is undefined.")
        X_proc = np.log1p(X_proc)

    if scale:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    return X_proc

class ClusterModel:
    def __init__(self, n_components):
        self.model = GaussianMixture(n_components=n_components)
        self.cluster_targets = {}

    def fit(self, X, Y):
        self.model.fit(X)
        labels = self.model.predict(X)
        self.cluster_targets = {}

        for k in range(self.model.n_components):
            cluster_indices = labels == k
            cluster_Y_series = Y[cluster_indices]
            Y_df = pd.DataFrame(cluster_Y_series)
            self.cluster_targets[k] = Y_df

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def get_attributes(self, cluster_id) -> pd.DataFrame:
        return self.cluster_targets.get(cluster_id, None)
    
    def get_prediction(self, cluster_id):
        attributes = self.get_attributes(cluster_id)
        median = attributes.median()
        var = attributes.var()
        return (median, var)
    
    def predict_expected_target(self, X_batch):
        probs = self.predict_proba(X_batch) 
        # Get medians per cluster 
        medians = [self.get_prediction(k)[0].values for k in range(self.model.n_components)]  # list of Series
        medians_matrix = np.stack(medians) 

        # Weighted average across clusters for each sample
        expected_targets = probs.dot(medians_matrix) 

        return expected_targets  # List of expected target vectors per sample