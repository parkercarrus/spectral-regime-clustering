import pandas as pd
import numpy as np
from typing import List, Tuple
from logic.model import iterwindows, get_eigenvals, ClusterModel, get_lambda_array, process_data

def compute_testing(x_data: pd.DataFrame, y_data: pd.DataFrame, num_eigens: int = 10, days_window: int = 3, lookahead_days: int = 0) -> List[Tuple[pd.Timestamp, pd.Series, pd.Series]]:
    results = []
    for day_idx, window_df, target_series in iterwindows(x_data, y_data, days_window, lookahead_days):
        lambda_vec = get_eigenvals(window_df, num_eigens)
        results.append((day_idx, pd.Series(lambda_vec), target_series))
    return results

def itertest(test_data, model: ClusterModel):
    X_test = get_lambda_array(test_data)  # shape (n_samples, n_features)
    Y_test = np.array([entry[2] for entry in test_data])  # list of pd.Series
    preds = model.predict_proba(X_test)
    for cluster_pred, y_actual in zip(preds, Y_test):
        pred = model.get_prediction(cluster_pred)
        yield (pred, y_actual)
    

def train_model(X: pd.DataFrame, Y: pd.Series) -> ClusterModel:
    X_processed = process_data(X, log=True, scale=False)
    n_clusters = 10
    model = ClusterModel(n_clusters)
    model.fit(X_processed, Y)

    return model

def get_predictions(model: ClusterModel, test_data: List[Tuple[pd.Timestamp, pd.Series, pd.Series]], target_names: List[str]) -> pd.DataFrame:
    X_test = get_lambda_array(test_data)
    Y_test = np.array([entry[2] for entry in test_data])
    preds = model.predict_expected_target(X_test)
    preds_named = pd.DataFrame(preds, columns=target_names)

    evaluation = evaluate_predictions(preds, Y_test)

    return {'preds': preds_named,
            'evals': evaluation}

def evaluate_predictions(preds: pd.DataFrame, Y_test: np.array) -> pd.Series:
    return pd.Series
