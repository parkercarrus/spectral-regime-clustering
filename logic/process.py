import pandas as pd

def get_window_df(x_data: pd.DataFrame, start_idx: pd.Timestamp, days_window: int = 5):
    if start_idx not in x_data.index:
        return None
    hours_window = days_window * 7 # 7 trading hours in a business day
    start_loc = x_data.index.get_loc(start_idx)
    end_loc = start_loc + hours_window
    return x_data.iloc[start_loc:end_loc]

def get_target_series(y_data: pd.DataFrame, start_idx: pd.Timestamp, days_window: int = 5, lookahead_days: int = 0) -> pd.Series:
    if start_idx not in y_data.index:
        next_idxs = y_data.index[y_data.index >= pd.to_datetime(start_idx.date())]
        if len(next_idxs) == 0:
            return None
        start_idx = next_idxs[0]
    start_loc = y_data.index.get_loc(start_idx)
    # Target is the next trading day after the window
    target_loc = start_loc + days_window + lookahead_days
    if target_loc >= len(y_data): # Out of Bounds
        return None
    return y_data.iloc[target_loc]


def split_train_test(x_data: pd.DataFrame, y_data: pd.DataFrame, split: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(x_data) * split)

    # Split x_data
    x_train = x_data.iloc[:split_idx]
    x_test = x_data.iloc[split_idx:]

    # Split y_data
    split_idx_y = int(len(y_data) * 2 / 3)
    y_train = y_data.iloc[:split_idx_y]
    y_test = y_data.iloc[split_idx_y:]
    
    return x_train, x_test, y_train, y_test


    