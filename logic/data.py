import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Microcap Biotech Stocks ($50M - $500M Market Cap)
# These companies exhibit high correlation during FDA events and binary outcomes
# Updated as of June 2025

X_SYMBOLS = [
    "JPM", "BAC", "GS", "MS", "USB", "PNC", "TFC", "CMA", "SCHW",
    "AXP", "COF", "DFS", "CB", "AIG", "MET", "PGR", "ALL", "PRU", "TRV"
]

Y_SYMBOLS = "XLF"

def get_close_data_hourly(symbol_list: list, start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
    """Get CLOSE data from YFINANCE for symbol_list between start_datetime and end_datetime"""
    data = yf.download(
        tickers=symbol_list,
        start=start_datetime.strftime('%Y-%m-%d'),
        end=end_datetime.strftime('%Y-%m-%d'),
        interval='1h',
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    close_data = data.xs('Close', axis=1, level=1)
    return close_data

def get_close_data_daily(symbol_list: list, start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
    """Get CLOSE data from YFINANCE for symbol_list between start_datetime and end_datetime"""
    data = yf.download(
        tickers=symbol_list,
        start=start_datetime.strftime('%Y-%m-%d'),
        end=end_datetime.strftime('%Y-%m-%d'),
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    close_data = data.xs('Close', axis=1, level=1)
    return close_data

def get_data(days_period: int = 365, x_symbols: list = X_SYMBOLS, y_symbols: list = Y_SYMBOLS) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = datetime.now()
    start = end - timedelta(days=days_period)

    x_data = get_close_data_hourly(x_symbols, start, end)
    y_data = get_close_data_daily(y_symbols, start, end)
    y_data = y_data.pct_change()

    return x_data, y_data
