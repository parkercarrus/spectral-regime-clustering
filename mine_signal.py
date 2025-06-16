import numpy as np
import pandas as pd
from itertools import combinations
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import logic.data as data
from logic.process import split_train_test
import logic.model as model
import logic.eval as eval
import time
import matplotlib.pyplot as plt
import yfinance as yf

def simulate_trading(preds: pd.DataFrame, y_actual: pd.DataFrame, threshold: float = 0.0, capital: float = 1_000_000):
    positions = preds.copy()
    positions[:] = 0
    positions[preds > threshold] = 1
    positions[preds < -threshold] = -1

    daily_returns = (positions * y_actual).mean(axis=1)
    equity_curve = (1 + daily_returns).cumprod() * capital

    return equity_curve, daily_returns

def run_process(x_train, x_test, y_train, y_test, num_eigens=8, days_window=7, lookahead_days=1):
    training_data = model.compute_training(x_train, y_train, num_eigens, days_window, lookahead_days)
    test_data = eval.compute_testing(x_test, y_test, num_eigens, days_window, lookahead_days)

    X = model.get_lambda_array(training_data)
    Y = np.array([entry[2] for entry in training_data])

    if X.size == 0 or Y.size == 0:
        return -np.inf
    
    clf = eval.train_model(X, Y)

    target_names = test_data[0][2].index.to_list()
    preds = eval.get_predictions(clf, test_data, target_names)
    p = preds.get('preds')

    y_actual = pd.DataFrame([entry[2] for entry in test_data])
    p.index = y_actual.index.copy()

    alignment = p * y_actual
    weighted_accuracy = alignment.sum() / alignment.abs().sum() * 100
        
    return weighted_accuracy.mean(), p, y_actual

def get_subset_data(x_data: pd.DataFrame, y_data: pd.DataFrame, 
                    x_symbols: list, y_symbols: list) -> tuple:
    x_subset = x_data[x_symbols].copy()
    y_subset = y_data[y_symbols].copy()
    return x_subset, y_subset

if __name__ == "__main__":
    starttime = time.time()
    parallel = False

    X_UNIVERSE = ["JPM", "BAC", "GS", "MS", "WFC", "TFC", "USB", "C", "SCHW", "PNC"]
    Y_UNIVERSE = ["XLF"]

    x_data, y_data = data.get_data(days_period=500, x_symbols=X_UNIVERSE, y_symbols=Y_UNIVERSE)
    stock_combos = list(combinations(X_UNIVERSE, 4))

    def score_combo(stock_subset):
        x_subset, y_subset = get_subset_data(x_data, y_data, list(stock_subset), Y_UNIVERSE)
        x_train, x_test, y_train, y_test = split_train_test(x_subset, y_subset, 2/3)
        score, p, y_actual = run_process(x_train, x_test, y_train, y_test)

        equity_curve, daily_returns = simulate_trading(p, y_actual)
        final_value = equity_curve.iloc[-1]
        cagr = (final_value / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        return {
            'stocks': stock_subset,
            'score': score,
            'cagr': cagr,
            'sharpe': sharpe,
            'final_value': final_value
        }
    
    results = []

    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(score_combo, combo): combo for combo in stock_combos}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        for combo in stock_combos:
            result = score_combo(combo)
            results.append(result)

    results_df = pd.DataFrame(results)
    print("\n Top Stock Sets:")
    print(results_df.sort_values(by="sharpe", ascending=False).head())
    print(f"Results Calculated in {(time.time() - starttime):.2f} and saved to results.csv")
    results_df.to_csv("results.csv")

    best_result = results_df.sort_values(by="sharpe", ascending=False).iloc[0]
    best_stocks = list(best_result['stocks'])

    x_subset, y_subset = get_subset_data(x_data, y_data, best_stocks, Y_UNIVERSE)
    x_train, x_test, y_train, y_test = split_train_test(x_subset, y_subset, 2/3)

    _, p, y_actual = run_process(x_train, x_test, y_train, y_test)
    equity_curve, _ = simulate_trading(p, y_actual)


    strategy_curve = equity_curve / equity_curve.iloc[0]
    underlying_returns = y_actual[Y_UNIVERSE[0]]
    underlying_curve = (1 + underlying_returns).cumprod()

    underlying_curve = underlying_curve / underlying_curve.iloc[0]  # Normalize

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(strategy_curve.index, strategy_curve, label='Strategy')
    plt.plot(underlying_curve.index, underlying_curve, label=Y_UNIVERSE[0])
    plt.title(f"Equity Curve vs {Y_UNIVERSE[0]}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
