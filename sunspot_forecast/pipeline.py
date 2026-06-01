"""ARIMA pipeline helpers and multi-model evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from darts.models import ARIMA, TBATS, Theta
from darts.utils.utils import SeasonalityMode
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_loader import DataLoader


def calculate_metrics(actual, predicted):
    actual = actual.values().flatten()
    predicted = predicted.values().flatten()
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(2 * np.abs(predicted - actual) / denominator) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    nrmse = rmse / (actual.max() - actual.min())
    return {"sMAPE": smape, "RMSE": rmse, "NRMSE": nrmse, "MAE": mae, "R2": r2}


def create_models():
    return {
        "ARIMA": ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 11)),
        "Theta": Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=11),
        "TBATS": TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[11]),
    }


def evaluate_predictions(test, predictions):
    from darts.metrics import mape, rmse

    return {"MAPE": mape(test, predictions), "RMSE": rmse(test, predictions)}


def implement_arima(series, future_covariates, forecast_horizon=6):
    train_cutoff = series.time_index[-forecast_horizon]
    train, test = series.split_before(train_cutoff)
    model = ARIMA(p=12, d=1, q=2)
    model.fit(train, future_covariates=future_covariates)
    predictions = model.predict(forecast_horizon, future_covariates=future_covariates)
    return predictions, model, test


def plot_predictions(series, predictions, test=None, save_prefix=""):
    plt.figure(figsize=(15, 7))
    series.plot(label="Historical", alpha=0.7)
    if test is not None:
        test.plot(label="Actual (Test Set)", alpha=0.7)
    predictions.plot(label="ARIMA Predictions")
    plt.title("Time Series: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_prefix}arima_predictions.png")
    plt.close()


def run_analysis(data_type="synthetic", n_points=500):
    if data_type == "synthetic":
        series, future_cov = DataLoader.generate_synthetic_data(n_points)
    elif data_type == "sunspot":
        series, future_cov = DataLoader.load_sunspot_data()
        if series is None:
            return None, None, None
    else:
        raise ValueError("data_type must be either 'synthetic' or 'sunspot'")
    print(f"\nImplementing ARIMA model for {data_type} data...")
    predictions, model, test = implement_arima(series, future_cov)
    plot_predictions(series, predictions, test, save_prefix=f"{data_type}_")
    metrics = evaluate_predictions(test, predictions)
    print(f"\nMetrics for {data_type} data:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    return series, predictions, model


def train_and_evaluate(series, forecast_horizon=5):
    train_size = int(len(series) * 0.8)
    train = series[:train_size]
    test = series[train_size:]
    models = create_models()
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(train)
            pred = model.predict(len(test))
            metrics = calculate_metrics(test, pred)
            results[name] = {"prediction": pred, **metrics}
            print(f"{name} Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.3f}" if k == "R2" else f"  {k}: {v:.2f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
    return results, train, test
