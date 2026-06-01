"""Train and compare Darts forecasting models on sunspot series."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape, rmse
from darts.models import ARIMA, TBATS, Theta
from darts.utils.utils import SeasonalityMode


class TimeSeriesAnalyzer:
    def __init__(self):
        self.models = {
            "ARIMA": ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 11)),
            "Theta": Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=11),
            "TBATS": TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[11]),
        }

    def _load_csv_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            df[["Year", "Month"]] = df["Month"].str.split("-", expand=True)
            df["Date"] = pd.to_datetime(df["Year"] + "-" + df["Month"] + "-01")
            df["Sunspot"] = np.where(df["Sunspot"] == 0, 1, df["Sunspot"])
            df_yearly = df.groupby(df["Date"].dt.year)["Sunspot"].mean().reset_index()
            df_yearly["Date"] = pd.to_datetime(df_yearly["Date"].astype(str) + "-01-01")
            return TimeSeries.from_dataframe(df_yearly, "Date", "Sunspot")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _generate_synthetic_data(self, n_points):
        dates = pd.date_range(start="1920-01-01", periods=n_points, freq="Y")
        trend = np.linspace(0, 10, n_points)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 11)
        noise = np.random.normal(0, 1, n_points)
        values = trend + seasonal + noise
        df = pd.DataFrame({"Date": dates, "Value": values})
        return TimeSeries.from_dataframe(df, "Date", "Value")

    def load_data(self, filepath=None, synthetic=False, n_points=100):
        if synthetic:
            return self._generate_synthetic_data(n_points)
        return self._load_csv_data(filepath)

    def analyze(self, series, train_test_split=0.8):
        if series is None:
            print("No data to analyze")
            return None, None, None
        train_size = int(len(series) * train_test_split)
        train = series[:train_size]
        test = series[train_size:]
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(train)
                pred = model.predict(len(test))
                metrics = self._calculate_metrics(test, pred)
                results[name] = {"prediction": pred, **metrics}
                self._print_metrics(name, metrics)
            except Exception as e:
                print(f"Error training {name}: {e}")
        return results, train, test

    def _calculate_metrics(self, actual, predicted):
        try:
            return {"MAPE": mape(actual, predicted), "RMSE": rmse(actual, predicted)}
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {"MAPE": np.nan, "RMSE": np.nan}

    def _print_metrics(self, model_name, metrics):
        print(f"{model_name} Performance:")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric}: {value:.2f}")

    def plot_results(self, series, results, train, test, save_path="forecast.png"):
        if series is None or results is None:
            print("No data to plot")
            return
        plt.figure(figsize=(15, 7))
        train.plot(label="Training", alpha=0.6)
        test.plot(label="Test", alpha=0.6)
        if results:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
            for (name, result), color in zip(results.items(), colors):
                if "prediction" in result:
                    result["prediction"].plot(
                        label=f"{name} (MAPE: {result['MAPE']:.1f}%)", color=color
                    )
        plt.title("Sunspot Number Forecasting Comparison")
        plt.xlabel("Time")
        plt.ylabel("Sunspot Number")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
