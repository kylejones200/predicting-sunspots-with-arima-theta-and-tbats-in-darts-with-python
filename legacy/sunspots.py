"""Generated from Jupyter notebook: sunspots

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from datetime import datetime, timedelta

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape, mase, rmse
from darts.models import (
    ARIMA,
    TBATS,
    ExponentialSmoothing,
    LinearRegressionModel,
    NHiTSModel,
    RandomForest,
    Theta,
)
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.utils import ModelMode, SeasonalityMode
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DataLoader:
    @staticmethod
    def load_sunspot_data():
        """Load and prepare sunspot data."""
        try:
            df = pd.read_csv(
                "SN_m_tot_V2.0.csv",
                delimiter=";",
                header=None,
                names=[
                    "Year",
                    "Month",
                    "Decimal_Date",
                    "Sunspots",
                    "Std",
                    "Observations",
                    "Definitive",
                ],
                na_values=["*******"],
            )
            df["date"] = pd.to_datetime(
                df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
            )
            current_date = datetime.now()
            start_date = current_date - timedelta(days=365 * 30)
            df = df[df["date"] >= start_date]
            series = TimeSeries.from_dataframe(df, "date", "Sunspots")
            future_cov = datetime_attribute_timeseries(series, "month", cyclic=True)
            return (series, future_cov)
        except FileNotFoundError:
            print(
                "Sunspot data file not found. Please ensure SN_m_tot_V2.0.csv exists."
            )
            return (None, None)

    @staticmethod
    def generate_synthetic_data(n_points=500):
        """Generate synthetic time series data."""
        current_date = datetime.now()
        start_date = current_date - timedelta(days=n_points - 1)
        dates = pd.date_range(start=start_date, periods=n_points, freq="D")
        trend = np.linspace(0, 10, n_points)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 365)
        noise = np.random.normal(0, 1, n_points)
        y = trend + seasonal + noise
        df = pd.DataFrame({"date": dates, "value": y})
        series = TimeSeries.from_dataframe(df, "date", "value")
        future_cov = datetime_attribute_timeseries(series, "month", cyclic=True)
        return (series, future_cov)


class TimeSeriesAnalyzer:
    def __init__(self):
        self.models = {
            "ARIMA": ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 11)),
            "Theta": Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=11),
            "TBATS": TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[11]),
        }

    def _load_csv_data(self, filepath):
        """Load and prepare sunspot data."""
        try:
            df = pd.read_csv(filepath)
            df[["Year", "Month"]] = df["Month"].str.split("-", expand=True)
            df["Date"] = pd.to_datetime(df["Year"] + "-" + df["Month"] + "-01")
            df["Sunspot"] = np.where(df["Sunspot"] == 0, 1, df["Sunspot"])
            df_yearly = df.groupby(df["Date"].dt.year)["Sunspot"].mean().reset_index()
            df_yearly["Date"] = pd.to_datetime(df_yearly["Date"].astype(str) + "-01-01")
            series = TimeSeries.from_dataframe(df_yearly, "Date", "Sunspot")
            return series
        except Exception as e:
            print(f"Error loading data: {str(e)}")
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
        """Load either real data from CSV or generate synthetic data"""
        if synthetic:
            return self._generate_synthetic_data(n_points)
        else:
            return self._load_csv_data(filepath)

    def analyze(self, series, train_test_split=0.8):
        """Train models and evaluate predictions"""
        if series is None:
            print("No data to analyze")
            return (None, None, None)
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
                print(f"Error training {name}: {str(e)}")
                continue
        return (results, train, test)

    def _calculate_metrics(self, actual, predicted):
        try:
            return {"MAPE": mape(actual, predicted), "RMSE": rmse(actual, predicted)}
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {"MAPE": np.nan, "RMSE": np.nan}

    def _print_metrics(self, model_name, metrics):
        print(f"{model_name} Performance:")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric}: {value:.2f}")

    def plot_results(self, series, results, train, test, save_path="forecast.png"):
        """Plot predictions from all models"""
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


def animate(frame):
    current_idx = frame + 1
    for name in models.keys():
        pred = results[name]["prediction"]
        lines[name].set_data(pred.time_index[:current_idx], pred.values()[:current_idx])
    return list(lines.values())


def calculate_metrics(actual, predicted):
    """Calculate multiple metrics including sMAPE, RMSE, MAE, and R²."""
    actual = actual.values().flatten()
    predicted = predicted.values().flatten()
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(2 * np.abs(predicted - actual) / denominator) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    nrmse = rmse / (actual.max() - actual.min())
    return {"sMAPE": smape, "RMSE": rmse, "NRMSE": nrmse, "MAE": mae, "R2": r2}


def create_metrics_table(results, data_type):
    """Create a formatted table of metrics for all models."""
    metrics = ["sMAPE", "RMSE", "NRMSE", "MAE", "R2"]
    df = pd.DataFrame(columns=metrics, index=results.keys())
    for model_name, model_results in results.items():
        for metric in metrics:
            df.loc[model_name, metric] = model_results[metric]
    df = df.round(3)
    df["Data Type"] = data_type
    return df


def create_models():
    """Create dictionary of models with adjusted parameters."""
    return {
        "ARIMA": ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 11)),
        "Theta": Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=11),
        "TBATS": TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[11]),
    }


def evaluate_predictions(test, predictions):
    """Calculate and return evaluation metrics."""
    from darts.metrics import mape, rmse

    mape_score = mape(test, predictions)
    rmse_score = rmse(test, predictions)
    return {"MAPE": mape_score, "RMSE": rmse_score}


def implement_arima(series, future_covariates, forecast_horizon=6):
    """Implement ARIMA model."""
    train_cutoff = series.time_index[-forecast_horizon]
    train, test = series.split_before(train_cutoff)
    model = ARIMA(p=12, d=1, q=2)
    model.fit(train, future_covariates=future_covariates)
    predictions = model.predict(forecast_horizon, future_covariates=future_covariates)
    return (predictions, model, test)


def main():
    analyzer = TimeSeriesAnalyzer()
    print("\nAnalyzing sunspot data...")
    try:
        real_series = analyzer.load_data("SN_m_tot_V2.0.csv")
        if real_series is not None:
            real_results, real_train, real_test = analyzer.analyze(real_series)
            analyzer.plot_results(
                real_series, real_results, real_train, real_test, "sunspot_forecast.png"
            )
    except FileNotFoundError:
        print("Sunspot data file not found. Please ensure the CSV file exists.")


def plot_predictions(series, predictions, test=None, save_prefix=""):
    """Plot actual vs predicted values."""
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
    """Run the complete analysis pipeline."""
    if data_type == "synthetic":
        series, future_cov = DataLoader.generate_synthetic_data(n_points)
    elif data_type == "sunspot":
        series, future_cov = DataLoader.load_sunspot_data()
        if series is None:
            return (None, None, None)
    else:
        raise ValueError("data_type must be either 'synthetic' or 'sunspot'")
    print(f"\nImplementing ARIMA model for {data_type} data...")
    predictions, model, test = implement_arima(series, future_cov)
    plot_predictions(series, predictions, test, save_prefix=f"{data_type}_")
    metrics = evaluate_predictions(test, predictions)
    print(f"\nMetrics for {data_type} data:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    return (series, predictions, model)


def train_and_evaluate(series, forecast_horizon=5):
    """Train models and make predictions."""
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
            print(f"sMAPE: {metrics['sMAPE']:.2f}%")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"NRMSE: {metrics['NRMSE']:.2f}")
            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"R²: {metrics['R2']:.3f}")
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    return (results, train, test)


def main() -> None:
    df = pd.read_csv(
        "SN_m_tot_V2.0.csv",
        delimiter=";",
        header=None,
        names=[
            "Year",
            "Month",
            "Decimal_Date",
            "Sunspots",
            "Std",
            "Observations",
            "Definitive",
        ],
        na_values=["*******"],
    )

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )

    df.set_index("Date", inplace=True)

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    df_yearly = df.resample("YE")["Sunspots"].mean().reset_index()

    df_yearly.columns = ["Year", "Sunspots"]

    series = TimeSeries.from_dataframe(df_yearly, "Year", "Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend="add",
            seasonal="add",
            seasonal_periods=11,
            damped=True,
            initialization_method="estimated",
        ),
        "Theta": Theta(
            seasonality_period=11, season_mode=SeasonalityMode.MULTIPLICATIVE
        ),
        "Linear Regression": LinearRegressionModel(
            lags=[-1, -2, -11, -12], output_chunk_length=20
        ),
        "Random Forest": RandomForest(
            lags=[-1, -2, -11, -12],
            output_chunk_length=20,
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
        ),
        "TBATS": TBATS(
            seasonal_periods=[11],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(
            input_chunk_length=24,
            output_chunk_length=20,
            n_epochs=100,
            num_blocks=3,
            num_stacks=2,
            num_layers=2,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        val_pred = model.predict(len(val))
        mase_score = mase(val_pred, val, train, seasonal_period=11)
        future_pred = model.predict(20)
        results[name] = {"prediction": future_pred, "mase": mase_score}
        print(f"{name} MASE: {mase_score:.2f}")

    fig = plt.figure(figsize=(15, 7))

    ax = plt.axes()

    start_date = pd.Timestamp("19450101")

    end_date = results[list(models.keys())[0]]["prediction"].time_index[-1]

    historical_df = series.pd_dataframe()

    historical_df = historical_df[historical_df.index >= start_date]

    ax.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    ax.set_xlim(start_date, end_date)

    ax.set_ylim(0, series.values().max() * 1.1)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    lines = {}

    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot(
            [],
            [],
            label=f"{name} (MASE: {results[name]['mase']:.2f})",
            lw=2,
            color=color,
        )
        lines[name] = line

    plt.title("Yearly Sunspots Forecast - 20 Years into Future", pad=20)

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=20, interval=200, blit=True, repeat=True
    )

    anim.save("sunspots_forecast.gif", writer="pillow", fps=5, dpi=100)

    plt.close()

    print("\nFinal predicted values (20 years into future):")

    for name, result in results.items():
        final_value = result["prediction"].values()[-1][0]
        print(f"{name}: {final_value:.2f} (MASE: {result['mase']:.2f})")

    plt.figure(figsize=(15, 7))

    plt.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        plt.plot(
            pred.time_index,
            pred.values(),
            label=f"{name} (MASE: {results[name]['mase']:.2f})",
            color=color,
        )

    plt.xlim(start_date, end_date)

    plt.ylim(
        0,
        max(
            historical_df["Sunspots"].max(),
            max([result["prediction"].values().max() for result in results.values()]),
        )
        * 1.1,
    )

    plt.title("Sunspots Forecast - All Models")

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig("sunspots_forecast_all_models.png", bbox_inches="tight", dpi=300)

    plt.close()

    print("\nModel Comparison:")

    print("-" * 50)

    print(f"{'Model':<25} {'MASE':>10}")

    print("-" * 50)

    for name, result in results.items():
        print(f"{name:<25} {result['mase']:>10.2f}")

    print("-" * 50)

    df = pd.read_csv(
        "SN_m_tot_V2.0.csv",
        delimiter=";",
        header=None,
        names=[
            "Year",
            "Month",
            "Decimal_Date",
            "Sunspots",
            "Std",
            "Observations",
            "Definitive",
        ],
        na_values=["*******"],
    )

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )

    df.set_index("Date", inplace=True)

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    series = TimeSeries.from_dataframe(df, "Date", "Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend="multiplicative",
            seasonal="multiplicative",
            seasonal_periods=132,
            damped=True,
            initialization_method="heuristic",
        ),
        "Theta": Theta(
            seasonality_period=132, season_mode=SeasonalityMode.MULTIPLICATIVE
        ),
        "Linear Regression": LinearRegressionModel(
            lags=list(range(1, 133)), output_chunk_length=240
        ),
        "Random Forest": RandomForest(
            lags=list(range(1, 133)),
            output_chunk_length=240,
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
        ),
        "TBATS": TBATS(
            seasonal_periods=[12, 132],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(
            input_chunk_length=264,
            output_chunk_length=240,
            n_epochs=100,
            num_blocks=3,
            num_stacks=2,
            num_layers=2,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        val_pred = model.predict(len(val))
        mase_score = mase(val_pred, val, train, seasonal_period=132)
        future_pred = model.predict(240)
        results[name] = {"prediction": future_pred, "mase": mase_score}
        print(f"{name} MASE: {mase_score:.2f}")

    plt.figure(figsize=(15, 7))

    plt.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        plt.plot(pred.time_index, pred.values(), label=name, color=color)

    plt.xlim(start_date, end_date)

    plt.ylim(
        0,
        max(
            historical_df["Sunspots"].max(),
            max([result["prediction"].values().max() for result in results.values()]),
        )
        * 1.1,
    )

    plt.title("Sunspots Forecast - All Models")

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig("sunspots_forecast_all_models.png", bbox_inches="tight", dpi=300)

    plt.close()

    df = pd.read_csv(
        "SN_m_tot_V2.0.csv",
        delimiter=";",
        header=None,
        names=[
            "Year",
            "Month",
            "Decimal_Date",
            "Sunspots",
            "Std",
            "Observations",
            "Definitive",
        ],
        na_values=["*******"],
    )

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )

    df.set_index("Date", inplace=True)

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    df_yearly = df.resample("Y")["Sunspots"].mean().reset_index()

    df_yearly.columns = ["Year", "Sunspots"]

    series = TimeSeries.from_dataframe(df_yearly, "Year", "Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.MULTIPLICATIVE,
            seasonal=SeasonalityMode.MULTIPLICATIVE,
            seasonal_periods=11,
            damped=True,
        ),
        "Theta": Theta(
            seasonality_period=11, season_mode=SeasonalityMode.MULTIPLICATIVE
        ),
        "Linear Regression": LinearRegressionModel(
            lags=[-1, -2, -11, -12], output_chunk_length=20
        ),
        "Random Forest": RandomForest(
            lags=[-1, -2, -11, -12],
            output_chunk_length=20,
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
        ),
        "TBATS": TBATS(
            seasonal_periods=[11],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(
            input_chunk_length=24,
            output_chunk_length=20,
            n_epochs=100,
            num_blocks=3,
            num_stacks=2,
            num_layers=2,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        val_pred = model.predict(len(val))
        mape_score = mape(val_pred, val)
        future_pred = model.predict(20)
        results[name] = {"prediction": future_pred, "mape": mape_score}
        print(f"{name} MAPE: {mape_score:.2f}%")

    fig = plt.figure(figsize=(15, 7))

    ax = plt.axes()

    start_date = pd.Timestamp("19450101")

    end_date = results[list(models.keys())[0]]["prediction"].time_index[-1]

    historical_df = series.pd_dataframe()

    historical_df = historical_df[historical_df.index >= start_date]

    ax.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    ax.set_xlim(start_date, end_date)

    ax.set_ylim(0, series.values().max() * 1.1)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    lines = {}

    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot(
            [],
            [],
            label=f"{name} (MAPE: {results[name]['mape']:.2f}%)",
            lw=2,
            color=color,
        )
        lines[name] = line

    plt.title("Yearly Sunspots Forecast - 20 Years into Future", pad=20)

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=20, interval=200, blit=True, repeat=True
    )

    anim.save("sunspots_forecast.gif", writer="pillow", fps=5, dpi=100)

    plt.close()

    print("\nFinal predicted values (20 years into future):")

    for name, result in results.items():
        final_value = result["prediction"].values()[-1][0]
        print(f"{name}: {final_value:.2f} (MAPE: {result['mape']:.2f}%)")

    plt.figure(figsize=(15, 7))

    plt.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        plt.plot(
            pred.time_index,
            pred.values(),
            label=f"{name} (MAPE: {results[name]['mape']:.2f}%)",
            color=color,
        )

    plt.xlim(start_date, end_date)

    plt.ylim(
        0,
        max(
            historical_df["Sunspots"].max(),
            max([result["prediction"].values().max() for result in results.values()]),
        )
        * 1.1,
    )

    plt.title("Sunspots Forecast - All Models")

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig("sunspots_forecast_all_models.png", bbox_inches="tight", dpi=300)

    plt.close()

    print("\nModel Comparison:")

    print("-" * 50)

    print(f"{'Model':<25} {'MAPE':>10}")

    print("-" * 50)

    for name, result in results.items():
        print(f"{name:<25} {result['mape']:>10.2f}%")

    print("-" * 50)

    df = pd.read_csv(
        "SN_m_tot_V2.0.csv",
        delimiter=";",
        header=None,
        names=[
            "Year",
            "Month",
            "Decimal_Date",
            "Sunspots",
            "Std",
            "Observations",
            "Definitive",
        ],
        na_values=["*******"],
    )

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )

    df.set_index("Date", inplace=True)

    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])

    df_yearly = df.resample("YE")["Sunspots"].mean().reset_index()

    df_yearly.columns = ["Year", "Sunspots"]

    series = TimeSeries.from_dataframe(df_yearly, "Year", "Sunspots")

    train, val = series.split_before(pd.Timestamp("19800101"))

    print("training set: ", len(train))

    print("validation set: ", len(val))

    models = {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=11,
        ),
        "Theta": Theta(seasonality_period=11, season_mode=SeasonalityMode.ADDITIVE),
        "Linear Regression": LinearRegressionModel(lags=11, output_chunk_length=20),
        "Random Forest": RandomForest(
            lags=[-1, -2, -11],
            output_chunk_length=20,
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            criterion="absolute_error",
        ),
        "TBATS": TBATS(
            seasonal_periods=[11],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(input_chunk_length=10, output_chunk_length=20, n_epochs=80),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        pred = model.predict(20)
        results[name] = {"prediction": pred}

    fig = plt.figure(figsize=(15, 7))

    ax = plt.axes()

    start_date = pd.Timestamp("19450101")

    end_date = results[list(models.keys())[0]]["prediction"].time_index[-1]

    historical_df = series.pd_dataframe()

    historical_df = historical_df[historical_df.index >= start_date]

    ax.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
    )

    ax.set_xlim(start_date, end_date)

    ax.set_ylim(0, series.values().max() * 1.1)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    lines = {}

    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot([], [], label=f"{name}", lw=2, color=color)
        lines[name] = line

    plt.title("Yearly Sunspots Forecast - 20 Years into Future", pad=20)

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=20, interval=200, blit=True, repeat=True
    )

    anim.save("sunspots_forecast.gif", writer="pillow", fps=5, dpi=100)

    plt.close()

    print("\nFinal predicted values (20 years into future):")

    for name, result in results.items():
        final_value = result["prediction"].values()[-1][0]
        print(f"{name}: {final_value:.2f}")

    plt.figure(figsize=(15, 7))

    historical_df.plot(label="Historical", color="black", alpha=0.6)

    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        pred.plot(label=name, color=color)

    plt.title("Sunspots Forecast - All Models")

    plt.xlabel("Time")

    plt.ylabel("Sunspots")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig("sunspots_forecast_all_models.png")

    plt.close()

    print("Running analysis on synthetic data...")

    synthetic_series, synthetic_predictions, synthetic_model = run_analysis("synthetic")

    print("\nRunning analysis on sunspot data...")

    sunspot_series, sunspot_predictions, sunspot_model = run_analysis("sunspot")

    print("Running analysis on synthetic data...")

    synthetic_series, synthetic_results, synthetic_split = run_analysis("synthetic")

    print("\nRunning analysis on sunspot data...")

    sunspot_series, sunspot_results, sunspot_split = run_analysis("sunspot")

    synthetic_table = create_metrics_table(synthetic_results, "Synthetic")

    sunspot_table = create_metrics_table(sunspot_results, "Sunspot")

    combined_table = synthetic_table.append(sunspot_table)

    print("\nModel Performance Metrics:")

    print(combined_table)

    combined_table.to_csv("model_metrics.csv")

    main()


if __name__ == "__main__":
    main()
