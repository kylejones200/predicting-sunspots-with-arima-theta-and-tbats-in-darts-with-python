"""Sunspot forecasting with Darts (monthly hold-out + yearly long-horizon).

Consolidated from `sunspots with darts.ipynb`:
  - Monthly series: several models, validation MAPE, frame-per-model GIF.
  - Yearly series: resampled monthly data, 20-step-ahead forecasts, animated GIF + static PNG.

Install: pip install darts u8darts[all]  # or your project's darts extras (torch, etc.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from darts.models import (
    ExponentialSmoothing,
    LinearRegressionModel,
    NHiTSModel,
    RandomForest,
    RNNModel,
    TBATS,
    Theta,
)
from darts.utils.utils import ModelMode, SeasonalityMode


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_monthly_sunspots(csv_path: Path) -> TimeSeries:
    df = pd.read_csv(csv_path)
    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])
    df = df.iloc[-800:].reset_index(drop=True)
    return TimeSeries.from_dataframe(df, "Month", "Sunspots")


def build_monthly_models() -> dict:
    return {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=124,
        ),
        "Theta": Theta(
            seasonality_period=124,
            season_mode=SeasonalityMode.ADDITIVE,
        ),
        "Linear Regression": LinearRegressionModel(
            lags=124,
            output_chunk_length=20,
        ),
        "Random Forest": RandomForest(
            lags=[-1, -12, -124],
            output_chunk_length=20,
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            criterion="absolute_error",
        ),
        "RNN": RNNModel(
            model="LSTM",
            training_length=240,
            input_chunk_length=120,
            n_epochs=200,
            batch_size=20,
            optimizer_kwargs={"lr": 1e-3},
        ),
        "TBATS": TBATS(
            seasonal_periods=[12],
            use_box_cox=True,
            use_trend=True,
            use_arma_errors=True,
        ),
        "NHiTS": NHiTSModel(
            input_chunk_length=100,
            output_chunk_length=20,
            n_epochs=80,
        ),
    }


def run_monthly_validation(out_dir: Path) -> None:
    csv_path = out_dir / "monthly-sunspots.csv"
    series = load_monthly_sunspots(csv_path)
    train, val = series.split_before(pd.Timestamp("19800101"))
    print("training set:", len(train))
    print("validation set:", len(val))

    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)

    models = build_monthly_models()
    results: dict = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "RNN":
            model.fit(train_transformed, verbose=True)
            pred = model.predict(len(val))
            pred = transformer.inverse_transform(pred)
        else:
            model.fit(train)
            pred = model.predict(len(val))

        mape_score = float(np.round(mape(pred, val), 2))
        results[name] = {"prediction": pred, "mape": mape_score}

    fig = plt.figure(figsize=(15, 7))
    ax = plt.axes()
    ax.set_xlim(series.time_index.min(), series.time_index.max())
    ax.set_ylim(0, float(series.values().max()) * 1.1)
    series.plot(label="Actual", ax=ax, color="black", alpha=0.6)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
    lines: dict = {}
    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot(
            [],
            [],
            label=f"{name} (MAPE: {results[name]['mape']}%)",
            lw=2,
            color=color,
        )
        lines[name] = line

    plt.title("Sunspots forecast — validation predictions", pad=20)
    plt.xlabel("Time")
    plt.ylabel("Sunspots")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    def animate(frame: int):
        name = list(models.keys())[frame]
        pred = results[name]["prediction"]
        lines[name].set_data(pred.time_index, pred.values())
        return (lines[name],)

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(models),
        interval=1000,
        blit=True,
        repeat=True,
    )
    anim.save(out_dir / "model_predictions.gif", writer="pillow", fps=1, dpi=100)
    plt.close(fig)

    print("\nFinal MAPE scores (validation):")
    for name, result in results.items():
        print(f"  {name}: {result['mape']}%")


def load_yearly_sunspots(csv_path: Path) -> TimeSeries:
    df = pd.read_csv(csv_path)
    df["Sunspots"] = np.where(df["Sunspots"] == 0, 1, df["Sunspots"])
    df = df.iloc[-800:].reset_index(drop=True)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")
    df_yearly = df.resample("YE").mean()
    return TimeSeries.from_dataframe(df_yearly, value_cols="Sunspots")


def build_yearly_models() -> dict:
    return {
        "Exponential Smoothing": ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=11,
        ),
        "Theta": Theta(
            seasonality_period=11,
            season_mode=SeasonalityMode.ADDITIVE,
        ),
        "Linear Regression": LinearRegressionModel(
            lags=11,
            output_chunk_length=20,
        ),
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
        "NHiTS": NHiTSModel(
            input_chunk_length=10,
            output_chunk_length=20,
            n_epochs=80,
        ),
    }


def run_yearly_future_forecast(out_dir: Path, horizon: int = 20) -> None:
    csv_path = out_dir / "monthly-sunspots.csv"
    series = load_yearly_sunspots(csv_path)
    train, val = series.split_before(pd.Timestamp("19800101"))
    print("\n(yearly) training set:", len(train))
    print("(yearly) validation set:", len(val))

    models = build_yearly_models()
    results: dict = {}
    for name, model in models.items():
        print(f"\n(yearly) Training {name}...")
        model.fit(train)
        pred = model.predict(horizon)
        results[name] = {"prediction": pred}

    first_pred = results[next(iter(models))]["prediction"]
    end_date = first_pred.time_index[-1]

    fig = plt.figure(figsize=(15, 7))
    ax = plt.axes()
    start_date = pd.Timestamp("19450101")
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
    ax.set_ylim(0, float(series.values().max()) * 1.1)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
    lines: dict = {}
    for name, color in zip(models.keys(), colors):
        (line,) = ax.plot([], [], label=name, lw=2, color=color)
        lines[name] = line

    plt.title("Yearly sunspots — multi-model forecast", pad=20)
    plt.xlabel("Time")
    plt.ylabel("Sunspots")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    def animate(frame: int):
        current_idx = frame + 1
        for name in models:
            pred = results[name]["prediction"]
            lines[name].set_data(
                pred.time_index[:current_idx],
                pred.values()[:current_idx],
            )
        return list(lines.values())

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=horizon,
        interval=200,
        blit=True,
        repeat=True,
    )
    anim.save(out_dir / "sunspots_forecast.gif", writer="pillow", fps=5, dpi=100)
    plt.close(fig)

    print(f"\nFinal predicted values ({horizon} yearly steps):")
    for name, result in results.items():
        final_value = float(result["prediction"].values()[-1][0])
        print(f"  {name}: {final_value:.2f}")

    # Static overview (same style as former notebook static cell)
    plt.figure(figsize=(15, 8))
    plt.plot(
        historical_df.index,
        historical_df.values,
        label="Historical",
        color="black",
        alpha=0.6,
        linewidth=2,
    )
    for name, color in zip(models.keys(), colors):
        pred = results[name]["prediction"]
        plt.plot(pred.time_index, pred.values(), label=name, color=color, linewidth=2)
    prediction_start = historical_df.index[-1]
    plt.axvline(x=prediction_start, color="gray", linestyle="--", alpha=0.5)
    plt.title("Sunspots: historical and yearly forecasts", pad=20, fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Sunspots", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "sunspots_all_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    out_dir = _repo_root()
    run_monthly_validation(out_dir)
    run_yearly_future_forecast(out_dir, horizon=20)


if __name__ == "__main__":
    main()
