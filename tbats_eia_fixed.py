import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from tbats import TBATS

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    if not p.exists():
        raise FileNotFoundError("EIA CSV not found")
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s


def rolling_origin_tbats(s: pd.Series, horizon: int, n_splits: int, season: int):
    idx = np.arange(len(s))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    last_true, last_pred = None, None
    for train_idx, test_idx in tscv.split(idx):
        end = train_idx[-1]
        y_tr = s.iloc[: end + 1]
        y_te = s.iloc[end + 1 : end + 1 + horizon]
        if len(y_te) == 0:
            continue
        estimator = TBATS(seasonal_periods=[season])
        model = estimator.fit(y_tr.values)
        yhat = model.forecast(steps=len(y_te))
        maes.append(mean_absolute_error(y_te.values, yhat))
        last_true = y_te
        last_pred = pd.Series(yhat, index=y_te.index)
    return float(np.mean(maes)), last_true, last_pred


def main():
    cfg = Config()
    s = load_series(cfg)
    mean_mae, y_true, y_pred = rolling_origin_tbats(s, cfg.horizon, cfg.n_splits, cfg.season)
    print(f"TBATS mean MAE: {mean_mae}")

    plt.figure(figsize=(9,4))
    plt.plot(s.index, s.values, label="history", alpha=0.6)
    if y_pred is not None:
        plt.plot(y_pred.index, y_pred.values, label="TBATS last fold")
    plt.legend()
    save_fig("eia_tbats_last_fold.png")

    # Save last fold predictions for integration
    if y_pred is not None:
        out = pd.DataFrame({
            'model': 'TBATS',
            'date': y_pred.index,
            'true': y_true.values,
            'pred': y_pred.values,
        })
        out.to_csv('eia_preds_tbats.csv', index=False)

if __name__ == "__main__":
    main()
