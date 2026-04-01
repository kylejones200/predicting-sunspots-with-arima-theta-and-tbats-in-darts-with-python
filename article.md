# Predicting Sunspots with ARIMA, Theta, and TBATS in DARTS with Python Using DARTS to forecast solar cycles

::::### Predicting Sunspots with ARIMA, Theta, and TBATS in DARTS with Python 

#### Using DARTS to forecast solar cycles
Sunspot observations are one of science's longest continuous datasets
and a good case study for time series analysis. Let's explore how modern
forecasting techniques can help predict future solar activity.

The sunspot dataset, maintained by the [WDC-SILSO, Royal Observatory of
Belgium, Brussels](https://www.sidc.be/SILSO/datafiles), contains monthly observations since
1749.


<figcaption>This isn an earlier version of the project. I was
experimenting with smoothing options.</figcaption>


In our analysis, we transform this monthly data into yearly averages to
better observe the Sun's long-term patterns, particularly the roughly
11-year solar cycle first discovered by Heinrich Schwabe in 1843. This
dataset has been extensively studied by researchers worldwide, from
Daines Analytics' SARIMA modeling to Python's Gurus' work with the Darts
library. Our analysis builds upon this foundation.

Our analysis implements multiple forecasting models:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA, Theta, TBATS
from darts.metrics import mape, rmse
from darts.utils.utils import SeasonalityMode
import warnings

warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self):
        self.models = {
            'ARIMA': ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 11)),
            'Theta': Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=11),
            'TBATS': TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[11])
        }

    def _load_csv_data(self, filepath):
        """Load and prepare sunspot data."""
        try:
            # Read the CSV file with the new format
            df = pd.read_csv(filepath)
            
            # Split the Month column into Year and Month
            df[['Year', 'Month']] = df['Month'].str.split('-', expand=True)
            
            # Convert to proper datetime
            df['Date'] = pd.to_datetime(df['Year'] + '-' + df['Month'] + '-01')
            
            # Replace 0 values with 1 to avoid log transform issues
            df['Sunspot'] = np.where(df['Sunspot'] == 0, 1, df['Sunspot'])
            
            # Convert to yearly averages
            df_yearly = df.groupby(df['Date'].dt.year)['Sunspot'].mean().reset_index()
            df_yearly['Date'] = pd.to_datetime(df_yearly['Date'].astype(str) + '-01-01')
            
            # Create TimeSeries object
            series = TimeSeries.from_dataframe(df_yearly, 'Date', 'Sunspot')
            
            return series
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def load_data(self, filepath=None):
        return self._load_csv_data(filepath)

    def analyze(self, series, train_test_split=0.8):
        """Train models and evaluate predictions"""
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
                results[name] = {'prediction': pred, **metrics}
                self._print_metrics(name, metrics)
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results, train, test

    def _calculate_metrics(self, actual, predicted):
        try:
            return {
                'MAPE': mape(actual, predicted),
                'RMSE': rmse(actual, predicted),
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {'MAPE': np.nan, 'RMSE': np.nan}

    def _print_metrics(self, model_name, metrics):
        print(f"{model_name} Performance:")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric}: {value:.2f}")

    def plot_results(self, series, results, train, test, save_path='forecast.png'):
        """Plot predictions from all models"""
        if series is None or results is None:
            print("No data to plot")
            return

        plt.figure(figsize=(15, 7))
        train.plot(label='Training', alpha=0.6)
        test.plot(label='Test', alpha=0.6)
        
        if results:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
            for (name, result), color in zip(results.items(), colors):
                if 'prediction' in result:
                    result['prediction'].plot(
                        label=f'{name} (MAPE: {result["MAPE"]:.1f}%)',
                        color=color
                    )

        plt.title('Sunspot Number Forecasting Comparison')
        plt.xlabel('Time')
        plt.ylabel('Sunspot Number')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def main():
    analyzer = TimeSeriesAnalyzer()
    
    real_series = analyzer.load_data("SN_m_tot_V2.0.csv")  
    real_results, real_train, real_test = analyzer.analyze(real_series)
    analyzer.plot_results(real_series, real_results, 
                                real_train, real_test, 'sunspot_forecast.png')

if __name__ == "__main__":
    main()
```


This real-world example demonstrates both the power and limitations of
time series forecasting. While we can capture major patterns, solar
activity's inherent complexity means predictions should be used as
guidance rather than absolute forecasts.
::::Update (2025--11--04) I created another project that isn't about
sunspots but uses the same ML methods.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from darts import TimeSeries
from darts.models import ARIMA, Theta

np.random.seed(42)
plt.rcParams.update({
    'axes.grid': False,'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5


def load_series(cfg: Config) -> TimeSeries:
    p = Path(cfg.csv_path)
    if not p.exists():
        raise FileNotFoundError("EIA CSV not found")
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    s = TimeSeries.from_dataframe(df, time_col="date", value_cols="value", freq=cfg.freq).astype(float)
    return s


def rolling_origin_eval(ts: TimeSeries, model_ctor, horizon: int, n_splits: int):
    values = ts.values().ravel()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    idx = np.arange(len(values))
    maes = []
    last_true, last_pred = None, None
    for train_idx, test_idx in tscv.split(idx):
        end = train_idx[-1]
        cutoff = ts.time_index[end]
        y_tr, future = ts.split_after(cutoff)
        # Take the first `horizon` timestamps from the future using DateTimeIndex
        if len(future) == 0:
            continue
        end_idx = min(horizon, len(future)) - 1
        y_te = future.drop_after(future.time_index[end_idx])
        m = model_ctor()
        m.fit(y_tr)
        fc = m.predict(len(y_te))
        mae = mean_absolute_error(y_te.values().ravel(), fc.values().ravel())
        maes.append(mae)
        last_true, last_pred = y_te, fc
    return np.mean(maes), (last_true, last_pred)


def maybe_load_tbats_csv():
    p = Path('eia_preds_tbats.csv')
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df

def main():
    cfg = Config()
    ts = load_series(cfg)

    results = {}
    preds = {}

    mean_mae, (y_true, y_pred) = rolling_origin_eval(ts, lambda: ARIMA(p=1, d=1, q=1), cfg.horizon, cfg.n_splits)
    results['ARIMA mean MAE'] = mean_mae; preds['ARIMA'] = (y_true, y_pred)

    mean_mae, (y_true, y_pred) = rolling_origin_eval(ts, lambda: Theta(), cfg.horizon, cfg.n_splits)
    results['Theta mean MAE'] = mean_mae; preds['Theta'] = (y_true, y_pred)

    tbats_df = maybe_load_tbats_csv()
    if tbats_df is not None and not tbats_df.empty:
        tbats_mae = mean_absolute_error(tbats_df['true'], tbats_df['pred'])
        results['TBATS mean MAE (from CSV)'] = tbats_mae

    print("\n".join(f"{k}: {v}" for k,v in results.items()))

    # Tufte-style final figure: 2024 history, dashed vline at Jan 2025, forecasts/actuals Jan–Aug 2025 only
    start_2024 = pd.Period('2024-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period('2024-12', freq='M').start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period('2025-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period('2025-08', freq='M').start_time + pd.offsets.MonthBegin(0)

    s = ts.to_series()
    y_hist = s.loc[start_2024:end_2024]
    y_act = s.loc[jan_2025:aug_2025]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)

    # Collect forecasts restricted to Jan–Aug 2025
    end_labels = []
    for name, (yt, yp) in preds.items():
        f_idx = yp.time_index
        mask = (f_idx >= jan_2025) & (f_idx <= aug_2025)
        if mask.any():
            f = pd.Series(yp.values().ravel()[mask], index=f_idx[mask])
            ax.plot(f.index, f.values, color="#000000", lw=2.0, alpha=0.85)
            end_labels.append((f.index[-1], f.values[-1], name))

    if tbats_df is not None and not tbats_df.empty:
        f_tb = tbats_df[(tbats_df['date'] >= jan_2025) & (tbats_df['date'] <= aug_2025)]
        if not f_tb.empty:
            ax.plot(f_tb['date'], f_tb['pred'], color="#000000", lw=1.6, alpha=0.6)
            end_labels.append((f_tb['date'].iloc[-1], f_tb['pred'].iloc[-1], 'TBATS'))

    # Minimal y-axis
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        ax.set_xlabel('')

    # End-of-line labels
    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666')
    if len(y_act):
        ax.annotate('Actual (Jan–Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444')
    for x, yv, name in end_labels:
        ax.annotate(name, xy=(x, yv), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000')

    ax.set_title('EIA Net Generation — ARIMA/Theta/TBATS last-fold forecasts Jan–Aug 2025')
    save_fig("eia_darts_tbats_last_fold.png")

    # Save ARIMA/Theta last fold predictions for reproducibility
    rows = []
    for name,(yt,yp) in preds.items():
        d = pd.DataFrame({
            'model': name,
            'date': yt.time_index,
            'true': yt.values().ravel(),
            'pred': yp.values().ravel(),
        })
        rows.append(d)
    pd.concat(rows).to_csv('eia_preds_darts.csv', index=False)

if __name__ == "__main__":
    main()
```
::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[February 3, 2025](https://medium.com/p/bdbb117ae411).

[Canonical
link](https://medium.com/@kyle-t-jones/predicting-sunspots-with-arima-theta-and-tbats-in-darts-with-python-bdbb117ae411)

Exported from [Medium](https://medium.com) on November 10, 2025.
