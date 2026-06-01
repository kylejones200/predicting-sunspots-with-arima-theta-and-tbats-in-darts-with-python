"""Load sunspot CSV or synthetic series for Darts."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries


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
            return series, future_cov
        except FileNotFoundError:
            print(
                "Sunspot data file not found. Please ensure SN_m_tot_V2.0.csv exists."
            )
            return None, None

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
        return series, future_cov
