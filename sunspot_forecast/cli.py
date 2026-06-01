"""Command-line entry for sunspot forecasting demo."""

from .analyzer import TimeSeriesAnalyzer


def main() -> None:
    analyzer = TimeSeriesAnalyzer()
    print("\nAnalyzing sunspot data...")
    try:
        real_series = analyzer.load_data("SN_m_tot_V2.0.csv")
        if real_series is not None:
            real_results, real_train, real_test = analyzer.analyze(real_series)
            analyzer.plot_results(
                real_series,
                real_results,
                real_train,
                real_test,
                "sunspot_forecast.png",
            )
    except FileNotFoundError:
        print("Sunspot data file not found. Please ensure the CSV file exists.")
