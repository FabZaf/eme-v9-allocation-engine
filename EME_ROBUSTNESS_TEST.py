import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "eme_timeframe_trades.csv"
OUTPUT_FILE = "eme_robustness_results.csv"

PERIODS = [
    ("2006-2010", "2006-01-01", "2010-12-31"),
    ("2011-2015", "2011-01-01", "2015-12-31"),
    ("2016-2020", "2016-01-01", "2020-12-31"),
    ("2021-2026", "2021-01-01", "2026-12-31"),
]

ETF_LIST = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        if gains > 0:
            return np.inf
        return np.nan

    return gains / losses


def max_drawdown(returns):
    if len(returns) == 0:
        return np.nan

    equity = (1 + returns / 100).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1

    return drawdown.min() * 100


def analyze_period(df, ticker, period_name, start_date, end_date):

    x = df[
        (df["ticker"] == ticker) &
        (df["timeframe"] == "DAILY")
    ].copy()

    x["signal_date"] = pd.to_datetime(x["signal_date"])

    x = x[
        (x["signal_date"] >= pd.Timestamp(start_date)) &
        (x["signal_date"] <= pd.Timestamp(end_date))
    ].copy()

    returns = pd.to_numeric(x["return_pct"], errors="coerce").dropna()

    trades = len(returns)

    if trades == 0:
        return {
            "period": period_name,
            "ticker": ticker,
            "trades": 0,
            "win_rate_pct": np.nan,
            "avg_return_pct": np.nan,
            "median_return_pct": np.nan,
            "profit_factor": np.nan,
            "compound_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
        }

    win_rate = (returns > 0).mean() * 100
    avg_return = returns.mean()
    median_return = returns.median()
    pf = profit_factor(returns)

    compound_return = (
        (1 + returns / 100).prod() - 1
    ) * 100

    dd = max_drawdown(returns)

    return {
        "period": period_name,
        "ticker": ticker,
        "trades": trades,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_return,
        "median_return_pct": median_return,
        "profit_factor": pf,
        "compound_return_pct": compound_return,
        "max_drawdown_pct": dd,
    }


def main():

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    results = []

    for period_name, start_date, end_date in PERIODS:
        for ticker in ETF_LIST:

            results.append(
                analyze_period(
                    df,
                    ticker,
                    period_name,
                    start_date,
                    end_date
                )
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("===== EME ROBUSTNESS TEST =====")
    print()

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print()
    print(f"Risultati salvati in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
