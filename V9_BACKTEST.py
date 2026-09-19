import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# EME V9 — HISTORICAL BACKTEST
# ============================================================

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

START_DATE = "2006-01-01"

INITIAL_CAPITAL = 10000


# ============================================================
# V9 ALLOCATION RULE
# ============================================================

def allocation_from_score(score):

    if score < 65:
        return 0.00

    elif score < 75:
        return 0.40

    elif score < 85:
        return 0.60

    elif score < 95:
        return 0.80

    else:
        return 1.00


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    print("\nDownloading historical data...")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise RuntimeError("No market data downloaded.")

    return data


# ============================================================
# CLOSE PRICES
# ============================================================

def get_close_prices(data):

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()

    close = close.dropna(how="all")

    return close


# ============================================================
# SCORE AT DATE
# ============================================================

def calculate_scores_at_date(close, date):

    historical = close.loc[:date]

    if len(historical) < 200:
        return None, False

    spy = historical["SPY"]

    spy_sma200 = spy.rolling(200).mean()

    market_on = (
        spy.iloc[-1] > spy_sma200.iloc[-1]
    )

    scores = {}

    for ticker in TICKERS:

        price = historical[ticker]

        sma200 = price.rolling(200).mean()

        roc10 = price.pct_change(10)

        roc20 = price.pct_change(20)

        volatility = (
            price.pct_change()
            .rolling(20)
            .std()
        )

        trend_score = 0

        if price.iloc[-1] > sma200.iloc[-1]:
            trend_score += 20

        if roc10.iloc[-1] > 0:
            trend_score += 10

        if roc20.iloc[-1] > 0:
            trend_score += 10

        relative_score = 0

        asset_return_20 = (
            price.pct_change(20).iloc[-1]
        )

        spy_return_20 = (
            spy.pct_change(20).iloc[-1]
        )

        if asset_return_20 > spy_return_20:
            relative_score += 20

        asset_return_60 = (
            price.pct_change(60).iloc[-1]
        )

        spy_return_60 = (
            spy.pct_change(60).iloc[-1]
        )

        if asset_return_60 > spy_return_60:
            relative_score += 20

        risk_score = 0

        median100 = (
            volatility
            .rolling(100)
            .median()
            .iloc[-1]
        )

        median200 = (
            volatility
            .rolling(200)
            .median()
            .iloc[-1]
        )

        if volatility.iloc[-1] < median100:
            risk_score += 10

        if volatility.iloc[-1] < median200:
            risk_score += 10

        total_score = (
            trend_score
            + relative_score
            + risk_score
        )

        scores[ticker] = float(total_score)

    return scores, market_on


# ============================================================
# SELECT ASSET
# ============================================================

def select_asset(scores, market_on):

    if not market_on:
        return None, 0

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    ticker, score = ranked[0]

    if score < 65:
        return None, score

    return ticker, score


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(equity):

    equity = pd.Series(equity).dropna()

    total_return = (
        equity.iloc[-1] /
        equity.iloc[0]
        - 1
    )

    years = (
        len(equity) / 252
    )

    cagr = (
        (equity.iloc[-1] /
         equity.iloc[0])
        ** (1 / years)
        - 1
    )

    daily_returns = equity.pct_change().dropna()

    volatility = daily_returns.std()

    if volatility > 0:

        sharpe = (
            daily_returns.mean() /
            volatility
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    return {
        "final": equity.iloc[-1],
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe
    }


# ============================================================
# V9 BACKTEST
# ============================================================

def run_v9_backtest(close):

    dates = close.index

    capital = INITIAL_CAPITAL

    equity = []

    current_asset = None
    current_allocation = 0.0

    rotations = 0

    cash_days = 0

    decisions = []

    for i in range(200, len(dates) - 1):

        date = dates[i]

        next_date = dates[i + 1]

        scores, market_on = (
            calculate_scores_at_date(
                close,
                date
            )
        )

        if scores is None:
            continue

        selected_asset, score = (
            select_asset(
                scores,
                market_on
            )
        )

        allocation = allocation_from_score(
            score
        )

        if selected_asset is None:
            allocation = 0.0

        if selected_asset != current_asset:
            if current_asset is not None:
                rotations += 1

        current_asset = selected_asset
        current_allocation = allocation

        if allocation == 0:
            cash_days += 1

        # ----------------------------------------------------
        # NEXT-DAY RETURN
        # ----------------------------------------------------

        if selected_asset is None:

            daily_return = 0.0

        else:

            asset_today = close.loc[
                date,
                selected_asset
            ]

            asset_next = close.loc[
                next_date,
                selected_asset
            ]

            asset_return = (
                asset_next /
                asset_today
                - 1
            )

            daily_return = (
                allocation *
                asset_return
            )

        capital *= (
            1 +
            daily_return
        )

        equity.append(capital)

        decisions.append({
            "date": date,
            "asset": (
                selected_asset
                if selected_asset
                else "CASH"
            ),
            "score": score,
            "allocation": allocation,
            "capital": capital
        })

    equity_series = pd.Series(
        equity,
        index=dates[200:-1]
    )

    metrics = calculate_metrics(
        equity_series
    )

    decisions_df = pd.DataFrame(
        decisions
    )

    return (
        metrics,
        decisions_df,
        equity_series
    )


# ============================================================
# BUY & HOLD SPY
# ============================================================

def run_spy_buy_hold(close):

    spy = close["SPY"].dropna()

    spy = spy.iloc[200:]

    initial = spy.iloc[0]

    equity = (
        INITIAL_CAPITAL *
        spy /
        initial
    )

    metrics = calculate_metrics(
        equity
    )

    return metrics, equity


# ============================================================
# REPORT
# ============================================================

def print_metrics(name, metrics):

    print("\n" + "=" * 70)

    print(name)

    print("=" * 70)

    print(
        f"Final capital     : "
        f"€{metrics['final']:,.2f}"
    )

    print(
        f"Total return      : "
        f"{metrics['total_return'] * 100:.2f}%"
    )

    print(
        f"CAGR              : "
        f"{metrics['cagr'] * 100:.2f}%"
    )

    print(
        f"Max drawdown      : "
        f"{metrics['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe            : "
        f"{metrics['sharpe']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    close = get_close_prices(
        download_data()
    )

    print("\nRunning V9 historical backtest...")

    (
        v9_metrics,
        decisions,
        v9_equity
    ) = run_v9_backtest(
        close
    )

    spy_metrics, spy_equity = (
        run_spy_buy_hold(
            close
        )
    )

    print_metrics(
        "EME V9 ALLOCATION",
        v9_metrics
    )

    print_metrics(
        "SPY BUY & HOLD",
        spy_metrics
    )

    rotations = (
        decisions["asset"]
        .ne(
            decisions["asset"].shift()
        )
        .sum()
    )

    cash_percentage = (
        decisions["allocation"]
        .eq(0)
        .mean()
    )

    print("\n" + "=" * 70)

    print("V9 BEHAVIOR")

    print("=" * 70)

    print(
        f"Rotations         : "
        f"{rotations}"
    )

    print(
        f"Time in CASH      : "
        f"{cash_percentage * 100:.2f}%"
    )

    print("\nAsset distribution:")

    distribution = (
        decisions["asset"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    for asset, percentage in distribution.items():

        print(
            f"{asset:5s} : "
            f"{percentage:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    decisions.to_csv(
        "eme_v9_backtest_decisions.csv",
        index=False
    )

    comparison = pd.DataFrame([
        {
            "strategy": "EME V9",
            **v9_metrics
        },
        {
            "strategy": "SPY BUY & HOLD",
            **spy_metrics
        }
    ])

    comparison.to_csv(
        "eme_v9_backtest_results.csv",
        index=False
    )

    print(
        "\nResults saved:"
    )

    print(
        "eme_v9_backtest_results.csv"
    )

    print(
        "eme_v9_backtest_decisions.csv"
    )


if __name__ == "__main__":

    main()
