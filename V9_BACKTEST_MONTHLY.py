import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# EME V9 — MONTHLY BACKTEST
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

    print("\nDownloading market data...")

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
# CLOSE DATA
# ============================================================

def get_close_prices(data):

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()

    else:
        close = data[["Close"]].copy()
        close.columns = TICKERS

    close = close.dropna(how="all")

    return close


# ============================================================
# MONTH-END TRADING DATES
# ============================================================

def get_month_end_dates(close):

    month_end_dates = (
        close.index.to_series()
        .groupby(close.index.to_period("M"))
        .max()
        .tolist()
    )

    return month_end_dates


# ============================================================
# CALCULATE SCORE AT A SPECIFIC DATE
# ============================================================

def calculate_scores_at_date(close, date):

    history = close.loc[:date]

    if len(history) < 200:
        return None, False

    scores = {}

    spy = history["SPY"]

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    spy_sma200 = spy.rolling(200).mean()

    market_on = (
        spy.iloc[-1] > spy_sma200.iloc[-1]
    )

    # --------------------------------------------------------
    # ASSET SCORES
    # --------------------------------------------------------

    for ticker in TICKERS:

        price = history[ticker]

        sma200 = price.rolling(200).mean()

        roc10 = price.pct_change(10)

        roc20 = price.pct_change(20)

        volatility = (
            price.pct_change()
            .rolling(20)
            .std()
        )

        # -----------------------------
        # TREND / MOMENTUM
        # -----------------------------

        trend_score = 0

        if price.iloc[-1] > sma200.iloc[-1]:
            trend_score += 20

        if roc10.iloc[-1] > 0:
            trend_score += 10

        if roc20.iloc[-1] > 0:
            trend_score += 10

        # -----------------------------
        # RELATIVE STRENGTH
        # -----------------------------

        relative_score = 0

        asset_return_20 = price.pct_change(20).iloc[-1]

        spy_return_20 = spy.pct_change(20).iloc[-1]

        if asset_return_20 > spy_return_20:
            relative_score += 20

        asset_return_60 = price.pct_change(60).iloc[-1]

        spy_return_60 = spy.pct_change(60).iloc[-1]

        if asset_return_60 > spy_return_60:
            relative_score += 20

        # -----------------------------
        # RISK
        # -----------------------------

        risk_score = 0

        volatility_100_median = (
            volatility
            .rolling(100)
            .median()
            .iloc[-1]
        )

        volatility_200_median = (
            volatility
            .rolling(200)
            .median()
            .iloc[-1]
        )

        if volatility.iloc[-1] < volatility_100_median:
            risk_score += 10

        if volatility.iloc[-1] < volatility_200_median:
            risk_score += 10

        total_score = (
            trend_score
            + relative_score
            + risk_score
        )

        scores[ticker] = {
            "score": float(total_score),
            "trend": trend_score,
            "relative": relative_score,
            "risk": risk_score
        }

    return scores, market_on


# ============================================================
# SELECT ASSET
# ============================================================

def select_asset(scores, market_on):

    if not market_on:
        return None, 0

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    ticker, data = ranked[0]

    if data["score"] < 65:
        return None, data["score"]

    return ticker, data["score"]


# ============================================================
# MONTHLY BACKTEST
# ============================================================

def run_backtest(close):

    month_end_dates = get_month_end_dates(close)

    capital = INITIAL_CAPITAL

    equity_curve = []

    decisions = []

    rotations = 0

    cash_months = 0

    previous_asset = None

    # --------------------------------------------------------
    # Each decision is made ONLY using information available
    # at the current month-end.
    #
    # Return is then measured from current month-end to the
    # following month-end.
    # --------------------------------------------------------

    for i in range(len(month_end_dates) - 1):

        decision_date = month_end_dates[i]

        next_date = month_end_dates[i + 1]

        scores, market_on = calculate_scores_at_date(
            close,
            decision_date
        )

        if scores is None:
            continue

        selected_asset, score = select_asset(
            scores,
            market_on
        )

        allocation = allocation_from_score(score)

        if selected_asset is None:
            allocation = 0.0

        # ----------------------------------------------------
        # ROTATION COUNT
        # ----------------------------------------------------

        current_asset = (
            selected_asset
            if allocation > 0
            else "CASH"
        )

        if (
            previous_asset is not None
            and current_asset != previous_asset
        ):
            rotations += 1

        previous_asset = current_asset

        # ----------------------------------------------------
        # MONTHLY RETURN
        # ----------------------------------------------------

        if selected_asset is None:

            asset_return = 0.0

            cash_months += 1

        else:

            start_price = close.loc[
                decision_date,
                selected_asset
            ]

            end_price = close.loc[
                next_date,
                selected_asset
            ]

            asset_return = (
                end_price / start_price
            ) - 1

        portfolio_return = (
            allocation * asset_return
        )

        capital *= (
            1 + portfolio_return
        )

        # ----------------------------------------------------
        # SAVE DECISION
        # ----------------------------------------------------

        decisions.append({
            "decision_date": decision_date.strftime("%Y-%m-%d"),
            "next_date": next_date.strftime("%Y-%m-%d"),
            "market_on": market_on,
            "asset": (
                selected_asset
                if selected_asset
                else "CASH"
            ),
            "score": round(score, 2),
            "allocation_pct": round(
                allocation * 100,
                2
            ),
            "asset_return_pct": round(
                asset_return * 100,
                4
            ),
            "portfolio_return_pct": round(
                portfolio_return * 100,
                4
            ),
            "capital": round(capital, 2)
        })

        equity_curve.append({
            "date": next_date,
            "capital": capital
        })

    return (
        capital,
        rotations,
        cash_months,
        decisions,
        equity_curve
    )


# ============================================================
# SPY BUY & HOLD
# ============================================================

def calculate_spy_buy_hold(close):

    spy = close["SPY"].dropna()

    start_price = spy.iloc[0]

    end_price = spy.iloc[-1]

    final_capital = (
        INITIAL_CAPITAL
        * end_price
        / start_price
    )

    returns = spy.pct_change().dropna()

    cumulative = (
        spy / spy.iloc[0]
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    years = (
        spy.index[-1] - spy.index[0]
    ).days / 365.25

    cagr = (
        final_capital / INITIAL_CAPITAL
    ) ** (1 / years) - 1

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    )

    return {
        "final_capital": final_capital,
        "total_return": (
            final_capital / INITIAL_CAPITAL
        ) - 1,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe
    }


# ============================================================
# V9 METRICS
# ============================================================

def calculate_v9_metrics(
    equity_curve,
    final_capital
):

    equity = pd.DataFrame(equity_curve)

    equity = equity.set_index("date")

    capital_series = equity["capital"]

    cumulative = (
        capital_series
        / INITIAL_CAPITAL
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    years = (
        capital_series.index[-1]
        - capital_series.index[0]
    ).days / 365.25

    cagr = (
        final_capital / INITIAL_CAPITAL
    ) ** (1 / years) - 1

    monthly_returns = (
        capital_series
        .pct_change()
        .dropna()
    )

    sharpe = (
        monthly_returns.mean()
        / monthly_returns.std()
        * np.sqrt(12)
    )

    return {
        "final_capital": final_capital,
        "total_return": (
            final_capital / INITIAL_CAPITAL
        ) - 1,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("        EME V9 — MONTHLY BACKTEST")
    print("=" * 70)

    data = download_data()

    close = get_close_prices(data)

    (
        final_capital,
        rotations,
        cash_months,
        decisions,
        equity_curve
    ) = run_backtest(close)

    v9 = calculate_v9_metrics(
        equity_curve,
        final_capital
    )

    spy = calculate_spy_buy_hold(close)

    total_months = len(decisions)

    cash_pct = (
        cash_months / total_months
        if total_months > 0
        else 0
    )

    print("\n")
    print("=" * 70)
    print("V9 MONTHLY")
    print("=" * 70)

    print(
        f"Final capital     : "
        f"€{v9['final_capital']:,.2f}"
    )

    print(
        f"Total return      : "
        f"{v9['total_return'] * 100:.2f}%"
    )

    print(
        f"CAGR              : "
        f"{v9['cagr'] * 100:.2f}%"
    )

    print(
        f"Max drawdown      : "
        f"{v9['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe            : "
        f"{v9['sharpe']:.2f}"
    )

    print("\n")
    print("=" * 70)
    print("SPY BUY & HOLD")
    print("=" * 70)

    print(
        f"Final capital     : "
        f"€{spy['final_capital']:,.2f}"
    )

    print(
        f"Total return      : "
        f"{spy['total_return'] * 100:.2f}%"
    )

    print(
        f"CAGR              : "
        f"{spy['cagr'] * 100:.2f}%"
    )

    print(
        f"Max drawdown      : "
        f"{spy['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe            : "
        f"{spy['sharpe']:.2f}"
    )

    print("\n")
    print("=" * 70)
    print("V9 BEHAVIOR")
    print("=" * 70)

    print(
        f"Monthly decisions : {total_months}"
    )

    print(
        f"Rotations         : {rotations}"
    )

    print(
        f"Time in CASH      : "
        f"{cash_pct * 100:.2f}%"
    )

    # --------------------------------------------------------
    # ASSET DISTRIBUTION
    # --------------------------------------------------------

    decisions_df = pd.DataFrame(decisions)

    distribution = (
        decisions_df["asset"]
        .value_counts(normalize=True)
        * 100
    )

    print("\nAsset distribution:")

    for asset, pct in distribution.items():

        print(
            f"{asset:5s}: {pct:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame([{
        "system": "V9_MONTHLY",
        "final_capital": v9["final_capital"],
        "total_return": v9["total_return"],
        "cagr": v9["cagr"],
        "max_drawdown": v9["max_drawdown"],
        "sharpe": v9["sharpe"],
        "rotations": rotations,
        "cash_pct": cash_pct
    }, {
        "system": "SPY_BUY_HOLD",
        "final_capital": spy["final_capital"],
        "total_return": spy["total_return"],
        "cagr": spy["cagr"],
        "max_drawdown": spy["max_drawdown"],
        "sharpe": spy["sharpe"],
        "rotations": 0,
        "cash_pct": 0
    }])

    results.to_csv(
        "eme_v9_monthly_results.csv",
        index=False
    )

    decisions_df.to_csv(
        "eme_v9_monthly_decisions.csv",
        index=False
    )

    print("\nResults saved:")
    print("eme_v9_monthly_results.csv")
    print("eme_v9_monthly_decisions.csv")


if __name__ == "__main__":
    main()
