import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# EME V9 — HISTORICAL STRESS TEST BY PERIOD
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

    return (
        close.index.to_series()
        .groupby(close.index.to_period("M"))
        .max()
        .tolist()
    )


# ============================================================
# SCORE ENGINE
# SAME LOGIC AS V9 MONTHLY
# ============================================================

def calculate_scores_at_date(close, date):

    history = close.loc[:date]

    if len(history) < 200:

        return None, False

    scores = {}

    spy = history["SPY"]

    spy_sma200 = spy.rolling(200).mean()

    market_on = (
        spy.iloc[-1] > spy_sma200.iloc[-1]
    )

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

        # ----------------------------------------------------
        # TREND / MOMENTUM
        # ----------------------------------------------------

        trend_score = 0

        if price.iloc[-1] > sma200.iloc[-1]:

            trend_score += 20

        if roc10.iloc[-1] > 0:

            trend_score += 10

        if roc20.iloc[-1] > 0:

            trend_score += 10

        # ----------------------------------------------------
        # RELATIVE STRENGTH
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RISK / VOLATILITY
        # ----------------------------------------------------

        risk_score = 0

        median_100 = (
            volatility
            .rolling(100)
            .median()
            .iloc[-1]
        )

        median_200 = (
            volatility
            .rolling(200)
            .median()
            .iloc[-1]
        )

        if volatility.iloc[-1] < median_100:

            risk_score += 10

        if volatility.iloc[-1] < median_200:

            risk_score += 10

        # ----------------------------------------------------
        # TOTAL SCORE
        # ----------------------------------------------------

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
# RUN V9 FOR ONE PERIOD
# ============================================================

def run_period(close, start_date, end_date):

    month_end_dates = get_month_end_dates(close)

    period_dates = [

        d for d in month_end_dates

        if (
            pd.Timestamp(start_date)
            <= d
            <= pd.Timestamp(end_date)
        )

    ]

    if len(period_dates) < 2:

        return None

    capital = INITIAL_CAPITAL

    equity_curve = []

    decisions = []

    rotations = 0

    cash_months = 0

    previous_asset = None

    for i in range(len(period_dates) - 1):

        decision_date = period_dates[i]

        next_date = period_dates[i + 1]

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

        decisions.append({

            "date": decision_date,

            "asset": current_asset,

            "score": score,

            "allocation": allocation,

            "portfolio_return": portfolio_return,

            "capital": capital

        })

        equity_curve.append({

            "date": next_date,

            "capital": capital

        })

    if len(equity_curve) < 2:

        return None

    equity = pd.DataFrame(
        equity_curve
    )

    equity = equity.set_index("date")

    capital_series = equity["capital"]

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

    cumulative = (
        capital_series / INITIAL_CAPITAL
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    years = (
        capital_series.index[-1]
        - capital_series.index[0]
    ).days / 365.25

    if years > 0:

        cagr = (
            capital / INITIAL_CAPITAL
        ) ** (1 / years) - 1

    else:

        cagr = np.nan

    # --------------------------------------------------------
    # MONTHLY SHARPE
    # --------------------------------------------------------

    monthly_returns = (
        capital_series
        .pct_change()
        .dropna()
    )

    if (
        len(monthly_returns) > 1
        and monthly_returns.std() > 0
    ):

        sharpe = (
            monthly_returns.mean()
            / monthly_returns.std()
            * np.sqrt(12)
        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # BEHAVIOR
    # --------------------------------------------------------

    total_months = len(decisions)

    cash_pct = (

        cash_months / total_months

        if total_months > 0

        else 0
    )

    decisions_df = pd.DataFrame(
        decisions
    )

    distribution = (
        decisions_df["asset"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    dominant_asset = (

        distribution.idxmax()

        if not distribution.empty

        else "N/A"
    )

    return {

        "start": start_date,

        "end": end_date,

        "initial_capital":
            INITIAL_CAPITAL,

        "final_capital":
            capital,

        "total_return":
            (
                capital / INITIAL_CAPITAL
            ) - 1,

        "cagr":
            cagr,

        "max_drawdown":
            max_drawdown,

        "sharpe":
            sharpe,

        "months":
            total_months,

        "rotations":
            rotations,

        "cash_pct":
            cash_pct,

        "dominant_asset":
            dominant_asset
    }


# ============================================================
# SPY BUY & HOLD — SAME MONTHLY FRAMEWORK
# ============================================================

def calculate_spy_period(close, start_date, end_date):

    month_end_dates = get_month_end_dates(close)

    period_dates = [

        d for d in month_end_dates

        if (
            pd.Timestamp(start_date)
            <= d
            <= pd.Timestamp(end_date)
        )

    ]

    if len(period_dates) < 2:

        return None

    spy = close["SPY"]

    start_date_actual = period_dates[0]

    end_date_actual = period_dates[-1]

    period = spy.loc[
        start_date_actual:
        end_date_actual
    ]

    if len(period) < 2:

        return None

    start_price = period.iloc[0]

    end_price = period.iloc[-1]

    final_capital = (

        INITIAL_CAPITAL
        * end_price
        / start_price
    )

    monthly_period = (

        period
        .resample("ME")
        .last()
        .dropna()
    )

    monthly_returns = (
        monthly_period
        .pct_change()
        .dropna()
    )

    cumulative = (
        period / period.iloc[0]
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    years = (
        period.index[-1]
        - period.index[0]
    ).days / 365.25

    if years > 0:

        cagr = (
            final_capital
            / INITIAL_CAPITAL
        ) ** (1 / years) - 1

    else:

        cagr = np.nan

    if (
        len(monthly_returns) > 1
        and monthly_returns.std() > 0
    ):

        sharpe = (
            monthly_returns.mean()
            / monthly_returns.std()
            * np.sqrt(12)
        )

    else:

        sharpe = np.nan

    return {

        "final_capital":
            final_capital,

        "total_return":
            (
                final_capital
                / INITIAL_CAPITAL
            ) - 1,

        "cagr":
            cagr,

        "max_drawdown":
            max_drawdown,

        "sharpe":
            sharpe
    }


# ============================================================
# PERIOD DEFINITIONS
# ============================================================

PERIODS = [

    ("2006-01-01", "2010-12-31"),

    ("2011-01-01", "2015-12-31"),

    ("2016-01-01", "2020-12-31"),

    ("2021-01-01", "2023-12-31"),

    ("2024-01-01", "2026-12-31")
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)

    print(
        "             EME V9 — HISTORICAL STRESS TEST"
    )

    print("=" * 90)

    data = download_data()

    close = get_close_prices(data)

    results = []

    for start_date, end_date in PERIODS:

        print("\n")

        print("-" * 90)

        print(
            f"PERIOD: {start_date} → {end_date}"
        )

        print("-" * 90)

        v9 = run_period(
            close,
            start_date,
            end_date
        )

        spy = calculate_spy_period(
            close,
            start_date,
            end_date
        )

        if v9 is None or spy is None:

            print(
                "Insufficient data."
            )

            continue

        print("\nV9 MONTHLY")

        print(
            f"Final capital : "
            f"€{v9['final_capital']:,.2f}"
        )

        print(
            f"Return        : "
            f"{v9['total_return'] * 100:.2f}%"
        )

        print(
            f"CAGR          : "
            f"{v9['cagr'] * 100:.2f}%"
        )

        print(
            f"Max Drawdown  : "
            f"{v9['max_drawdown'] * 100:.2f}%"
        )

        print(
            f"Sharpe        : "
            f"{v9['sharpe']:.2f}"
        )

        print(
            f"Rotations     : "
            f"{v9['rotations']}"
        )

        print(
            f"Cash          : "
            f"{v9['cash_pct'] * 100:.2f}%"
        )

        print(
            f"Main asset    : "
            f"{v9['dominant_asset']}"
        )

        print("\nSPY BUY & HOLD")

        print(
            f"Final capital : "
            f"€{spy['final_capital']:,.2f}"
        )

        print(
            f"Return        : "
            f"{spy['total_return'] * 100:.2f}%"
        )

        print(
            f"CAGR          : "
            f"{spy['cagr'] * 100:.2f}%"
        )

        print(
            f"Max Drawdown  : "
            f"{spy['max_drawdown'] * 100:.2f}%"
        )

        print(
            f"Sharpe        : "
            f"{spy['sharpe']:.2f}"
        )

        results.append({

            "period":
                f"{start_date} → {end_date}",

            "v9_final_capital":
                v9["final_capital"],

            "v9_total_return":
                v9["total_return"],

            "v9_cagr":
                v9["cagr"],

            "v9_max_drawdown":
                v9["max_drawdown"],

            "v9_sharpe":
                v9["sharpe"],

            "v9_rotations":
                v9["rotations"],

            "v9_cash_pct":
                v9["cash_pct"],

            "v9_main_asset":
                v9["dominant_asset"],

            "spy_final_capital":
                spy["final_capital"],

            "spy_total_return":
                spy["total_return"],

            "spy_cagr":
                spy["cagr"],

            "spy_max_drawdown":
                spy["max_drawdown"],

            "spy_sharpe":
                spy["sharpe"]

        })

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "eme_v9_stress_test.csv",
        index=False
    )

    print("\n")

    print("=" * 90)

    print(
        "                    STRESS TEST SUMMARY"
    )

    print("=" * 90)

    print()

    for _, row in results_df.iterrows():

        print(
            row["period"]
        )

        print(
            f"  V9  CAGR="
            f"{row['v9_cagr'] * 100:.2f}% "
            f"DD="
            f"{row['v9_max_drawdown'] * 100:.2f}% "
            f"Sharpe="
            f"{row['v9_sharpe']:.2f}"
        )

        print(
            f"  SPY CAGR="
            f"{row['spy_cagr'] * 100:.2f}% "
            f"DD="
            f"{row['spy_max_drawdown'] * 100:.2f}% "
            f"Sharpe="
            f"{row['spy_sharpe']:.2f}"
        )

        print(
            f"  V9 Cash="
            f"{row['v9_cash_pct'] * 100:.2f}% "
            f"Rotations="
            f"{int(row['v9_rotations'])} "
            f"Main="
            f"{row['v9_main_asset']}"
        )

        print()

    print(
        "Results saved:"
    )

    print(
        "eme_v9_stress_test.csv"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
