import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime


# ============================================================
# EME V9 — ALLOCATION ENGINE
# ============================================================

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

START_DATE = "2006-01-01"

SCORE_THRESHOLD = 65

DEFAULT_CAPITAL = 10000

OUTPUT_FILE = "eme_v9_decisions.csv"


# ============================================================
# V9 ALLOCATION RULE
# ============================================================

def allocation_from_score(score):
    """
    Converts the V8 score into a V9 capital allocation.
    """

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
# BUILD CLOSE DATAFRAME
# ============================================================

def get_close_prices(data):

    if isinstance(data.columns, pd.MultiIndex):

        close = data["Close"].copy()

    else:

        close = data[["Close"]].copy()

        if len(TICKERS) == 1:
            close.columns = TICKERS

    close = close.dropna(how="all")

    return close


# ============================================================
# V8 SCORE ENGINE
# ============================================================

def calculate_scores(close):

    scores = {}

    spy = close["SPY"]

    # SPY trend
    spy_sma200 = spy.rolling(200).mean()

    market_on = spy.iloc[-1] > spy_sma200.iloc[-1]

    for ticker in TICKERS:

        price = close[ticker]

        sma200 = price.rolling(200).mean()

        roc10 = price.pct_change(10)

        roc20 = price.pct_change(20)

        volatility = price.pct_change().rolling(20).std()

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

        asset_return = price.pct_change(20).iloc[-1]
        spy_return = spy.pct_change(20).iloc[-1]

        if asset_return > spy_return:
            relative_score += 20

        if price.pct_change(60).iloc[-1] > spy.pct_change(60).iloc[-1]:
            relative_score += 20

        # -----------------------------
        # RISK / VOLATILITY
        # -----------------------------

        risk_score = 0

        if volatility.iloc[-1] < volatility.rolling(100).median().iloc[-1]:
            risk_score += 10

        if volatility.iloc[-1] < volatility.rolling(200).median().iloc[-1]:
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

    if data["score"] < SCORE_THRESHOLD:
        return None, data["score"]

    return ticker, data["score"]


# ============================================================
# CREATE DECISION
# ============================================================

def create_decision(scores, market_on, capital):

    selected_asset, score = select_asset(
        scores,
        market_on
    )

    allocation = allocation_from_score(score)

    invested_amount = capital * allocation

    cash_amount = capital - invested_amount

    decision = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "market_on": market_on,
        "asset": selected_asset if selected_asset else "CASH",
        "score": round(score, 2),
        "allocation_pct": round(allocation * 100, 2),
        "invested_amount": round(invested_amount, 2),
        "cash_amount": round(cash_amount, 2)
    }

    return decision


# ============================================================
# SAVE DECISION
# ============================================================

def save_decision(decision):

    file = Path(OUTPUT_FILE)

    df_new = pd.DataFrame([decision])

    if file.exists():

        df_old = pd.read_csv(file)

        df = pd.concat(
            [df_old, df_new],
            ignore_index=True
        )

    else:

        df = df_new

    df.to_csv(file, index=False)


# ============================================================
# REPORT
# ============================================================

def print_report(scores, market_on, decision):

    print("\n")
    print("=" * 65)
    print("                 EME V9 ALLOCATION ENGINE")
    print("=" * 65)

    print(f"\nMarket regime : {'ON' if market_on else 'OFF'}")

    print("\nScores:")

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    for ticker, data in ranked:

        print(
            f"{ticker:5s} "
            f"Score={data['score']:6.2f} "
            f"Trend={data['trend']:2d} "
            f"Relative={data['relative']:2d} "
            f"Risk={data['risk']:2d}"
        )

    print("\n" + "-" * 65)

    print("V9 DECISION")

    print(f"Asset             : {decision['asset']}")
    print(f"Score             : {decision['score']}")
    print(f"Allocation        : {decision['allocation_pct']}%")
    print(f"Capital invested  : €{decision['invested_amount']:.2f}")
    print(f"Cash              : €{decision['cash_amount']:.2f}")

    print("-" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    capital = DEFAULT_CAPITAL

    data = download_data()

    close = get_close_prices(data)

    scores, market_on = calculate_scores(close)

    decision = create_decision(
        scores,
        market_on,
        capital
    )

    print_report(
        scores,
        market_on,
        decision
    )

    save_decision(decision)

    print("\nDecision saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
