import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# EME V9 — STRESS TEST BY PERIOD
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
# EXACTLY THE SAME LOGIC AS V9 MONTHLY
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

        roc10
