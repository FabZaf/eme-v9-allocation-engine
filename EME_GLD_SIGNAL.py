import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo


# ============================================================
# EME GLD DAILY SIGNAL
# ============================================================

TICKER = "GLD"
TIMEFRAME = "DAILY"

START_DATE = "2006-01-01"

BB_PERIOD = 20
BB_PERCENTILE_PERIOD = 252

ATR_FAST = 5
ATR_SLOW = 20
ATR_RATIO_THRESHOLD = 0.85

ROC_FAST = 3
ROC_SLOW = 10
ROC_MEDIAN_PERIOD = 126

SMA_PERIOD = 20
STD_PERIOD = 20

COMPRESSION_LOOKBACK = 3


# ============================================================
# DOWNLOAD DATI
# ============================================================

def download_data():

    df = yf.download(
        TICKER,
        start=START_DATE,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("Nessun dato ricevuto da Yahoo Finance.")

    # Gestione colonne MultiIndex di yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Colonne mancanti nei dati Yahoo Finance: {missing}"
        )

    df = df[required].copy()

    df.index = pd.to_datetime(df.index)

    return df


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period):

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


# ============================================================
# CALCOLO EME
# ============================================================

def calculate_eme(df):

    # --------------------------------------------------------
    # Bollinger Band Width
    # --------------------------------------------------------

    sma_bb = df["Close"].rolling(BB_PERIOD).mean()
    std_bb = df["Close"].rolling(BB_PERIOD).std()

    upper_bb = sma_bb + 2 * std_bb
    lower_bb = sma_bb - 2 * std_bb

    df["bb_width"] = (
        (upper_bb - lower_bb) / sma_bb
    )

    # Percentile storico della BB Width
    df["bb_width_percentile"] = (
        df["bb_width"]
        .rolling(BB_PERCENTILE_PERIOD)
        .quantile(0.20)
    )

    # --------------------------------------------------------
    # ATR compression
    # --------------------------------------------------------

    df["atr5"] = calculate_atr(df, ATR_FAST)
    df["atr20"] = calculate_atr(df, ATR_SLOW)

    df["atr_ratio"] = (
        df["atr5"] / df["atr20"]
    )

    # --------------------------------------------------------
    # Compression
    # --------------------------------------------------------

    df["compression"] = (
        (df["bb_width"] <= df["bb_width_percentile"])
        &
        (df["atr_ratio"] < ATR_RATIO_THRESHOLD)
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    df["roc3"] = (
        df["Close"].pct_change(ROC_FAST) * 100
    )

    df["roc10"] = (
        df["Close"].pct_change(ROC_SLOW) * 100
    )

    df["delta_roc3"] = (
        df["roc3"] - df["roc3"].shift(1)
    )

    df["roc10_median"] = (
        df["roc10"]
        .rolling(ROC_MEDIAN_PERIOD)
        .median()
    )

    # --------------------------------------------------------
    # SMA + standard deviation
    # --------------------------------------------------------

    df["sma20"] = (
        df["Close"].rolling(SMA_PERIOD).mean()
    )

    df["std20"] = (
        df["Close"].rolling(STD_PERIOD).std()
    )

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    df["acceleration"] = (
        (df["delta_roc3"] > 0)
        &
        (df["roc10"] > df["roc10_median"])
        &
        (
            df["Close"]
            >
            df["sma20"] + df["std20"]
        )
    )

    # --------------------------------------------------------
    # Compression presente nei precedenti 1-3 bar
    # --------------------------------------------------------

    previous_compression = (
        df["compression"]
        .shift(1)
        .rolling(COMPRESSION_LOOKBACK)
        .max()
        .fillna(0)
        .astype(bool)
    )

    # --------------------------------------------------------
    # SEGNALE EME
    # --------------------------------------------------------

    df["signal"] = (
        previous_compression
        &
        df["acceleration"]
    )

    return df


# ============================================================
# INDIVIDUAZIONE ULTIMA BARRA COMPLETATA
# ============================================================

def get_completed_data(df):

    now_ny = datetime.now(
        ZoneInfo("America/New_York")
    )

    today_ny = pd.Timestamp(
        now_ny.date()
    )

    # Se il mercato USA è ancora aperto,
    # non utilizziamo la barra odierna.
    if now_ny.time() < time(16, 0):

        df = df[
            df.index.normalize() < today_ny
        ].copy()

    return df


# ============================================================
# EXIT LOGIC
# ============================================================

def calculate_exit(df, signal_position):

    entry_position = signal_position + 1

    if entry_position >= len(df):
        return None

    # Massimo 3 barre di permanenza
    max_exit_position = min(
        entry_position + 3,
        len(df) - 1
    )

    for i in range(
        entry_position,
        max_exit_position + 1
    ):

        delta_roc3 = df.iloc[i]["delta_roc3"]
        close = df.iloc[i]["Close"]
        sma5 = df["Close"].iloc[
            max(0, i - 4):i + 1
        ].mean()

        exit_condition = (
            (delta_roc3 < 0)
            &
            (close < sma5)
        )

        if exit_condition:
            return df.index[i]

    return df.index[max_exit_position]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("                 EME GLD DAILY")
    print("=" * 70)

    df = download_data()

    df = calculate_eme(df)

    df = get_completed_data(df)

    if len(df) < 300:
        raise RuntimeError(
            "Dati insufficienti per il calcolo EME."
        )

    last_date = df.index[-1]
    last = df.iloc[-1]

    print()
    print(f"Ultima barra completata : {last_date.date()}")
    print(f"Close                    : {last['Close']:.2f}")

    # ========================================================
    # CONTROLLO SEGNALE
    # ========================================================

    if bool(last["signal"]):

        print()
        print("STATUS : BUY")
        print(f"SIGNAL DATE : {last_date.date()}")

        print()
        print(
            "ENTRY DATE  : NEXT TRADING SESSION"
        )

        print(
            "ENTRY PRICE : NEXT OPEN"
        )

        print()
        print(
            "Il prezzo di ingresso non è ancora noto:"
        )

        print(
            "EME entra alla prossima apertura."
        )

        # Salvataggio segnale
        result = pd.DataFrame([{
            "ticker": TICKER,
            "timeframe": TIMEFRAME,
            "status": "BUY",
            "signal_date": last_date.date(),
            "entry_date": "NEXT_TRADING_SESSION",
            "entry_price": "NEXT_OPEN"
        }])

    else:

        print()
        print("STATUS : WAIT")
        print("NO ACTIVE EME SIGNAL")

        result = pd.DataFrame([{
            "ticker": TICKER,
            "timeframe": TIMEFRAME,
            "status": "WAIT",
            "signal_date": last_date.date(),
            "entry_date": "",
            "entry_price": ""
        }])

    result.to_csv(
        "eme_gld_signal.csv",
        index=False
    )

    print()
    print("=" * 70)
    print("Risultato salvato in: eme_gld_signal.csv")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
