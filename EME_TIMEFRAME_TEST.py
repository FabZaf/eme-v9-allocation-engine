import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# EME — TIMEFRAME TEST
# ============================================================
# Obiettivo:
# confrontare la stessa logica EME V7 su:
#
#   ASSET:      SPY QQQ IWM DIA GLD
#   TIMEFRAME:  DAILY / WEEKLY / MONTHLY
#
# Nessuna ottimizzazione.
# Nessuna allocazione.
# Nessun look-ahead.
#
# Entry: Open della barra successiva al segnale.
# Exit:
#   - Open della barra successiva quando si verifica
#     il segnale di uscita
#   - oppure dopo MAX_HOLD_BARS
# ============================================================


TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

START_DATE = "2006-01-01"

TIMEFRAMES = {
    "DAILY": "1d",
    "WEEKLY": "1wk",
    "MONTHLY": "1mo"
}


# ============================================================
# EME V7 PARAMETERS
# ============================================================

BB_PERIOD = 20
BB_STD = 2

BB_PERCENTILE_WINDOW = 252
BB_PERCENTILE = 0.20

ATR_FAST = 5
ATR_SLOW = 20
ATR_RATIO_THRESHOLD = 0.85

ROC_FAST = 3
ROC_SLOW = 10
ROC_MEDIAN_WINDOW = 126

SMA_EXIT = 5

MAX_HOLD_BARS = 3


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_FILE = "eme_timeframe_results.csv"
TRADES_FILE = "eme_timeframe_trades.csv"


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data(ticker, interval):

    print(
        f"Downloading {ticker} "
        f"{interval} data..."
    )

    data = yf.download(
        ticker,
        start=START_DATE,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return None

    # Gestione eventuale MultiIndex restituito da yfinance
    if isinstance(data.columns, pd.MultiIndex):

        if ticker in data.columns.get_level_values(-1):

            data = data.xs(
                ticker,
                axis=1,
                level=-1
            )

        else:

            data.columns = data.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in required:

        if column not in data.columns:
            raise RuntimeError(
                f"{ticker}: colonna {column} non disponibile."
            )

    data = data[required].copy()

    data = data.dropna()

    return data


# ============================================================
# TRUE RANGE
# ============================================================

def calculate_true_range(data):

    previous_close = data["Close"].shift(1)

    tr1 = data["High"] - data["Low"]

    tr2 = (
        data["High"] - previous_close
    ).abs()

    tr3 = (
        data["Low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range


# ============================================================
# BUILD EME INDICATORS
# ============================================================

def calculate_indicators(data):

    df = data.copy()

    close = df["Close"]

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    bb_middle = (
        close
        .rolling(BB_PERIOD)
        .mean()
    )

    bb_std = (
        close
        .rolling(BB_PERIOD)
        .std()
    )

    bb_upper = (
        bb_middle
        + BB_STD * bb_std
    )

    bb_lower = (
        bb_middle
        - BB_STD * bb_std
    )

    bb_width = (
        (bb_upper - bb_lower)
        / bb_middle
    )

    # --------------------------------------------------------
    # BB WIDTH PERCENTILE
    # --------------------------------------------------------

    bb_threshold = (
        bb_width
        .rolling(BB_PERCENTILE_WINDOW)
        .quantile(BB_PERCENTILE)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    true_range = calculate_true_range(df)

    atr_fast = (
        true_range
        .rolling(ATR_FAST)
        .mean()
    )

    atr_slow = (
        true_range
        .rolling(ATR_SLOW)
        .mean()
    )

    atr_ratio = (
        atr_fast / atr_slow
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    roc_fast = (
        close
        .pct_change(ROC_FAST)
    )

    roc_slow = (
        close
        .pct_change(ROC_SLOW)
    )

    delta_roc_fast = (
        roc_fast.diff()
    )

    roc_slow_median = (
        roc_slow
        .rolling(ROC_MEDIAN_WINDOW)
        .median()
    )

    # --------------------------------------------------------
    # SMA + STANDARD DEVIATION
    # --------------------------------------------------------

    sma20 = (
        close
        .rolling(20)
        .mean()
    )

    std20 = (
        close
        .rolling(20)
        .std()
    )

    sma5 = (
        close
        .rolling(SMA_EXIT)
        .mean()
    )

    # --------------------------------------------------------
    # COMPRESSION
    #
    # Compression must exist in one of the
    # previous 3 bars.
    # --------------------------------------------------------

    compression_now = (
        (bb_width <= bb_threshold)
        &
        (atr_ratio < ATR_RATIO_THRESHOLD)
    )

    compression_recent = (
        compression_now
        .shift(1)
        .rolling(3)
        .max()
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------

    acceleration = (
        (delta_roc_fast > 0)
        &
        (roc_slow > roc_slow_median)
        &
        (close > (sma20 + std20))
    )

    # --------------------------------------------------------
    # ENTRY SIGNAL
    # --------------------------------------------------------

    signal = (
        compression_recent
        &
        acceleration
    )

    # --------------------------------------------------------
    # EXIT SIGNAL
    # --------------------------------------------------------

    exit_signal = (
        (delta_roc_fast < 0)
        &
        (close < sma5)
    )

    df["signal"] = signal
    df["exit_signal"] = exit_signal

    return df


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df, ticker, timeframe):

    trades = []

    i = 0

    while i < len(df) - 1:

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        if not bool(df["signal"].iloc[i]):

            i += 1
            continue

        # ----------------------------------------------------
        # ENTRY = NEXT BAR OPEN
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(df):

            break

        entry_date = df.index[entry_index]

        entry_price = float(
            df["Open"].iloc[entry_index]
        )

        # ----------------------------------------------------
        # SEARCH EXIT
        # ----------------------------------------------------

        exit_index = None
        exit_reason = None

        last_possible_index = min(
            entry_index + MAX_HOLD_BARS,
            len(df) - 1
        )

        for candidate in range(
            entry_index + 1,
            last_possible_index + 1
        ):

            # Exit condition observed on previous bar,
            # executed at current bar OPEN.
            signal_bar = candidate - 1

            if bool(
                df["exit_signal"].iloc[signal_bar]
            ):

                exit_index = candidate
                exit_reason = "SIGNAL_EXIT"
                break

        # ----------------------------------------------------
        # MAX HOLD
        # ----------------------------------------------------

        if exit_index is None:

            exit_index = last_possible_index
            exit_reason = "MAX_HOLD"

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        if exit_index <= entry_index:

            i = entry_index
            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        exit_date = df.index[exit_index]

        exit_price = float(
            df["Open"].iloc[exit_index]
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        trade_return = (
            exit_price / entry_price
        ) - 1.0

        trades.append({

            "ticker": ticker,
            "timeframe": timeframe,

            "signal_date": df.index[i].strftime(
                "%Y-%m-%d"
            ),

            "entry_date": entry_date.strftime(
                "%Y-%m-%d"
            ),

            "exit_date": exit_date.strftime(
                "%Y-%m-%d"
            ),

            "entry_price": entry_price,
            "exit_price": exit_price,

            "return_pct": trade_return * 100,

            "win": (
                1
                if trade_return > 0
                else 0
            ),

            "exit_reason": exit_reason,

            "bars_held": (
                exit_index - entry_index
            )
        })

        # ----------------------------------------------------
        # NO OVERLAPPING TRADES
        # ----------------------------------------------------

        i = exit_index + 1

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if len(trades) == 0:

        return {

            "signals": 0,
            "win_rate_pct": np.nan,
            "avg_return_pct": np.nan,
            "median_return_pct": np.nan,
            "profit_factor": np.nan,
            "total_return_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "final_capital": np.nan
        }

    returns = np.array(
        [
            t["return_pct"] / 100
            for t in trades
        ]
    )

    # --------------------------------------------------------
    # WIN RATE
    # --------------------------------------------------------

    win_rate = (
        (returns > 0).mean()
        * 100
    )

    # --------------------------------------------------------
    # AVERAGE / MEDIAN
    # --------------------------------------------------------

    avg_return = (
        returns.mean()
        * 100
    )

    median_return = (
        np.median(returns)
        * 100
    )

    # --------------------------------------------------------
    # PROFIT FACTOR
    # --------------------------------------------------------

    gross_profit = returns[
        returns > 0
    ].sum()

    gross_loss = abs(
        returns[
            returns < 0
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    # --------------------------------------------------------
    # COMPOUNDED EQUITY
    # --------------------------------------------------------

    equity = [10000.0]

    capital = 10000.0

    for r in returns:

        capital *= (
            1.0 + r
        )

        equity.append(capital)

    equity = np.array(equity)

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / running_max
    ) - 1.0

    max_drawdown = (
        drawdown.min()
        * 100
    )

    total_return = (
        (capital / 10000.0) - 1.0
    ) * 100

    return {

        "signals": len(trades),

        "win_rate_pct": win_rate,

        "avg_return_pct": avg_return,

        "median_return_pct": median_return,

        "profit_factor": profit_factor,

        "total_return_pct": total_return,

        "max_drawdown_pct": max_drawdown,

        "final_capital": capital
    }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("             EME — ASSET × TIMEFRAME TEST")
    print("=" * 80)

    all_results = []
    all_trades = []

    for timeframe_name, interval in TIMEFRAMES.items():

        print("\n")
        print("-" * 80)
        print(
            f"TIMEFRAME: {timeframe_name}"
        )
        print("-" * 80)

        for ticker in TICKERS:

            try:

                data = download_data(
                    ticker,
                    interval
                )

                if data is None:

                    print(
                        f"{ticker}: NO DATA"
                    )

                    continue

                print(
                    f"{ticker}: "
                    f"{len(data)} bars"
                )

                # ------------------------------------------------
                # HISTORY CHECK
                #
                # Monthly with 252-bar percentile requires
                # more historical observations than available
                # in the requested period.
                # Do NOT alter the V7 parameter.
                # ------------------------------------------------

                minimum_required = (
                    BB_PERCENTILE_WINDOW
                    + ROC_MEDIAN_WINDOW
                    + 20
                )

                if len(data) <= minimum_required:

                    print(
                        f"{ticker}: "
                        f"INSUFFICIENT HISTORY "
                        f"for unchanged EME V7 parameters"
                    )

                    all_results.append({

                        "timeframe": timeframe_name,
                        "ticker": ticker,
                        "status": "INSUFFICIENT_HISTORY",
                        "signals": 0,
                        "win_rate_pct": np.nan,
                        "avg_return_pct": np.nan,
                        "median_return_pct": np.nan,
                        "profit_factor": np.nan,
                        "total_return_pct": np.nan,
                        "max_drawdown_pct": np.nan,
                        "final_capital": np.nan
                    })

                    continue

                # ------------------------------------------------
                # INDICATORS
                # ------------------------------------------------

                df = calculate_indicators(
                    data
                )

                # ------------------------------------------------
                # BACKTEST
                # ------------------------------------------------

                trades = run_backtest(
                    df,
                    ticker,
                    timeframe_name
                )

                metrics = calculate_metrics(
                    trades
                )

                # ------------------------------------------------
                # SAVE TRADES
                # ------------------------------------------------

                all_trades.extend(
                    trades
                )

                # ------------------------------------------------
                # SAVE RESULTS
                # ------------------------------------------------

                result = {

                    "timeframe": timeframe_name,
                    "ticker": ticker,
                    "status": (
                        "OK"
                        if metrics["signals"] > 0
                        else "NO_SIGNALS"
                    ),

                    **metrics
                }

                all_results.append(
                    result
                )

                # ------------------------------------------------
                # PRINT RESULT
                # ------------------------------------------------

                print(
                    f"  Signals       : "
                    f"{metrics['signals']}"
                )

                if metrics["signals"] > 0:

                    print(
                        f"  Win rate      : "
                        f"{metrics['win_rate_pct']:.2f}%"
                    )

                    print(
                        f"  Avg return    : "
                        f"{metrics['avg_return_pct']:.4f}%"
                    )

                    print(
                        f"  Median return : "
                        f"{metrics['median_return_pct']:.4f}%"
                    )

                    print(
                        f"  Profit factor : "
                        f"{metrics['profit_factor']:.3f}"
                    )

                    print(
                        f"  Total return  : "
                        f"{metrics['total_return_pct']:.2f}%"
                    )

                    print(
                        f"  Max drawdown  : "
                        f"{metrics['max_drawdown_pct']:.2f}%"
                    )

                    print(
                        f"  Final capital : "
                        f"€{metrics['final_capital']:.2f}"
                    )

                else:

                    print(
                        "  NO SIGNALS"
                    )

            except Exception as e:

                print(
                    f"{ticker}: ERROR — {e}"
                )

                all_results.append({

                    "timeframe": timeframe_name,
                    "ticker": ticker,
                    "status": "ERROR",
                    "signals": 0,
                    "win_rate_pct": np.nan,
                    "avg_return_pct": np.nan,
                    "median_return_pct": np.nan,
                    "profit_factor": np.nan,
                    "total_return_pct": np.nan,
                    "max_drawdown_pct": np.nan,
                    "final_capital": np.nan
                })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    trades_df = pd.DataFrame(
        all_trades
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    trades_df.to_csv(
        TRADES_FILE,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("                    FINAL COMPARISON")
    print("=" * 100)

    if not results_df.empty:

        display_columns = [

            "timeframe",
            "ticker",
            "status",
            "signals",
            "win_rate_pct",
            "avg_return_pct",
            "median_return_pct",
            "profit_factor",
            "total_return_pct",
            "max_drawdown_pct"
        ]

        print(
            results_df[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}"
            )
        )

    print("\n")
    print("=" * 100)
    print("Files created:")
    print(f"  {RESULTS_FILE}")
    print(f"  {TRADES_FILE}")
    print("=" * 100)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
