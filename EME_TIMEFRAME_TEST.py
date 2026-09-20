import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# EME — ASSET × TIMEFRAME TEST
# ============================================================

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

START_DATE = "2006-01-01"

TIMEFRAMES = {
    "DAILY": "1d",
    "WEEKLY": "1wk",
    "MONTHLY": "1mo"
}

INITIAL_CAPITAL = 10000


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
# DOWNLOAD
# ============================================================

def download_data(interval):

    print(
        f"\nDownloading data: {interval}"
    )

    data = yf.download(
        TICKERS,
        start=START_DATE,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:

        raise RuntimeError(
            f"No data downloaded for {interval}"
        )

    return data


# ============================================================
# CLOSE / OHLC
# ============================================================

def get_ohlc(data):

    if not isinstance(
        data.columns,
        pd.MultiIndex
    ):

        raise RuntimeError(
            "Unexpected Yahoo Finance column structure."
        )

    result = {}

    for field in ["Open", "High", "Low", "Close"]:

        result[field] = data[field].copy()

    return result


# ============================================================
# ATR
# ============================================================

def calculate_atr(high, low, close, period):

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return (
        true_range
        .rolling(period)
        .mean()
    )


# ============================================================
# EME SIGNAL
# ============================================================

def calculate_signal(open_price, high, low, close):

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    sma20 = (
        close
        .rolling(BB_PERIOD)
        .mean()
    )

    std20 = (
        close
        .rolling(BB_PERIOD)
        .std()
    )

    upper_band = (
        sma20
        + BB_STD * std20
    )

    lower_band = (
        sma20
        - BB_STD * std20
    )

    bb_width = (
        upper_band - lower_band
    ) / sma20

    bb_threshold = (
        bb_width
        .rolling(BB_PERCENTILE_WINDOW)
        .quantile(BB_PERCENTILE)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr5 = calculate_atr(
        high,
        low,
        close,
        ATR_FAST
    )

    atr20 = calculate_atr(
        high,
        low,
        close,
        ATR_SLOW
    )

    atr_ratio = atr5 / atr20

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    roc3 = close.pct_change(
        ROC_FAST
    )

    roc10 = close.pct_change(
        ROC_SLOW
    )

    delta_roc3 = (
        roc3 - roc3.shift(1)
    )

    roc10_median = (
        roc10
        .rolling(ROC_MEDIAN_WINDOW)
        .median()
    )

    # --------------------------------------------------------
    # PHASE 1 — COMPRESSION
    # --------------------------------------------------------

    compression_now = (

        (bb_width <= bb_threshold)

        & (atr_ratio < ATR_RATIO_THRESHOLD)
    )

    compression_recent = (

        compression_now
        .shift(1)
        .rolling(3)
        .max()
        .fillna(0)
        .astype(bool)
    )

    # --------------------------------------------------------
    # PHASE 2 — ACCELERATION
    # --------------------------------------------------------

    acceleration = (

        (delta_roc3 > 0)

        & (roc10 > roc10_median)

        & (
            close
            > sma20 + std20
        )
    )

    # --------------------------------------------------------
    # FINAL EME SIGNAL
    # --------------------------------------------------------

    signal = (
        compression_recent
        & acceleration
    )

    # --------------------------------------------------------
    # EXIT CONDITIONS
    # --------------------------------------------------------

    exit_signal = (

        (delta_roc3 < 0)

        & (close < close.rolling(SMA_EXIT).mean())
    )

    return signal, exit_signal


# ============================================================
# RUN ONE ASSET
# ============================================================

def run_asset(
    open_price,
    high,
    low,
    close,
    ticker
):

    signal, exit_signal = calculate_signal(
        open_price,
        high,
        low,
        close
    )

    trades = []

    i = 0

    while i < len(close) - 1:

        if not bool(signal.iloc[i]):

            i += 1

            continue

        # ----------------------------------------------------
        # ENTRY = NEXT BAR OPEN
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(close):

            break

        entry_price = (
            open_price.iloc[entry_index]
        )

        entry_date = (
            open_price.index[entry_index]
        )

        exit_index = None

        exit_reason = None

        # ----------------------------------------------------
        # SEARCH EXIT
        # ----------------------------------------------------

        for hold in range(
            1,
            MAX_HOLD_BARS + 1
        ):

            candidate = (
                entry_index + hold
            )

            if candidate >= len(close):

                break

            if bool(
                exit_signal.iloc[candidate - 1]
            ):

                exit_index = candidate

                exit_reason = (
                    "SIGNAL_EXIT"
                )

                break

        # ----------------------------------------------------
        # MAX HOLD
        # ----------------------------------------------------

        if exit_index is None:

            exit_index = min(
                entry_index + MAX_HOLD_BARS,
                len(close) - 1
            )

            exit_reason = (
                "MAX_HOLD"
            )

        exit_price = (
            open_price.iloc[exit_index]
        )

        exit_date = (
            open_price.index[exit_index]
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        trade_return = (
            exit_price / entry_price
        ) - 1

        trades.append({

            "ticker": ticker,

            "signal_date":
                close.index[i],

            "entry_date":
                entry_date,

            "exit_date":
                exit_date,

            "entry_price":
                entry_price,

            "exit_price":
                exit_price,

            "return":
                trade_return,

            "hold_bars":
                exit_index - entry_index,

            "exit_reason":
                exit_reason
        })

        # ----------------------------------------------------
        # NO OVERLAPPING TRADES
        # ----------------------------------------------------

        i = exit_index + 1

    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if trades.empty:

        return {

            "signals": 0,

            "win_rate": np.nan,

            "avg_return": np.nan,

            "median_return": np.nan,

            "profit_factor": np.nan,

            "total_return": 0.0,

            "max_drawdown": np.nan,

            "final_capital":
                INITIAL_CAPITAL
        }

    returns = trades["return"]

    wins = returns[returns > 0]

    losses = returns[returns < 0]

    win_rate = (
        (returns > 0).mean()
    )

    avg_return = returns.mean()

    median_return = returns.median()

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    # --------------------------------------------------------
    # EQUITY CURVE
    # --------------------------------------------------------

    equity = (
        INITIAL_CAPITAL
        * (1 + returns).cumprod()
    )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1

    max_drawdown = drawdown.min()

    final_capital = equity.iloc[-1]

    total_return = (
        final_capital
        / INITIAL_CAPITAL
    ) - 1

    return {

        "signals":
            len(trades),

        "win_rate":
            win_rate,

        "avg_return":
            avg_return,

        "median_return":
            median_return,

        "profit_factor":
            profit_factor,

        "total_return":
            total_return,

        "max_drawdown":
            max_drawdown,

        "final_capital":
            final_capital
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)

    print(
        "             EME — ASSET × TIMEFRAME TEST"
    )

    print("=" * 90)

    all_results = []

    all_trades = []

    for timeframe_name, interval in TIMEFRAMES.items():

        print("\n")
        print("=" * 90)

        print(
            f"TIMEFRAME: {timeframe_name}"
        )

        print("=" * 90)

        data = download_data(
            interval
        )

        ohlc = get_ohlc(data)

        for ticker in TICKERS:

            print(
                f"\nTesting {ticker}..."
            )

            trades = run_asset(

                ohlc["Open"][ticker],

                ohlc["High"][ticker],

                ohlc["Low"][ticker],

                ohlc["Close"][ticker],

                ticker
            )

            metrics = calculate_metrics(
                trades
            )

            result = {

                "timeframe":
                    timeframe_name,

                "ticker":
                    ticker,

                **metrics
            }

            all_results.append(
                result
            )

            if not trades.empty:

                trades = trades.copy()

                trades["timeframe"] = (
                    timeframe_name
                )

                all_trades.append(
                    trades
                )

            print(

                f"Signals={metrics['signals']} | "

                f"WinRate="
                f"{metrics['win_rate'] * 100:.2f}% "
                if not np.isnan(
                    metrics["win_rate"]
                )
                else "Signals=0 | WinRate=N/A "
            )

            print(

                f"Avg={metrics['avg_return'] * 100:.3f}% | "

                f"PF={metrics['profit_factor']:.2f} | "

                f"Total="
                f"{metrics['total_return'] * 100:.2f}% | "

                f"DD="
                f"{metrics['max_drawdown'] * 100:.2f}%"

                if not np.isnan(
                    metrics["avg_return"]
                )

                else "No trades"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        "eme_timeframe_results.csv",
        index=False
    )

    if all_trades:

        trades_df = pd.concat(
            all_trades,
            ignore_index=True
        )

    else:

        trades_df = pd.DataFrame()

    trades_df.to_csv(
        "eme_timeframe_trades.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 90)

    print(
        "                    FINAL MATRIX"
    )

    print("=" * 90)

    for timeframe in TIMEFRAMES:

        print(
            f"\n--- {timeframe} ---"
        )

        subset = results_df[
            results_df["timeframe"]
            == timeframe
        ]

        for _, row in subset.iterrows():

            if row["signals"] == 0:

                print(
                    f"{row['ticker']:5s} "
                    f"NO SIGNALS"
                )

                continue

            print(

                f"{row['ticker']:5s} "

                f"Signals={int(row['signals']):4d} "

                f"Win={row['win_rate'] * 100:6.2f}% "

                f"Avg={row['avg_return'] * 100:7.3f}% "

                f"PF={row['profit_factor']:5.2f} "

                f"Total={row['total_return'] * 100:8.2f}% "

                f"DD={row['max_drawdown'] * 100:7.2f}%"
            )

    print("\n")
    print(
        "Results saved:"
    )

    print(
        "eme_timeframe_results.csv"
    )

    print(
        "eme_timeframe_trades.csv"
    )


if __name__ == "__main__":

    main()
