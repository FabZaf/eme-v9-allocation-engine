import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


# ============================================================
# EME ETF MONITOR
# ============================================================
# Scopo:
#
# 1. Analizzare la robustezza storica dell'EME V7.
# 2. Monitorare SPY / QQQ / IWM / DIA / GLD.
# 3. Individuare un eventuale deterioramento dell'ETF attuale.
# 4. Segnalare quando un altro ETF supera l'attuale.
#
# TIMEFRAME OPERATIVO:
# DAILY
#
# La logica EME V7 NON viene modificata.
# ============================================================


TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "GLD"
]

TIMEFRAME = "1d"

START_DATE = "2006-01-01"

CURRENT_ETF_FILE = "eme_current_etf.txt"

RESULTS_FILE = "eme_etf_monitor.csv"

ALERT_FILE = "eme_etf_alert.txt"


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

def download_data(ticker):

    data = yf.download(
        ticker,
        start=START_DATE,
        interval=TIMEFRAME,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise RuntimeError(
            f"No data for {ticker}"
        )

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data = data.xs(
            ticker,
            axis=1,
            level=-1
        )

    data = data[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ].dropna()

    return data


# ============================================================
# TRUE RANGE
# ============================================================

def true_range(df):

    previous_close = df["Close"].shift(1)

    tr1 = (
        df["High"] -
        df["Low"]
    )

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    return pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)


# ============================================================
# EME INDICATORS
# ============================================================

def calculate_indicators(df):

    x = df.copy()

    close = x["Close"]

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    middle = (
        close
        .rolling(BB_PERIOD)
        .mean()
    )

    std = (
        close
        .rolling(BB_PERIOD)
        .std()
    )

    upper = (
        middle +
        BB_STD * std
    )

    lower = (
        middle -
        BB_STD * std
    )

    bb_width = (
        (upper - lower) /
        middle
    )

    bb_threshold = (
        bb_width
        .rolling(BB_PERCENTILE_WINDOW)
        .quantile(BB_PERCENTILE)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    tr = true_range(x)

    atr_fast = (
        tr
        .rolling(ATR_FAST)
        .mean()
    )

    atr_slow = (
        tr
        .rolling(ATR_SLOW)
        .mean()
    )

    atr_ratio = (
        atr_fast /
        atr_slow
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    roc3 = (
        close
        .pct_change(ROC_FAST)
    )

    roc10 = (
        close
        .pct_change(ROC_SLOW)
    )

    delta_roc3 = roc3.diff()

    roc10_median = (
        roc10
        .rolling(ROC_MEDIAN_WINDOW)
        .median()
    )

    # --------------------------------------------------------
    # SMA
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
        (delta_roc3 > 0)
        &
        (roc10 > roc10_median)
        &
        (close > (sma20 + std20))
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    x["signal"] = (
        compression_recent &
        acceleration
    )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    x["exit_signal"] = (
        (delta_roc3 < 0)
        &
        (close < sma5)
    )

    return x


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):

    trades = []

    i = 0

    while i < len(df) - 1:

        if not bool(
            df["signal"].iloc[i]
        ):

            i += 1
            continue

        entry_index = i + 1

        if entry_index >= len(df):
            break

        entry_price = float(
            df["Open"].iloc[entry_index]
        )

        exit_index = None

        last_index = min(
            entry_index +
            MAX_HOLD_BARS,
            len(df) - 1
        )

        for candidate in range(
            entry_index + 1,
            last_index + 1
        ):

            signal_bar = candidate - 1

            if bool(
                df["exit_signal"].iloc[
                    signal_bar
                ]
            ):

                exit_index = candidate
                break

        if exit_index is None:

            exit_index = last_index

        if exit_index <= entry_index:

            i = entry_index
            continue

        exit_price = float(
            df["Open"].iloc[exit_index]
        )

        r = (
            exit_price /
            entry_price
        ) - 1.0

        trades.append({

            "signal_date":
                df.index[i],

            "entry_date":
                df.index[entry_index],

            "exit_date":
                df.index[exit_index],

            "return":
                r

        })

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
            "profit_factor": np.nan,
            "total_return": np.nan
        }

    r = trades["return"].values

    wins = r[r > 0]
    losses = r[r < 0]

    if len(losses) > 0:

        pf = (
            wins.sum() /
            abs(losses.sum())
        )

    else:

        pf = np.inf

    equity = 10000.0

    for value in r:

        equity *= (
            1.0 + value
        )

    return {

        "signals": len(r),

        "win_rate":
            (r > 0).mean() * 100,

        "avg_return":
            r.mean() * 100,

        "profit_factor":
            pf,

        "total_return":
            (equity / 10000.0 - 1)
            * 100
    }


# ============================================================
# LOAD CURRENT ETF
# ============================================================

def load_current_etf():

    file = Path(
        CURRENT_ETF_FILE
    )

    if file.exists():

        value = (
            file.read_text()
            .strip()
            .upper()
        )

        if value in TICKERS:
            return value

    # Primo stato:
    # utilizziamo GLD perché è l'ETF
    # che nel test precedente ha prodotto
    # il maggior numero di trade tra
    # i risultati più interessanti.
    return "GLD"


# ============================================================
# ROBUSTNESS
# ============================================================

def calculate_period_metrics(
    trades
):

    periods = {

        "2006-2010":
            ("2006-01-01", "2010-12-31"),

        "2011-2015":
            ("2011-01-01", "2015-12-31"),

        "2016-2020":
            ("2016-01-01", "2020-12-31"),

        "2021-2026":
            ("2021-01-01", "2026-12-31")
    }

    result = {}

    for name, dates in periods.items():

        start, end = dates

        subset = trades[
            (trades["entry_date"] >= start)
            &
            (trades["entry_date"] <= end)
        ]

        metrics = calculate_metrics(
            subset
        )

        result[name] = metrics

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    current_etf = load_current_etf()

    print("\n")
    print("=" * 80)
    print("             EME ETF MONITOR")
    print("=" * 80)

    print(
        f"\nETF ATTUALE: {current_etf}"
    )

    all_metrics = []

    trade_data = {}

    for ticker in TICKERS:

        print(
            f"\nAnalysing {ticker}..."
        )

        try:

            data = download_data(
                ticker
            )

            df = calculate_indicators(
                data
            )

            trades = run_backtest(
                df
            )

            metrics = calculate_metrics(
                trades
            )

            trade_data[ticker] = trades

            all_metrics.append({

                "ticker": ticker,

                **metrics
            })

            print(
                f"Signals: "
                f"{metrics['signals']}"
            )

            print(
                f"Win rate: "
                f"{metrics['win_rate']:.2f}%"
            )

            print(
                f"Avg return: "
                f"{metrics['avg_return']:.4f}%"
            )

            print(
                f"Profit factor: "
                f"{metrics['profit_factor']:.3f}"
            )

            print(
                f"Total return: "
                f"{metrics['total_return']:.2f}%"
            )

        except Exception as e:

            print(
                f"{ticker}: ERROR {e}"
            )

    results = pd.DataFrame(
        all_metrics
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ========================================================
    # CURRENT ETF
    # ========================================================

    current_row = results[
        results["ticker"] ==
        current_etf
    ]

    # ========================================================
    # CANDIDATE SELECTION
    # ========================================================
    #
    # NON scegliamo automaticamente.
    #
    # Un candidato deve:
    #
    # - avere almeno 30 trade
    # - avere profit factor > 1
    # - avere rendimento medio > 0
    #
    # Questi criteri sono usati come
    # filtro di monitoraggio, non come
    # previsione.
    # ========================================================

    eligible = results[
        (results["signals"] >= 30)
        &
        (results["profit_factor"] > 1.0)
        &
        (results["avg_return"] > 0)
    ].copy()

    # ========================================================
    # CURRENT ETF STATUS
    # ========================================================

    current_valid = False

    if not current_row.empty:

        row = current_row.iloc[0]

        current_valid = (
            row["signals"] >= 30
            and row["profit_factor"] > 1.0
            and row["avg_return"] > 0
        )

    # ========================================================
    # CANDIDATE
    # ========================================================

    if not eligible.empty:

        eligible = eligible.sort_values(
            by=[
                "profit_factor",
                "avg_return",
                "signals"
            ],
            ascending=False
        )

        candidate = (
            eligible.iloc[0]["ticker"]
        )

    else:

        candidate = None

    # ========================================================
    # ALERT LOGIC
    # ========================================================

    alert = False

    reason = "NO_CHANGE"

    if candidate is not None:

        if candidate != current_etf:

            if not current_valid:

                alert = True

                reason = (
                    "CURRENT_ETF_NO_LONGER_MEETS_FILTER"
                )

            else:

                current_pf = float(
                    current_row.iloc[0][
                        "profit_factor"
                    ]
                )

                candidate_pf = float(
                    eligible.iloc[0][
                        "profit_factor"
                    ]
                )

                if candidate_pf > current_pf:

                    alert = True

                    reason = (
                        "ALTERNATIVE_ETF_EXCEEDS_CURRENT"
                    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n")
    print("=" * 80)
    print("                    EME DECISION")
    print("=" * 80)

    print(
        f"Current ETF : {current_etf}"
    )

    print(
        f"Candidate   : "
        f"{candidate if candidate else 'NONE'}"
    )

    if alert:

        print(
            "\n🚨 ALERT: ETF CHANGE"
        )

        print(
            f"Reason: {reason}"
        )

        alert_text = (
            "EME ETF CHANGE ALERT\n"
            f"Current ETF: {current_etf}\n"
            f"Candidate ETF: {candidate}\n"
            f"Reason: {reason}\n"
        )

        Path(
            ALERT_FILE
        ).write_text(
            alert_text
        )

    else:

        print(
            "\nSTATUS: NO CHANGE"
        )

        Path(
            ALERT_FILE
        ).write_text(
            "NO_CHANGE\n"
            f"Current ETF: {current_etf}\n"
        )

    # ========================================================
    # PERIOD ROBUSTNESS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("              PERIOD ROBUSTNESS")
    print("=" * 80)

    for ticker in TICKERS:

        trades = trade_data.get(
            ticker,
            pd.DataFrame()
        )

        periods = calculate_period_metrics(
            trades
        )

        print(
            f"\n{ticker}"
        )

        for period, metrics in periods.items():

            print(
                f"{period}: "
                f"{metrics['signals']} trades | "
                f"PF={metrics['profit_factor']:.3f} | "
                f"Avg={metrics['avg_return']:.3f}%"
            )

    # ========================================================
    # SAVE CURRENT STATE
    # ========================================================

    if alert:

        Path(
            CURRENT_ETF_FILE
        ).write_text(
            candidate
        )

    print("\n")
    print("=" * 80)
    print(
        f"Results saved: {RESULTS_FILE}"
    )
    print(
        f"Alert saved:   {ALERT_FILE}"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()
