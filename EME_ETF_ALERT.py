import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# EME ETF ALERT
# ============================================================

INPUT_FILE = "eme_timeframe_trades.csv"
OUTPUT_FILE = "eme_etf_alert.csv"

ETF_LIST = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

CURRENT_ETF = "GLD"
TIMEFRAME = "DAILY"

# Usiamo la stessa numerosità del periodo 2021-2026
# del test di robustezza appena eseguito.
RECENT_TRADES = 15

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 20260920


# ============================================================
# BOOTSTRAP DIFFERENCE
# ============================================================

def bootstrap_difference(a, b):

    rng = np.random.default_rng(RANDOM_SEED)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan

    observed = np.mean(b) - np.mean(a)

    differences = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):

        sample_a = rng.choice(
            a,
            size=len(a),
            replace=True
        )

        sample_b = rng.choice(
            b,
            size=len(b),
            replace=True
        )

        differences[i] = (
            np.mean(sample_b) -
            np.mean(sample_a)
        )

    lower = np.percentile(differences, 2.5)
    upper = np.percentile(differences, 97.5)

    return observed, lower, upper


# ============================================================
# PROFIT FACTOR
# ============================================================

def profit_factor(returns):

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:

        if gains > 0:
            return np.inf

        return np.nan

    return gains / losses


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not Path(INPUT_FILE).exists():

        raise FileNotFoundError(
            f"File non trovato: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "ticker",
        "timeframe",
        "signal_date",
        "return_pct"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    df["signal_date"] = pd.to_datetime(
        df["signal_date"],
        errors="coerce"
    )

    df["return_pct"] = pd.to_numeric(
        df["return_pct"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "ticker",
            "timeframe",
            "signal_date",
            "return_pct"
        ]
    )

    df = df[
        df["timeframe"] == TIMEFRAME
    ].copy()

    return df


# ============================================================
# RECENT PERFORMANCE
# ============================================================

def recent_performance(df, ticker):

    x = df[
        df["ticker"] == ticker
    ].copy()

    x = x.sort_values(
        "signal_date"
    )

    x = x.tail(
        RECENT_TRADES
    )

    returns = x["return_pct"].to_numpy(
        dtype=float
    )

    if len(returns) == 0:

        return {
            "ticker": ticker,
            "trades": 0,
            "avg_return": np.nan,
            "median_return": np.nan,
            "profit_factor": np.nan,
            "compound_return": np.nan
        }

    compound = (
        np.prod(1 + returns / 100) - 1
    ) * 100

    return {
        "ticker": ticker,
        "trades": len(returns),
        "avg_return": np.mean(returns),
        "median_return": np.median(returns),
        "profit_factor": profit_factor(returns),
        "compound_return": compound
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    performance = {}

    for ticker in ETF_LIST:

        performance[ticker] = recent_performance(
            df,
            ticker
        )

    current = performance[CURRENT_ETF]

    current_data = df[
        df["ticker"] == CURRENT_ETF
    ].sort_values(
        "signal_date"
    ).tail(
        RECENT_TRADES
    )

    current_returns = current_data[
        "return_pct"
    ].to_numpy(
        dtype=float
    )

    candidates = []

    for ticker in ETF_LIST:

        if ticker == CURRENT_ETF:
            continue

        candidate_data = df[
            df["ticker"] == ticker
        ].sort_values(
            "signal_date"
        ).tail(
            RECENT_TRADES
        )

        candidate_returns = candidate_data[
            "return_pct"
        ].to_numpy(
            dtype=float
        )

        if len(current_returns) < RECENT_TRADES:
            continue

        if len(candidate_returns) < RECENT_TRADES:
            continue

        difference, lower, upper = (
            bootstrap_difference(
                current_returns,
                candidate_returns
            )
        )

        candidate_pf = (
            performance[ticker]["profit_factor"]
        )

        current_pf = current[
            "profit_factor"
        ]

        # Il candidato deve:
        # 1. avere rendimento medio superiore
        # 2. avere differenza statisticamente positiva
        # 3. avere PF superiore a quello dell'ETF attuale

        statistically_better = (
            lower > 0
        )

        better_pf = (
            candidate_pf > current_pf
        )

        better_mean = (
            performance[ticker]["avg_return"]
            > current["avg_return"]
        )

        if (
            statistically_better
            and better_pf
            and better_mean
        ):

            candidates.append({
                "ticker": ticker,
                "difference": difference,
                "lower_ci": lower,
                "upper_ci": upper,
                "profit_factor": candidate_pf,
                "avg_return": performance[ticker][
                    "avg_return"
                ]
            })

    # ========================================================
    # DECISION
    # ========================================================

    if len(candidates) == 0:

        status = "HOLD"
        candidate = ""

        reason = (
            "No sufficient statistical evidence "
            "to change the current ETF."
        )

    else:

        candidates = sorted(
            candidates,
            key=lambda x: x["difference"],
            reverse=True
        )

        best = candidates[0]

        status = "CHANGE ALERT"
        candidate = best["ticker"]

        reason = (
            "Candidate shows statistically stronger "
            "recent EME performance than the current ETF."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    result = {
        "date": pd.Timestamp.now().strftime(
            "%Y-%m-%d"
        ),
        "current_etf": CURRENT_ETF,
        "timeframe": TIMEFRAME,
        "current_trades": current["trades"],
        "current_avg_return_pct": current[
            "avg_return"
        ],
        "current_profit_factor": current[
            "profit_factor"
        ],
        "status": status,
        "candidate_etf": candidate,
        "reason": reason
    }

    result_df = pd.DataFrame([
        result
    ])

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("                 EME ETF ALERT")
    print("=" * 70)

    print(
        f"\nCurrent ETF : {CURRENT_ETF}"
    )

    print(
        f"Timeframe   : {TIMEFRAME}"
    )

    print(
        f"Recent trades analyzed : {RECENT_TRADES}"
    )

    print(
        f"\nCurrent average return : "
        f"{current['avg_return']:.4f}%"
    )

    print(
        f"Current profit factor  : "
        f"{current['profit_factor']:.4f}"
    )

    print(
        f"\nSTATUS : {status}"
    )

    if candidate:

        print(
            f"CANDIDATE ETF : {candidate}"
        )

    print(
        f"REASON : {reason}"
    )

    print("\n" + "=" * 70)

    print(
        f"\nRisultato salvato in: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
