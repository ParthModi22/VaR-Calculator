"""
Phase 2: Historical Simulation VaR.

Sort observed portfolio returns from worst to best; the VaR at confidence level c
is the (1-c) percentile of those returns — no distributional assumption required.
"""

import pandas as pd
import numpy as np

INPUT_FILE = "returns.csv"
CONFIDENCE_LEVELS = [0.95, 0.99]


def historical_var(returns: pd.Series, confidence: float) -> float:
    """Return the VaR as a positive loss value (e.g. 0.023 = 2.3% daily loss)."""
    return -np.percentile(returns, (1 - confidence) * 100)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    port = df["Portfolio"]

    print("=== Historical Simulation VaR ===")
    print(f"{'Confidence':<15} {'VaR (daily loss)':<20}")
    print("-" * 35)
    for cl in CONFIDENCE_LEVELS:
        var = historical_var(port, cl)
        print(f"{cl * 100:.0f}%{'':<12} {var:.4f}  ({var * 100:.2f}%)")
