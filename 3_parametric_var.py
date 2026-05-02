import pandas as pd
import numpy as np
from scipy import stats

INPUT_FILE = "returns.csv"
CONFIDENCE_LEVELS = [0.95, 0.99]


def parametric_var(returns: pd.Series, confidence: float) -> float:
    """Return the VaR as a positive loss value under the normality assumption."""
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    # ppf gives the left-tail z-score; multiply by sigma and shift by mu
    return -(mu + stats.norm.ppf(1 - confidence) * sigma)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    port = df["Portfolio"]

    mu = port.mean()
    sigma = port.std(ddof=1)
    print(f"Portfolio daily μ = {mu:.5f},  σ = {sigma:.5f}")
    print()
    print("=== Parametric VaR (Normal Distribution) ===")
    print(f"{'Confidence':<15} {'VaR (daily loss)':<20}")
    print("-" * 35)
    for cl in CONFIDENCE_LEVELS:
        var = parametric_var(port, cl)
        print(f"{cl * 100:.0f}%{'':<12} {var:.4f}  ({var * 100:.2f}%)")
