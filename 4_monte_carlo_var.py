"""
Phase 4: Monte Carlo VaR.

Simulates N random 1-day return paths drawn from N(μ, σ²) fit to historical data.
The VaR is the (1-c) percentile of that simulated distribution. Extends naturally to
non-linear portfolios (e.g. options) where Historical and Parametric methods break down.
"""

import pandas as pd
import numpy as np

INPUT_FILE = "returns.csv"
CONFIDENCE_LEVELS = [0.95, 0.99]
N_SIMULATIONS = 100_000
RANDOM_SEED = 42


def monte_carlo_var(returns: pd.Series, confidence: float, n_sims: int, seed: int) -> float:
    """Simulate n_sims daily returns and return the VaR at the given confidence level."""
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    simulated = rng.normal(loc=mu, scale=sigma, size=n_sims)
    return -np.percentile(simulated, (1 - confidence) * 100)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    port = df["Portfolio"]

    print(f"Running {N_SIMULATIONS:,} simulations (seed={RANDOM_SEED})")
    print()
    print("=== Monte Carlo VaR ===")
    print(f"{'Confidence':<15} {'VaR (daily loss)':<20}")
    print("-" * 35)
    for cl in CONFIDENCE_LEVELS:
        var = monte_carlo_var(port, cl, N_SIMULATIONS, RANDOM_SEED)
        print(f"{cl * 100:.0f}%{'':<12} {var:.4f}  ({var * 100:.2f}%)")
