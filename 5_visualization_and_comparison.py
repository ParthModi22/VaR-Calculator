import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

INPUT_FILE = "returns.csv"
CONFIDENCE_LEVELS = [0.95, 0.99]
N_SIMULATIONS = 100_000
RANDOM_SEED = 42


def historical_var(returns: pd.Series, confidence: float) -> float:
    return -np.percentile(returns, (1 - confidence) * 100)


def parametric_var(returns: pd.Series, confidence: float) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return -(mu + stats.norm.ppf(1 - confidence) * sigma)


def monte_carlo_var(returns: pd.Series, confidence: float, n_sims: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    simulated = rng.normal(loc=mu, scale=sigma, size=n_sims)
    return -np.percentile(simulated, (1 - confidence) * 100)



def build_table(port: pd.Series) -> pd.DataFrame:
    rows = []
    for cl in CONFIDENCE_LEVELS:
        rows.append({
            "Confidence": f"{cl * 100:.0f}%",
            "Historical":  f"{historical_var(port, cl) * 100:.2f}%",
            "Parametric":  f"{parametric_var(port, cl) * 100:.2f}%",
            "Monte Carlo": f"{monte_carlo_var(port, cl, N_SIMULATIONS, RANDOM_SEED) * 100:.2f}%",
        })
    return pd.DataFrame(rows).set_index("Confidence")


def plot(port: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(port, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3,
            density=True, alpha=0.85, label="Daily returns")

    colors = {"Historical": "#DD4444", "Parametric": "#22AA55", "Monte Carlo": "#FF8800"}
    styles = {0.95: "-", 0.99: "--"}

    for cl in CONFIDENCE_LEVELS:
        for method, fn in [
            ("Historical",  lambda r, c: historical_var(r, c)),
            ("Parametric",  lambda r, c: parametric_var(r, c)),
            ("Monte Carlo", lambda r, c: monte_carlo_var(r, c, N_SIMULATIONS, RANDOM_SEED)),
        ]:
            var = fn(port, cl)
            ax.axvline(
                x=-var,
                color=colors[method],
                linestyle=styles[cl],
                linewidth=1.8,
                label=f"{method} {cl * 100:.0f}% VaR: {var * 100:.2f}%",
            )

    ax.set_xlabel("Daily Portfolio Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Portfolio Return Distribution & VaR Thresholds\n"
                 "HDFCBANK · RELIANCE · SBIN  (equally weighted, 5-year daily returns)",
                 fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("var_comparison.png", dpi=150)
    print("Chart saved to var_comparison.png")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    port = df["Portfolio"]

    print("=== VaR Comparison Table (daily loss at portfolio level) ===\n")
    table = build_table(port)
    print(table.to_string())
    print()

    plot(port)
