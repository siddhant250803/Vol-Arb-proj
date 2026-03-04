#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_OUTPUT_DIR, FIGURES_DIR


def load_pnl(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "cumulative_pnl" not in df.columns:
        raise ValueError(f"{path} is missing cumulative_pnl")
    return df[["date", "cumulative_pnl"]].rename(columns={"cumulative_pnl": label})


def main() -> None:
    q5_path = DATA_OUTPUT_DIR / "logistic_q5_daily_pnl.csv"
    q10_path = DATA_OUTPUT_DIR / "logistic_q10_daily_pnl.csv"

    q5 = load_pnl(q5_path, "q=5")
    q10 = load_pnl(q10_path, "q=10")

    merged = q5.merge(q10, on="date", how="outer").sort_values("date")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "logistic_cumulative_pnl_comparison.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)
    ax.plot(merged["date"], merged["q=5"], label="Logistic q=5", linewidth=2.2, color="#1f6feb")
    ax.plot(merged["date"], merged["q=10"], label="Logistic q=10", linewidth=2.2, color="#d97706")

    ax.set_title("Cumulative PnL: Logistic Signal Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(frameon=True)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
