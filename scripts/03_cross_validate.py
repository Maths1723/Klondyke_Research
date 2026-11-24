"""Step 3: spatial cross-validation of every method on a processed creek.

Usage:  python scripts/03_cross_validate.py [--name dominion] [--seeds 3]
Outputs: reports/cv_<name>.csv (summary), reports/cv_<name>_raw.csv,
         reports/figures/cv_<name>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from klondyke import experiment  # noqa: E402


def plot(summ: pd.DataFrame, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    schemes = summ.index.get_level_values(0).unique()
    fig, axs = plt.subplots(1, len(schemes), figsize=(5.5 * len(schemes), 4), sharex=True)
    for ax, sch in zip(axs if len(schemes) > 1 else [axs], schemes):
        d = summ.loc[sch].sort_values("crps", ascending=False)
        ax.barh(d.index, d.crps, color=["#b5651d" if "gp" in i or "kriging_sn" in i else "#999" for i in d.index])
        ax.set_title(f"{sch}: CRPS (lower is better)")
        for i, (c, r) in enumerate(zip(d.crps, d.spearman)):
            ax.text(c, i, f"  ρ={r:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / f"cv_{name}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dominion")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "data" / "processed" / f"{a.name}_holes.csv")
    print(f"{len(df)} holes, {df.line_id.nunique()} lines, {df.s.max():.0f} m of valley")
    res = pd.concat([experiment.run_all(df, seed=s)[0] for s in range(a.seeds)])
    res.to_csv(ROOT / "reports" / f"cv_{a.name}_raw.csv", index=False)
    summ = experiment.summarise(res)
    summ.to_csv(ROOT / "reports" / f"cv_{a.name}.csv")
    plot(summ, a.name)
    print("\n", summ.to_string())


if __name__ == "__main__":
    main()
