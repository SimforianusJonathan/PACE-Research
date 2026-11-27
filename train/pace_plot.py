"""Plot training and OOD curves from completed MiniGrid run exports."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def _training_curve(path, bin_width=5_000_000):
    bins = defaultdict(lambda: [0, 0, 0.0, 0])
    for row in _rows(path):
        steps = int(row["env_steps"])
        episodes = float(row["ppo_episodes"])
        if episodes:
            point = bins[(steps - 1) // bin_width]
            point[0] += episodes
            point[1] += float(row["ppo_solved"])
            point[2] += float(row["ppo_return_sum"])
            point[3] = steps
    points = [
        ((index * bin_width + steps) / 2e6, solved / episodes, reward / episodes)
        for index, (episodes, solved, reward, steps) in sorted(bins.items())
    ]
    return list(zip(*points)) if points else ([], [], [])


def plot_curves(run_dirs, output_dir):
    ood_panels = (
        ("solve_rate_mean", "OOD solve rate"), ("mean_return", "OOD mean return"),
        ("iqm", "OOD interquartile mean"), ("optimality_gap", "Optimality gap"),
    )
    names = ("PPO solve rate", "PPO mean return")
    output_dir.mkdir(parents=True, exist_ok=True)
    train, train_axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), layout="constrained")
    ood, ood_axes = plt.subplots(2, 2, sharex=True, figsize=(11, 7), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, run in enumerate(run_dirs):
        summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
        label = f"{summary['experiment']} (seed {summary['config']['seed']})"
        style = {"linewidth": 1.6, "linestyle": "-", "alpha": 1.0, "color": colors[index % len(colors)]}
        x, solve_rate, mean_return = _training_curve(run / "training_curve.csv")
        for ax, values in zip(train_axes, (solve_rate, mean_return)):
            ax.plot(x, values, label=label, **style)
        points = sorted(_rows(run / "ood_metrics_by_eval.csv"), key=lambda row: int(row["env_steps"]))
        ood_x = [int(row["env_steps"]) / 1e6 for row in points]
        for ax, (field, _) in zip(ood_axes.flat, ood_panels):
            values = [float(row[field]) for row in points]
            ax.plot(ood_x, values, label=label, **style)
            ax.scatter(ood_x[-1], values[-1], color=style["color"], s=16, zorder=3)
    for ax, title in [*zip(train_axes, names), *zip(ood_axes.flat, (p[1] for p in ood_panels))]:
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", alpha=0.25)
        if "return" not in title:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    train_axes[0].legend(frameon=False)
    ood_axes[0, 0].legend(frameon=False)
    for ax in (train_axes[-1], *ood_axes[-1]):
        ax.set_xlabel("Training interactions (millions)")
    train.savefig(output_dir / "training_learning_curve.png", dpi=160)
    ood.savefig(output_dir / "ood_learning_curves.png", dpi=160)
    plt.close(train)
    plt.close(ood)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plot_curves(args.run_dirs, args.output_dir)


if __name__ == "__main__":
    main()
