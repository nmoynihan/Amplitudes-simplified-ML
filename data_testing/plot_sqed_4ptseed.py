#!/usr/bin/env python3
"""Plot diagnostics for the SQED 4pt seed evaluation runs."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = ROOT / "data_testing" / "outputs"
RESULT_RE = re.compile(r"sqed_4ptseed_oneshot_(\d+)_beam_results\.csv$")


def load_results(outputs_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(outputs_dir.glob("sqed_4ptseed_oneshot_*_beam_results.csv")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        df = pd.read_csv(path)
        df["dataset"] = int(match.group(1))
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No sqed_4ptseed_oneshot_*_beam_results.csv files found in {outputs_dir}"
        )

    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "dataset",
        "target_scrambled_token_count",
        "top1_prediction_token_count",
        "top1_num_eq_scrambled",
        "original_top1_num_eq_scrambled",
        "any_beam_num_eq_scrambled",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target_scrambled_token_count", "top1_num_eq_scrambled"])
    df["correct"] = df["top1_num_eq_scrambled"].astype(bool)
    return df


def save_input_length_vs_correctness(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(7)

    for dataset, part in df.groupby("dataset", sort=True):
        y = part["top1_num_eq_scrambled"].to_numpy(dtype=float)
        jitter = rng.normal(0, 0.035, size=len(part))
        ax.scatter(
            part["target_scrambled_token_count"],
            y + jitter,
            s=28,
            alpha=0.65,
            label=f"dataset {dataset}",
        )

    ax.set_title("Input Length vs Correctness")
    ax.set_xlabel("Scrambled input token count")
    ax.set_ylabel("Top-1 numerically correct")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["incorrect", "correct"])
    ax.set_ylim(-0.18, 1.18)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_prediction_length_vs_input_length(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    incorrect = df[~df["correct"]]
    correct = df[df["correct"]]
    ax.scatter(
        incorrect["target_scrambled_token_count"],
        incorrect["top1_prediction_token_count"],
        s=30,
        alpha=0.65,
        color="#d95f02",
        label="incorrect",
    )
    ax.scatter(
        correct["target_scrambled_token_count"],
        correct["top1_prediction_token_count"],
        s=30,
        alpha=0.65,
        color="#1b9e77",
        label="correct",
    )

    max_len = max(
        df["target_scrambled_token_count"].max(),
        df["top1_prediction_token_count"].max(),
    )
    ax.plot([0, max_len], [0, max_len], color="#666666", lw=1, ls="--", label="y = x")
    ax.set_title("Prediction Length vs Input Length")
    ax.set_xlabel("Scrambled input token count")
    ax.set_ylabel("Top-1 prediction token count")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_accuracy_vs_scrambled_token_count(df: pd.DataFrame, path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    lengths = df["target_scrambled_token_count"]
    edges = np.unique(np.quantile(lengths, np.linspace(0, 1, bins + 1)).astype(int))
    if len(edges) < 3:
        edges = np.linspace(lengths.min(), lengths.max() + 1, min(bins, len(df)) + 1)

    for label, part, color, lw in [
        ("all datasets", df, "#222222", 2.5),
        *[
            (f"dataset {dataset}", part, None, 1.6)
            for dataset, part in df.groupby("dataset", sort=True)
        ],
    ]:
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi == edges[-1]:
                mask = (part["target_scrambled_token_count"] >= lo) & (
                    part["target_scrambled_token_count"] <= hi
                )
            else:
                mask = (part["target_scrambled_token_count"] >= lo) & (
                    part["target_scrambled_token_count"] < hi
                )
            bucket = part[mask]
            if len(bucket) < 3:
                continue
            rows.append(
                {
                    "mid": (lo + hi) / 2,
                    "accuracy": 100 * bucket["top1_num_eq_scrambled"].mean(),
                }
            )
        if not rows:
            continue
        binned = pd.DataFrame(rows)
        ax.plot(
            binned["mid"],
            binned["accuracy"],
            marker="o",
            lw=lw,
            color=color,
            label=label,
        )

    ax.set_title("Accuracy vs Scrambled Token Count")
    ax.set_xlabel("Scrambled input token count, binned")
    ax.set_ylabel("Top-1 numerical accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_by_scramble_count(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scrambles, part in df.groupby("dataset", sort=True):
        correct = part[part["correct"]]
        rows.append(
            {
                "num_scrambles": int(scrambles),
                "examples": int(len(part)),
                "correct_examples": int(part["top1_num_eq_scrambled"].sum()),
                "correct_pct": 100 * part["top1_num_eq_scrambled"].mean(),
                "avg_correct_pred_tokens": correct["top1_prediction_token_count"].mean(),
                "avg_correct_scrambled_tokens": correct["target_scrambled_token_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_scramble_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax_pct = plt.subplots(figsize=(9, 5.5))
    ax_tok = ax_pct.twinx()

    x = summary["num_scrambles"]
    ax_pct.plot(
        x,
        summary["correct_pct"],
        marker="o",
        lw=2.5,
        color="#1b9e77",
        label="correct",
    )
    ax_tok.plot(
        x,
        summary["avg_correct_pred_tokens"],
        marker="s",
        lw=2,
        color="#7570b3",
        label="avg predicted tokens when correct",
    )
    ax_tok.plot(
        x,
        summary["avg_correct_scrambled_tokens"],
        marker="^",
        lw=2,
        color="#d95f02",
        label="avg scrambled tokens when correct",
    )

    for _, row in summary.iterrows():
        ax_pct.annotate(
            f"{row['correct_examples']}/{row['examples']}",
            (row["num_scrambles"], row["correct_pct"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1b9e77",
        )

    ax_pct.set_title("Correctness and Token Length by Number of Scrambles")
    ax_pct.set_xlabel("Number of scrambles")
    ax_pct.set_ylabel("Top-1 numerical correctness (%)", color="#1b9e77")
    ax_tok.set_ylabel("Average token count, correct predictions only")
    ax_pct.set_xticks(x)
    ax_pct.set_ylim(0, 105)
    ax_pct.grid(axis="y", alpha=0.2)

    lines_pct, labels_pct = ax_pct.get_legend_handles_labels()
    lines_tok, labels_tok = ax_tok.get_legend_handles_labels()
    ax_pct.legend(
        lines_pct + lines_tok,
        labels_pct + labels_tok,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    plots_dir = (
        args.plots_dir.resolve()
        if args.plots_dir is not None
        else outputs_dir / "sqed_4ptseed_plots"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(outputs_dir)
    summary = summarize_by_scramble_count(df)
    summary.to_csv(plots_dir / "scramble_summary.csv", index=False)
    save_scramble_summary(summary, plots_dir / "scramble_summary.png")
    save_input_length_vs_correctness(df, plots_dir / "input_length_vs_correctness.png")
    save_prediction_length_vs_input_length(df, plots_dir / "prediction_length_vs_input_length.png")
    save_accuracy_vs_scrambled_token_count(df, plots_dir / "accuracy_vs_scrambled_token_count.png", args.bins)

    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
