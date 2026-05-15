#!/usr/bin/env python3
"""
Analyze evaluate_model_on_generated_data.py output CSVs.

Default usage:
    python data_testing/analyze_generated_eval_results.py

Useful options:
    python data_testing/analyze_generated_eval_results.py --stem generated_eval_4pt
    python data_testing/analyze_generated_eval_results.py --plots-dir data_testing/outputs/my_analysis
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = ROOT / "data_testing" / "outputs"
DEFAULT_STEM = "generated_eval_4pt"
MODE_ORDER = ["greedy", "beam", "nucleus"]

PASS_COLS = [
    "top1_exact_token_match",
    "top1_exact_string_match",
    "top1_num_eq_simple",
    "top1_num_eq_scrambled",
    "any_beam_exact_token_match",
    "any_beam_exact_string_match",
    "any_beam_num_eq_simple",
    "any_beam_num_eq_scrambled",
]

PLOT_STYLE = {
    "pass": "#1b9e77",
    "fail": "#d95f02",
    "decode": "#7570b3",
    "beam": "#4c78a8",
    "greedy": "#f58518",
}


def split_top_level_sum(expr: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in "+-" and i > 0:
            prev = expr[i - 1]
            if prev not in "*/^(+":
                parts.append("".join(current).strip())
                current = [ch]
                continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def term_count(expr: str) -> int:
    return len(split_top_level_sum(str(expr)))


def load_results(outputs_dir: Path, stem: str, modes: Iterable[str] | None) -> pd.DataFrame:
    selected_modes = list(modes) if modes else MODE_ORDER
    frames: list[pd.DataFrame] = []
    for mode in selected_modes:
        path = outputs_dir / f"{stem}_{mode}_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No result CSVs found for stem '{stem}' under {outputs_dir}"
        )
    df = pd.concat(frames, ignore_index=True)
    for col in PASS_COLS + ["top1_decode_ok"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    numeric_cols = [
        "row_id",
        "target_simple_token_count",
        "target_scrambled_token_count",
        "top1_prediction_token_count",
        "candidate_sequences_checked",
        "candidate_valid_decode_count",
        "candidate_exact_token_count",
        "candidate_exact_string_count",
        "candidate_num_eq_simple_count",
        "candidate_num_eq_scrambled_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["simple_term_count"] = df["target_simple"].map(term_count)
    df["prediction_token_delta"] = (
        df["top1_prediction_token_count"] - df["target_simple_token_count"]
    )
    df["prediction_length_ratio"] = (
        df["top1_prediction_token_count"] / df["target_simple_token_count"]
    )
    df["beam_found_non_top1_match"] = (
        (df["any_beam_exact_token_match"] == 1)
        & (df["top1_exact_token_match"] == 0)
    ).astype(int)
    df["diagnosis"] = df.apply(diagnose_row, axis=1)
    return df


def diagnose_row(row: pd.Series) -> str:
    if int(row["top1_exact_token_match"]) == 1:
        return "exact"
    if int(row["top1_decode_ok"]) == 0:
        return "malformed_top1"
    if int(row["any_beam_exact_token_match"]) == 1:
        return "beam_rescued"
    if int(row["top1_num_eq_simple"]) == 1:
        return "numeric_only"
    delta = float(row["prediction_token_delta"])
    if delta <= -10:
        return "too_short"
    if delta >= 10:
        return "too_long"
    return "valid_but_wrong"


def write_text_report(df: pd.DataFrame, summary_path: Path, report_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append("")
    for mode, group in df.groupby("mode", sort=False):
        total = len(group)
        lines.append(f"Mode: {mode}")
        lines.append(f"  total rows: {total}")
        for col in PASS_COLS:
            if col in group.columns:
                lines.append(f"  {col}: {int(group[col].sum())}/{total}")
        lines.append(
            "  target simple tokens: "
            f"min={group['target_simple_token_count'].min():.0f}, "
            f"median={group['target_simple_token_count'].median():.0f}, "
            f"max={group['target_simple_token_count'].max():.0f}"
        )
        lines.append(
            "  prediction token delta: "
            f"median={group['prediction_token_delta'].median():.0f}, "
            f"mean={group['prediction_token_delta'].mean():.1f}"
        )
        diag_counts = group["diagnosis"].value_counts().to_dict()
        lines.append(f"  diagnoses: {diag_counts}")
        lines.append("")
    if summary_path.exists():
        lines.append(f"Evaluator summary CSV: {summary_path}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_counts(df: pd.DataFrame, out_path: Path) -> None:
    metric_labels = [
        ("top1_exact_token_match", "top1 token"),
        ("top1_num_eq_simple", "top1 numeric"),
        ("any_beam_exact_token_match", "any candidate token"),
        ("any_beam_num_eq_simple", "any candidate numeric"),
    ]
    modes = list(df["mode"].drop_duplicates())
    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        group = df[df["mode"] == mode]
        vals = [group[col].sum() for col, _ in metric_labels]
        ax.bar(x + i * width, vals, width=width, label=mode)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels([label for _, label in metric_labels], rotation=20, ha="right")
    ax.set_ylabel("passing rows")
    ax.set_title("Pass counts by metric")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_length_scatter(df: pd.DataFrame, out_path: Path) -> None:
    modes = list(df["mode"].drop_duplicates())
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), squeeze=False)
    max_len = max(
        df["target_simple_token_count"].max(),
        df["top1_prediction_token_count"].max(),
    )
    for ax, mode in zip(axes[0], modes):
        group = df[df["mode"] == mode]
        passed = group["top1_exact_token_match"] == 1
        ax.scatter(
            group.loc[~passed, "target_simple_token_count"],
            group.loc[~passed, "top1_prediction_token_count"],
            color=PLOT_STYLE["fail"],
            label="fail",
            alpha=0.8,
        )
        ax.scatter(
            group.loc[passed, "target_simple_token_count"],
            group.loc[passed, "top1_prediction_token_count"],
            color=PLOT_STYLE["pass"],
            label="exact",
            alpha=0.9,
        )
        ax.plot([0, max_len], [0, max_len], color="#555555", linewidth=1, alpha=0.7)
        ax.set_title(f"{mode}: target vs prediction length")
        ax.set_xlabel("target simple token count")
        ax.set_ylabel("top1 prediction token count")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_length_bins(df: pd.DataFrame, out_path: Path) -> None:
    bins = sorted(set(df["target_simple_token_count"].quantile([0, 0.25, 0.5, 0.75, 1]).round().astype(int)))
    if len(bins) < 3:
        min_len = int(df["target_simple_token_count"].min())
        max_len = int(df["target_simple_token_count"].max())
        bins = list(np.linspace(min_len, max_len + 1, 5).astype(int))
    df = df.copy()
    df["length_bin"] = pd.cut(
        df["target_simple_token_count"],
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    )
    pivot = df.pivot_table(
        index="length_bin",
        columns="mode",
        values="top1_exact_token_match",
        aggfunc="mean",
        observed=False,
    ).fillna(0.0)
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=25)
    ax.set_ylabel("top1 exact match rate")
    ax.set_xlabel("target simple token length")
    ax.set_title("Accuracy by target length bin")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=160)
    plt.close(ax.figure)


def plot_term_counts(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot_table(
        index="simple_term_count",
        columns="mode",
        values="top1_exact_token_match",
        aggfunc=["mean", "count"],
        observed=False,
    ).fillna(0.0)
    rate = pivot["mean"]
    count = pivot["count"].max(axis=1)
    ax = rate.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylabel("top1 exact match rate")
    ax.set_xlabel("number of simple terms")
    ax.set_title("Accuracy by generated term count")
    ax.grid(axis="y", alpha=0.25)
    for i, n in enumerate(count):
        ax.text(i, 1.02, f"n={int(n)}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(1.15, float(rate.max().max()) + 0.15))
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=160)
    plt.close(ax.figure)


def plot_row_outcomes(df: pd.DataFrame, out_path: Path) -> None:
    modes = list(df["mode"].drop_duplicates())
    row_ids = sorted(df["row_id"].unique())
    matrix = np.full((len(modes), len(row_ids)), np.nan)
    for i, mode in enumerate(modes):
        group = df[df["mode"] == mode].set_index("row_id")
        for j, row_id in enumerate(row_ids):
            if row_id not in group.index:
                continue
            row = group.loc[row_id]
            if int(row["top1_exact_token_match"]) == 1:
                matrix[i, j] = 2
            elif int(row["top1_decode_ok"]) == 1:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    cmap = matplotlib.colors.ListedColormap(["#7570b3", "#d95f02", "#1b9e77"])
    fig, ax = plt.subplots(figsize=(12, 2.6 + 0.5 * len(modes)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xticks(np.arange(len(row_ids)))
    ax.set_xticklabels(row_ids, rotation=90)
    ax.set_xlabel("row id")
    ax.set_title("Row outcomes: malformed, valid wrong, exact")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["malformed", "valid wrong", "exact"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_beam_candidates(df: pd.DataFrame, out_path: Path) -> None:
    beam = df[df["candidate_sequences_checked"] > 1].copy()
    if beam.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = np.where(beam["top1_exact_token_match"] == 1, PLOT_STYLE["pass"], PLOT_STYLE["fail"])
    ax.bar(
        beam["row_id"].astype(str),
        beam["candidate_valid_decode_count"],
        color=colors,
    )
    ax.axhline(
        beam["candidate_sequences_checked"].median(),
        color="#555555",
        linewidth=1,
        linestyle="--",
        label="candidate count",
    )
    ax.set_title("Beam candidate validity by row")
    ax.set_xlabel("row id")
    ax.set_ylabel("valid decoded candidates")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_failure_categories(df: pd.DataFrame, out_path: Path) -> None:
    counts = df.groupby(["mode", "diagnosis"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(10, 5), rot=0)
    ax.set_ylabel("rows")
    ax.set_title("Outcome diagnosis counts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="diagnosis", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=160)
    plt.close(ax.figure)


def print_console_summary(df: pd.DataFrame, plots_dir: Path, diagnostics_path: Path) -> None:
    print(f"Analyzed {len(df)} mode/row records")
    for mode, group in df.groupby("mode", sort=False):
        total = len(group)
        exact = int(group["top1_exact_token_match"].sum())
        malformed = int((group["top1_decode_ok"] == 0).sum())
        beam_rescued = int(group["beam_found_non_top1_match"].sum())
        print(
            f"{mode}: exact={exact}/{total}, malformed_top1={malformed}, "
            f"beam_rescued={beam_rescued}, mean_len_delta={group['prediction_token_delta'].mean():.1f}"
        )
    print(f"Wrote diagnostics CSV to {diagnostics_path}")
    print(f"Wrote plots to {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--mode", action="append", help="Mode to include; may be repeated.")
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    plots_dir = (
        args.plots_dir.resolve()
        if args.plots_dir is not None
        else outputs_dir / f"{args.stem}_analysis"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(outputs_dir, args.stem, args.mode)
    diagnostics_path = plots_dir / f"{args.stem}_row_diagnostics.csv"
    report_path = plots_dir / f"{args.stem}_analysis_report.txt"
    summary_path = outputs_dir / f"{args.stem}_summary.csv"

    df.to_csv(diagnostics_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_text_report(df, summary_path, report_path)
    plot_metric_counts(df, plots_dir / "metric_counts.png")
    plot_length_scatter(df, plots_dir / "length_scatter.png")
    plot_length_bins(df, plots_dir / "length_bins_success.png")
    plot_term_counts(df, plots_dir / "term_count_success.png")
    plot_row_outcomes(df, plots_dir / "row_outcomes.png")
    plot_beam_candidates(df, plots_dir / "beam_candidate_validity.png")
    plot_failure_categories(df, plots_dir / "failure_categories.png")
    print_console_summary(df, plots_dir, diagnostics_path)


if __name__ == "__main__":
    main()
