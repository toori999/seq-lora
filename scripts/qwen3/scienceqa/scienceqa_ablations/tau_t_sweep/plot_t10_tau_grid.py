#!/usr/bin/env python3
"""Plot T=10 fixed-tau sensitivity as a 3x3 task/metric grid."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


TASKS = [
    ("scienceqa_closedchoice_grade2_11_iid", "SciQA IID"),
    ("mmlu_science_high_ood", "MMLU-H"),
    ("mmlu_science_college_ood", "MMLU-C"),
]

METRICS = [
    ("acc", "ACC (%)", "acc_bayes"),
    ("nll", "NLL", "nll_bayes"),
    ("ece", "ECE (%)", "ece_bayes"),
]


def parse_tau_from_name(path: Path) -> float | None:
    name = path.stem
    if name == "tau05":
        return 0.5
    if name == "tau1":
        return 1.0
    match = re.fullmatch(r"tau0p([0-9]+)", name)
    if match:
        return float(f"0.{match.group(1)}")
    return None


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    current_task: str | None = None
    out: dict[str, dict[str, float]] = {}
    task_names = {task for task, _ in TASKS}
    task_re = re.compile(r"^\[([^\]]+)\]$")
    metric_re = re.compile(r"^\s+(nll_bayes|ece_bayes|acc_bayes):\s+([0-9.]+)%?")

    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        task_match = task_re.match(line)
        if task_match:
            task = task_match.group(1)
            current_task = task if task in task_names else None
            if current_task:
                out.setdefault(current_task, {})
            continue

        metric_match = metric_re.match(raw_line)
        if current_task and metric_match:
            key, value = metric_match.groups()
            out[current_task][key] = float(value)

    return out


def collect(log_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in sorted(log_dir.glob("tau*.log")):
        tau = parse_tau_from_name(path)
        if tau is None:
            continue
        parsed = parse_log(path)
        for task, _ in TASKS:
            metrics = parsed.get(task, {})
            if not all(src in metrics for _, _, src in METRICS):
                missing = [src for _, _, src in METRICS if src not in metrics]
                raise RuntimeError(f"{path} missing {task}: {missing}")
            rows.append(
                {
                    "tau": tau,
                    "task": task,
                    "acc": metrics["acc_bayes"],
                    "nll": metrics["nll_bayes"],
                    "ece": metrics["ece_bayes"],
                    "kind": "fixed",
                    "log": str(path),
                }
            )
    return sorted(rows, key=lambda row: (float(row["tau"]), str(row["task"])))


def append_log_rows(
    rows: list[dict[str, float | str]],
    path: Path,
    *,
    tau: float | None = None,
    kind: str,
) -> None:
    parsed = parse_log(path)
    for task, _ in TASKS:
        metrics = parsed.get(task, {})
        if not all(src in metrics for _, _, src in METRICS):
            missing = [src for _, _, src in METRICS if src not in metrics]
            raise RuntimeError(f"{path} missing {task}: {missing}")
        if tau is None:
            tau_value = None
            text = path.read_text(errors="replace")
            match = re.search(r"posterior_tau:\s+([0-9.]+)", text)
            if not match:
                raise RuntimeError(f"{path} missing posterior_tau")
            tau_value = float(match.group(1))
        else:
            tau_value = tau
        rows.append(
            {
                "tau": tau_value,
                "task": task,
                "acc": metrics["acc_bayes"],
                "nll": metrics["nll_bayes"],
                "ece": metrics["ece_bayes"],
                "kind": kind,
                "log": str(path),
            }
        )


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tau", "task", "acc", "nll", "ece", "kind", "log"])
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]], out_base: Path) -> None:
    fixed_rows = [row for row in rows if row.get("kind") == "fixed"]
    auto_rows = [row for row in rows if row.get("kind") == "auto"]
    by_task: dict[str, list[dict[str, float | str]]] = {task: [] for task, _ in TASKS}
    by_task_auto: dict[str, list[dict[str, float | str]]] = {task: [] for task, _ in TASKS}
    for row in fixed_rows:
        by_task[str(row["task"])].append(row)
    for row in auto_rows:
        by_task_auto[str(row["task"])].append(row)

    fig, axes = plt.subplots(3, 3, figsize=(8.2, 6.4), sharex=True)
    color = "#2F6F9F"
    highlight_color = "#D95F02"
    auto_color = "#6A3D9A"

    for j, (task, task_label) in enumerate(TASKS):
        task_rows = sorted(by_task[task], key=lambda row: float(row["tau"]))
        xs = [float(row["tau"]) for row in task_rows]
        for i, (metric, metric_label, _) in enumerate(METRICS):
            ax = axes[i][j]
            ys = [float(row[metric]) for row in task_rows]
            interp = None
            if len(xs) >= 4:
                smooth_xs = [x / 200 for x in range(0, 201)]
                interp = PchipInterpolator(xs, ys)
                smooth_ys = interp(smooth_xs)
                ax.plot(smooth_xs, smooth_ys, color=color, linewidth=1.9, label="fixed tau")
            else:
                ax.plot(xs, ys, color=color, linewidth=1.9, label="fixed tau")
            ax.scatter(xs, ys, color=color, s=16, zorder=3)

            for special_tau in (0.5,):
                if special_tau in xs:
                    idx = xs.index(special_tau)
                    ax.scatter(
                        [xs[idx]],
                        [ys[idx]],
                        marker="o",
                        facecolor="white",
                        edgecolor=highlight_color,
                        linewidth=1.7,
                        s=48,
                        zorder=4,
                        label=r"fixed $\tau=0.5$" if i == 0 and j == 0 else None,
                    )

            if by_task_auto[task]:
                auto_row = sorted(by_task_auto[task], key=lambda row: float(row["tau"]))[-1]
                auto_tau = float(auto_row["tau"])
                auto_y = float(auto_row[metric])
                ax.scatter(
                    [auto_tau],
                    [auto_y],
                    marker="*",
                    color=auto_color,
                    edgecolor="white",
                    linewidth=0.5,
                    s=82,
                    zorder=5,
                    label=r"auto $\tau$" if i == 0 and j == 0 else None,
                )
            ax.grid(True, alpha=0.22, linewidth=0.6)
            if i == 0:
                ax.set_title(task_label, fontsize=11)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if i == 2:
                ax.set_xlabel(r"Posterior scale $\tau$", fontsize=10)
            ax.set_xlim(-0.02, 1.02)
            ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(r"T=10 Fixed-$\tau$ Posterior-Scale Sensitivity", fontsize=13, y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/logs/T10"),
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path(
            "logs/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/figures/paper/"
            "t10_fixed_tau_sensitivity_3x3"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "logs/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/summary/"
            "t10_fixed_tau_sensitivity.csv"
        ),
    )
    parser.add_argument(
        "--tau0-log",
        type=Path,
        default=Path("logs/qwen3/scienceqa/scienceqa_ablations/slice_structure/map/seed1.log"),
    )
    parser.add_argument(
        "--auto-log",
        type=Path,
        default=Path("logs/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/logs/T10/auto.log"),
    )
    args = parser.parse_args()

    rows = collect(args.log_dir)
    append_log_rows(rows, args.tau0_log, tau=0.0, kind="fixed")
    append_log_rows(rows, args.auto_log, tau=None, kind="auto")
    rows = sorted(rows, key=lambda row: (str(row["kind"]), float(row["tau"]), str(row["task"])))
    taus = sorted({float(row["tau"]) for row in rows if row.get("kind") == "fixed"})
    if len(taus) != 11:
        raise RuntimeError(f"Expected 11 fixed tau values including tau=0, found {len(taus)}: {taus}")
    write_csv(rows, args.csv)
    plot(rows, args.out_base)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.out_base.with_suffix('.pdf')}")
    print(f"Wrote {args.out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
