from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
RERUN_DIR = LOGS_DIR / "map_mcdrop_ens_seedset_1_3_7_11_13_rerun_20260415_tmux"
BENCHMARK_LOGS_DIR = ROOT / "benchmark_suite_scienceqa" / "logs"

TASK_ORDER = [
    "iid",
    "grade12",
    "obqa",
    "arc-c",
    "mmlu_science_high",
    "mmlu_science_college",
    "gpqa_main",
]
METRIC_ORDER = ["acc", "nll", "ece", "brier"]
METHOD_ORDER = [
    "base",
    "map",
    "mcdrop",
    "ens",
    "laplace",
    "blob sample",
    "tfb sample",
    "clora sample",
    "seq",
]

TAG_WITH_SPLIT_RE = re.compile(r"^(?P<task>.+)\((?P<split>[^()]+)\)$")
SEQ_TAG_RE = re.compile(r"^(?P<task>.+)_(?P<split>iid|ood)$")

BASE_OR_MAP_OR_ENSEMBLE_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[(?P<method>BASE|MAP|ENSEMBLE)\]$")
BASE_OR_MAP_OR_ENSEMBLE_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)"
)
MCDROP_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[MCDROP\]$")
MCDROP_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+"
    r"Brier=(?P<brier>[0-9.]+)\s+std=(?P<std>[0-9.]+)\s+mc=(?P<mc>[0-9.]+)"
)
LAP_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\s+n=(?P<n>[0-9]+)$")
LAP_METRICS_RE = re.compile(
    r"^(?P<method>MAP|LAP):\s+NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+"
    r"ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)$"
)
BLOB_OR_CLORA_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\s+Results\]$")
BLOB_OR_CLORA_METRICS_RE = re.compile(
    r"^\s*(?P<method>BLoB mean|BLoB samp|C-LoRA mean|C-LoRA samp)\s*: NLL=(?P<nll>[0-9.]+)\s+"
    r"ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)"
)
TFB_HEADER_RE = re.compile(r"^\[(?P<tag>.+)\]\[TFB\]$")
TFB_METRICS_RE = re.compile(
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+ECE=(?P<ece>[0-9.]+)%\s+"
    r"Brier=(?P<brier>[0-9.]+)\s+mc=(?P<mc>[0-9.]+)\s+beta=(?P<beta>[0-9.]+)"
)
SEQ_BLOCK_RE = re.compile(
    r"\[(?P<tag>[^\]]+)\]\s*\n\s*===== Bayesian \(Seq-LoRA\) Only =====\s*\n"
    r"\s*nll_bayes:\s*(?P<nll>[0-9.]+)\s*\n"
    r"\s*brier_bayes:\s*(?P<brier>[0-9.]+)\s*\n"
    r"\s*ece_bayes:\s*(?P<ece>[0-9.]+)%\s*\n"
    r"\s*acc_bayes:\s*(?P<acc>[0-9.]+)%",
    re.MULTILINE,
)


def normalize_task_from_parenthetical(tag: str) -> str:
    match = TAG_WITH_SPLIT_RE.match(tag.strip())
    if not match:
        raise ValueError(f"Could not parse tag: {tag}")
    task = match.group("task")
    if task == "scienceqa_closedchoice_grade2_11":
        return "iid"
    if task == "scienceqa_closedchoice_grade12":
        return "grade12"
    return task


def normalize_task_from_seq(tag: str) -> str:
    match = SEQ_TAG_RE.match(tag.strip())
    if not match:
        raise ValueError(f"Could not parse seq tag: {tag}")
    task = match.group("task")
    if task == "scienceqa_closedchoice_grade2_11":
        return "iid"
    if task == "scienceqa_closedchoice_grade12":
        return "grade12"
    return task


def new_metric_store() -> dict[str, dict[str, list[float]]]:
    return {task: {metric: [] for metric in METRIC_ORDER} for task in TASK_ORDER}


def add_metrics(store: dict[str, dict[str, list[float]]], task: str, metrics: dict[str, float]) -> None:
    if task not in store:
        raise ValueError(f"Unexpected task {task}")
    for metric, value in metrics.items():
        store[task][metric].append(float(value))


def parse_base_map_or_ensemble(path: Path) -> dict[str, dict[str, list[float]]]:
    store = new_metric_store()
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        header = BASE_OR_MAP_OR_ENSEMBLE_HEADER_RE.match(line.strip())
        if not header or idx + 1 >= len(lines):
            continue
        metrics = BASE_OR_MAP_OR_ENSEMBLE_METRICS_RE.search(lines[idx + 1].strip())
        if not metrics:
            continue
        task = normalize_task_from_parenthetical(header.group("tag"))
        add_metrics(
            store,
            task,
            {
                "acc": float(metrics.group("acc")),
                "nll": float(metrics.group("nll")),
                "ece": float(metrics.group("ece")),
                "brier": float(metrics.group("brier")),
            },
        )
    return store


def parse_mcdrop(path: Path) -> dict[str, dict[str, list[float]]]:
    store = new_metric_store()
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        header = MCDROP_HEADER_RE.match(line.strip())
        if not header or idx + 1 >= len(lines):
            continue
        metrics = MCDROP_METRICS_RE.search(lines[idx + 1].strip())
        if not metrics:
            continue
        task = normalize_task_from_parenthetical(header.group("tag"))
        add_metrics(
            store,
            task,
            {
                "acc": float(metrics.group("acc")),
                "nll": float(metrics.group("nll")),
                "ece": float(metrics.group("ece")),
                "brier": float(metrics.group("brier")),
            },
        )
    return store


def parse_laplace(path: Path) -> dict[str, dict[str, list[float]]]:
    store = new_metric_store()
    current_task: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = LAP_HEADER_RE.match(line.strip())
        if header:
            current_task = normalize_task_from_parenthetical(header.group("tag"))
            continue
        metrics = LAP_METRICS_RE.match(line.strip())
        if metrics and metrics.group("method") == "LAP" and current_task is not None:
            add_metrics(
                store,
                current_task,
                {
                    "acc": float(metrics.group("acc")),
                    "nll": float(metrics.group("nll")),
                    "ece": float(metrics.group("ece")),
                    "brier": float(metrics.group("brier")),
                },
            )
    return store


def parse_blob_or_clora_sample(path: Path, sample_label: str) -> dict[str, dict[str, list[float]]]:
    store = new_metric_store()
    current_task: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = BLOB_OR_CLORA_HEADER_RE.match(line.strip())
        if header:
            current_task = normalize_task_from_parenthetical(header.group("tag"))
            continue
        metrics = BLOB_OR_CLORA_METRICS_RE.match(line.strip())
        if metrics and metrics.group("method") == sample_label and current_task is not None:
            add_metrics(
                store,
                current_task,
                {
                    "acc": float(metrics.group("acc")),
                    "nll": float(metrics.group("nll")),
                    "ece": float(metrics.group("ece")),
                    "brier": float(metrics.group("brier")),
                },
            )
    return store


def parse_tfb(path: Path) -> dict[str, dict[str, list[float]]]:
    store = new_metric_store()
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        header = TFB_HEADER_RE.match(line.strip())
        if not header or idx + 1 >= len(lines):
            continue
        metrics = TFB_METRICS_RE.search(lines[idx + 1].strip())
        if not metrics:
            continue
        task = normalize_task_from_parenthetical(header.group("tag"))
        add_metrics(
            store,
            task,
            {
                "acc": float(metrics.group("acc")),
                "nll": float(metrics.group("nll")),
                "ece": float(metrics.group("ece")),
                "brier": float(metrics.group("brier")),
            },
        )
    return store


def parse_seq(path: Path) -> dict[str, dict[str, list[float]]]:
    store = new_metric_store()
    text = path.read_text(encoding="utf-8")
    for match in SEQ_BLOCK_RE.finditer(text):
        task = normalize_task_from_seq(match.group("tag"))
        add_metrics(
            store,
            task,
            {
                "acc": float(match.group("acc")),
                "nll": float(match.group("nll")),
                "ece": float(match.group("ece")),
                "brier": float(match.group("brier")),
            },
        )
    return store


def merge_stores(stores: list[dict[str, dict[str, list[float]]]]) -> dict[str, dict[str, list[float]]]:
    merged = new_metric_store()
    for store in stores:
        for task in TASK_ORDER:
            for metric in METRIC_ORDER:
                merged[task][metric].extend(store[task][metric])
    return merged


def summarize(store: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, tuple[float, float, int]]]:
    summary: dict[str, dict[str, tuple[float, float, int]]] = {}
    for task in TASK_ORDER:
        summary[task] = {}
        for metric in METRIC_ORDER:
            values = store[task][metric]
            if not values:
                raise ValueError(f"Missing values for task={task} metric={metric}")
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[task][metric] = (mean, sd, len(values))
    return summary


def format_cell(metric: str, mean: float, sd: float) -> str:
    if metric in {"acc", "ece"}:
        return f"{mean:.2f} ± {sd:.2f}%"
    return f"{mean:.4f} ± {sd:.4f}"


def build_method_summaries() -> dict[str, dict[str, dict[str, tuple[float, float, int]]]]:
    ensemble_group_logs = sorted((LOGS_DIR).glob("ensemble_order_group*.log"))
    if len(ensemble_group_logs) < 5:
        ensemble_group_logs = sorted(BENCHMARK_LOGS_DIR.glob("ensemble_order_group*.log"))
    if not ensemble_group_logs:
        ensemble_group_logs = [RERUN_DIR / "map_ensemble_eval_seedset_1_3_7_11_13.log"]

    method_files: dict[str, tuple[list[Path], str]] = {
        "base": ([LOGS_DIR / "base_seed0.log"], "base"),
        "map": (sorted(RERUN_DIR.glob("map_eval_seed*.log")), "map"),
        "mcdrop": (sorted(RERUN_DIR.glob("mcdrop_eval_seed*.log")), "mcdrop"),
        "ens": (ensemble_group_logs, "ens"),
        "laplace": (sorted((LOGS_DIR / "laplace").glob("laplace_seed*.log")), "laplace"),
        "blob sample": (sorted((LOGS_DIR / "thirdparty_blob_train_once_mc32").glob("seed_*.log")), "blob sample"),
        "tfb sample": (sorted((LOGS_DIR / "tfb" / "logs").glob("official_tfblora_bench_lora_seed*.log")), "tfb sample"),
        "clora sample": (sorted((LOGS_DIR / "thirdparty_clora_train_once_mc32_eval100").glob("seed_*.log")), "clora sample"),
        "seq": (sorted((LOGS_DIR / "seq").glob("seq_constantq4_order_seed*.log")), "seq"),
    }

    summaries: dict[str, dict[str, dict[str, tuple[float, float, int]]]] = {}
    for method, (paths, _) in method_files.items():
        if not paths:
            raise ValueError(f"No log files found for {method}")
        stores = []
        for path in paths:
            if method in {"base", "map", "ens"}:
                stores.append(parse_base_map_or_ensemble(path))
            elif method == "mcdrop":
                stores.append(parse_mcdrop(path))
            elif method == "laplace":
                stores.append(parse_laplace(path))
            elif method == "blob sample":
                stores.append(parse_blob_or_clora_sample(path, "BLoB samp"))
            elif method == "tfb sample":
                stores.append(parse_tfb(path))
            elif method == "clora sample":
                stores.append(parse_blob_or_clora_sample(path, "C-LoRA samp"))
            elif method == "seq":
                stores.append(parse_seq(path))
            else:
                raise ValueError(f"Unhandled method {method}")
        summaries[method] = summarize(merge_stores(stores))
    return summaries


def write_markdown_table(
    path: Path,
    summaries: dict[str, dict[str, dict[str, tuple[float, float, int]]]],
) -> None:
    header = ["Metric", "Method", *TASK_ORDER]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for metric in METRIC_ORDER:
        for idx, method in enumerate(METHOD_ORDER):
            row = [metric.upper() if idx == 0 else "", method]
            for task in TASK_ORDER:
                mean, sd, _n = summaries[method][task][metric]
                row.append(format_cell(metric, mean, sd))
            lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(
    path: Path,
    summaries: dict[str, dict[str, dict[str, tuple[float, float, int]]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "method", *TASK_ORDER])
        for metric in METRIC_ORDER:
            for method in METHOD_ORDER:
                row = [metric.upper(), method]
                for task in TASK_ORDER:
                    mean, sd, _n = summaries[method][task][metric]
                    row.append(format_cell(metric, mean, sd))
                writer.writerow(row)


def write_stats_csv(
    path: Path,
    summaries: dict[str, dict[str, dict[str, tuple[float, float, int]]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "task", "metric", "mean", "sd", "n"])
        for method in METHOD_ORDER:
            for task in TASK_ORDER:
                for metric in METRIC_ORDER:
                    mean, sd, n = summaries[method][task][metric]
                    writer.writerow([method, task, metric, f"{mean:.10f}", f"{sd:.10f}", n])


def main() -> None:
    summaries = build_method_summaries()
    write_markdown_table(LOGS_DIR / "all_methods_mean_sd_table.md", summaries)
    write_csv_table(LOGS_DIR / "all_methods_mean_sd_table.csv", summaries)
    write_stats_csv(LOGS_DIR / "all_methods_mean_sd_stats.csv", summaries)
    print("Wrote:")
    print(LOGS_DIR / "all_methods_mean_sd_table.md")
    print(LOGS_DIR / "all_methods_mean_sd_table.csv")
    print(LOGS_DIR / "all_methods_mean_sd_stats.csv")


if __name__ == "__main__":
    main()
