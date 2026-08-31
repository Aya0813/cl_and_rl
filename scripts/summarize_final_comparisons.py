import argparse
import re
from pathlib import Path
from statistics import mean, stdev


SECTION_HEADER = "========== Final Comparison =========="
SECTION_FOOTER = "====================================="

FLOAT_PATTERNS = {
    "teacher_final_test_loss": r"^Frozen Teacher\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+",
    "teacher_final_test_acc": r"^Frozen Teacher\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "uniform_final_test_loss": r"^Uniform Baseline\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+",
    "uniform_final_test_acc": r"^Uniform Baseline\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "delta_final": r"^Delta_final,s = .* = ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "mean_final_loss_teacher": r"^Mean final loss for teacher: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "mean_final_loss_uniform": r"^Mean final loss for uniform: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "mean_paired_improvement": r"^Mean paired improvement: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "std_paired_improvement": r"^Std paired improvement: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "mean_val_auc_teacher": r"^Mean validation-loss AUC for teacher: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "mean_val_auc_uniform": r"^Mean validation-loss AUC for uniform: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    "paired_auc_improvement": r"^Paired AUC improvement \(uniform - teacher\): ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
}

SEED_PATTERN = re.compile(r"^seed:\s*(\d+)\s*$", re.MULTILINE)
SEEDS_WON_PATTERN = re.compile(r"^Number of seeds won: (\d+)\s*/\s*(\d+)$", re.MULTILINE)


def extract_final_comparison_section(text):
    start = text.find(SECTION_HEADER)
    if start == -1:
        raise ValueError(f"Could not find section header: {SECTION_HEADER}")

    end = text.find(SECTION_FOOTER, start)
    if end == -1:
        raise ValueError(f"Could not find section footer after: {SECTION_HEADER}")

    end += len(SECTION_FOOTER)
    return text[start:end]


def extract_seed(text, path):
    match = SEED_PATTERN.search(text)
    if match:
        return int(match.group(1))

    filename_match = re.search(r"seed[_-]?(\d+)", path.stem)
    if filename_match:
        return int(filename_match.group(1))

    return path.stem


def parse_float_metric(section, key, pattern):
    match = re.search(pattern, section, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing metric '{key}' in final comparison section")
    return float(match.group(1))


def parse_final_comparison(path):
    text = path.read_text()
    section = extract_final_comparison_section(text)

    metrics = {
        key: parse_float_metric(section, key, pattern)
        for key, pattern in FLOAT_PATTERNS.items()
    }

    seeds_won_match = SEEDS_WON_PATTERN.search(section)
    if not seeds_won_match:
        raise ValueError("Missing 'Number of seeds won' metric in final comparison section")

    metrics["num_seeds_won"] = int(seeds_won_match.group(1))
    metrics["num_pairs"] = int(seeds_won_match.group(2))
    metrics["seed"] = extract_seed(text, path)
    metrics["source_file"] = str(path)
    return metrics


def aggregate_metrics(per_seed_metrics):
    aggregate_keys = [
        "teacher_final_test_loss",
        "teacher_final_test_acc",
        "uniform_final_test_loss",
        "uniform_final_test_acc",
        "delta_final",
        "mean_final_loss_teacher",
        "mean_final_loss_uniform",
        "mean_paired_improvement",
        "std_paired_improvement",
        "mean_val_auc_teacher",
        "mean_val_auc_uniform",
        "paired_auc_improvement",
    ]

    aggregated = {}
    for key in aggregate_keys:
        values = [metrics[key] for metrics in per_seed_metrics]
        aggregated[f"avg_{key}"] = mean(values)
        aggregated[f"std_{key}"] = stdev(values) if len(values) > 1 else 0.0

    aggregated["total_seed_files"] = len(per_seed_metrics)
    aggregated["seeds_where_teacher_beats_uniform"] = sum(
        metrics["delta_final"] > 0 for metrics in per_seed_metrics
    )
    aggregated["seeds_where_teacher_has_better_auc"] = sum(
        metrics["paired_auc_improvement"] > 0 for metrics in per_seed_metrics
    )
    return aggregated


def build_report(per_seed_metrics, aggregated):
    lines = []
    lines.append("========== Cross-Seed Final Comparison Summary ==========")
    lines.append(f"Analyzed seed files: {aggregated['total_seed_files']}")
    lines.append("")
    lines.append("Per-seed extracted metrics:")

    for metrics in per_seed_metrics:
        lines.append(f"Seed: {metrics['seed']}")
        lines.append(f"Source file: {metrics['source_file']}")
        lines.append(f"Frozen teacher final test loss: {metrics['teacher_final_test_loss']:.6f}")
        lines.append(f"Frozen teacher final test acc: {metrics['teacher_final_test_acc']:.6f}")
        lines.append(f"Uniform final test loss: {metrics['uniform_final_test_loss']:.6f}")
        lines.append(f"Uniform final test acc: {metrics['uniform_final_test_acc']:.6f}")
        lines.append(f"Delta_final,s (uniform - teacher): {metrics['delta_final']:.6f}")
        lines.append(f"Mean final loss for teacher: {metrics['mean_final_loss_teacher']:.6f}")
        lines.append(f"Mean final loss for uniform: {metrics['mean_final_loss_uniform']:.6f}")
        lines.append(f"Mean paired improvement: {metrics['mean_paired_improvement']:.6f}")
        lines.append(f"Std paired improvement: {metrics['std_paired_improvement']:.6f}")
        lines.append(f"Mean validation-loss AUC for teacher: {metrics['mean_val_auc_teacher']:.6f}")
        lines.append(f"Mean validation-loss AUC for uniform: {metrics['mean_val_auc_uniform']:.6f}")
        lines.append(f"Paired AUC improvement (uniform - teacher): {metrics['paired_auc_improvement']:.6f}")
        lines.append("")

    lines.append("Across-seed averages:")
    lines.append(f"Average frozen teacher final test loss: {aggregated['avg_teacher_final_test_loss']:.6f}")
    lines.append(f"Std frozen teacher final test loss: {aggregated['std_teacher_final_test_loss']:.6f}")
    lines.append(f"Average frozen teacher final test acc: {aggregated['avg_teacher_final_test_acc']:.6f}")
    lines.append(f"Std frozen teacher final test acc: {aggregated['std_teacher_final_test_acc']:.6f}")
    lines.append(f"Average uniform final test loss: {aggregated['avg_uniform_final_test_loss']:.6f}")
    lines.append(f"Std uniform final test loss: {aggregated['std_uniform_final_test_loss']:.6f}")
    lines.append(f"Average uniform final test acc: {aggregated['avg_uniform_final_test_acc']:.6f}")
    lines.append(f"Std uniform final test acc: {aggregated['std_uniform_final_test_acc']:.6f}")
    lines.append(f"Average Delta_final,s (uniform - teacher): {aggregated['avg_delta_final']:.6f}")
    lines.append(f"Std Delta_final,s (uniform - teacher): {aggregated['std_delta_final']:.6f}")
    lines.append(f"Average mean final loss for teacher: {aggregated['avg_mean_final_loss_teacher']:.6f}")
    lines.append(f"Average mean final loss for uniform: {aggregated['avg_mean_final_loss_uniform']:.6f}")
    lines.append(f"Average mean paired improvement: {aggregated['avg_mean_paired_improvement']:.6f}")
    lines.append(f"Std mean paired improvement: {aggregated['std_mean_paired_improvement']:.6f}")
    lines.append(f"Average per-seed std paired improvement: {aggregated['avg_std_paired_improvement']:.6f}")
    lines.append(f"Average teacher validation-loss AUC: {aggregated['avg_mean_val_auc_teacher']:.6f}")
    lines.append(f"Average uniform validation-loss AUC: {aggregated['avg_mean_val_auc_uniform']:.6f}")
    lines.append(f"Average paired AUC improvement (uniform - teacher): {aggregated['avg_paired_auc_improvement']:.6f}")
    lines.append(f"Std paired AUC improvement (uniform - teacher): {aggregated['std_paired_auc_improvement']:.6f}")
    lines.append(
        f"Seeds where teacher beats uniform on final loss: "
        f"{aggregated['seeds_where_teacher_beats_uniform']} / {aggregated['total_seed_files']}"
    )
    lines.append(
        f"Seeds where teacher beats uniform on validation-loss AUC: "
        f"{aggregated['seeds_where_teacher_has_better_auc']} / {aggregated['total_seed_files']}"
    )
    lines.append("========================================================")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the final comparison section from multiple .out files and summarize metrics across seeds."
    )
    parser.add_argument(
        "out_files",
        nargs="+",
        help="Paths to the .out files to analyze.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_comparison_summary.out",
        help="Path to the output .out summary file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_paths = [Path(path) for path in args.out_files]

    per_seed_metrics = [parse_final_comparison(path) for path in out_paths]
    per_seed_metrics.sort(key=lambda metrics: str(metrics["seed"]))
    aggregated = aggregate_metrics(per_seed_metrics)
    report = build_report(per_seed_metrics, aggregated)

    output_path = Path(args.output)
    output_path.write_text(report)
    print(report, end="")
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
