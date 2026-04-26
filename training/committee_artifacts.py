"""Artifact helpers for Round 2 training and judging outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Optional plotting dependency for Colab / GPU environments.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from training.committee_eval import DEFAULT_SEEDS, _parse_seeds, evaluate_policy, summarize_rows

PREFERRED_METRIC_KEYS = (
    "reward",
    "rewards/task_score_reward",
    "task_score_reward",
    "objective",
    "loss",
)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the output directory."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def export_baseline_report(
    output_dir: str | Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Path:
    """Write a baseline comparison JSON for random and heuristic policies."""

    output_path = ensure_output_dir(output_dir) / "baseline_report.json"
    payload: dict[str, Any] = {}
    for policy_name in ("random", "heuristic"):
        rows = evaluate_policy(policy_name, seeds=seeds)
        payload[policy_name] = {
            "rows": rows,
            "summary": summarize_rows(rows),
        }
    return _write_json(output_path, payload)


def export_training_log_history(output_dir: str | Path, log_history: list[dict[str, Any]]) -> Path:
    """Persist raw trainer log history to disk for later plotting."""

    output_path = ensure_output_dir(output_dir) / "training_log_history.json"
    return _write_json(output_path, log_history)


def load_training_log_history(path: str | Path) -> list[dict[str, Any]]:
    """Load previously saved trainer log history."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list for training_log_history.")
    return payload


def load_json(path: str | Path) -> Any:
    """Load arbitrary JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def available_numeric_keys(log_history: list[dict[str, Any]]) -> list[str]:
    """List numeric metric keys observed in trainer log history."""

    keys: set[str] = set()
    for entry in log_history:
        for key, value in entry.items():
            if key in {"step", "epoch", "total_flos"}:
                continue
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def choose_metric_key(log_history: list[dict[str, Any]], metric_key: str | None = None) -> str:
    """Select the most useful metric key for plotting."""

    if metric_key:
        return metric_key
    keys = available_numeric_keys(log_history)
    for preferred in PREFERRED_METRIC_KEYS:
        if preferred in keys:
            return preferred
    if not keys:
        raise ValueError("No numeric training metrics found in log history.")
    return keys[0]


def build_metric_series(
    log_history: list[dict[str, Any]],
    metric_key: str | None = None,
) -> dict[str, Any]:
    """Extract a single metric series from trainer log history."""

    selected_key = choose_metric_key(log_history, metric_key=metric_key)
    steps: list[int] = []
    values: list[float] = []
    for index, entry in enumerate(log_history):
        if selected_key not in entry:
            continue
        step = entry.get("step", index + 1)
        value = entry[selected_key]
        if not isinstance(value, (int, float)):
            continue
        steps.append(int(step))
        values.append(float(value))
    if not steps:
        raise ValueError(f"Metric {selected_key!r} has no numeric values in log history.")
    return {
        "metric_key": selected_key,
        "points": [{"step": step, "value": value} for step, value in zip(steps, values)],
    }


def export_metric_series(
    output_dir: str | Path,
    log_history: list[dict[str, Any]],
    metric_key: str | None = None,
) -> Path:
    """Write the chosen metric series to disk as JSON."""

    payload = build_metric_series(log_history, metric_key=metric_key)
    output_path = ensure_output_dir(output_dir) / f"{payload['metric_key'].replace('/', '_')}_series.json"
    return _write_json(output_path, payload)


def plot_metric_series(
    log_history: list[dict[str, Any]],
    output_dir: str | Path,
    metric_key: str | None = None,
    title: str | None = None,
) -> Path:
    """Render a PNG chart for a selected training metric."""

    if plt is None:
        raise ImportError("matplotlib is required to plot training metrics.")

    payload = build_metric_series(log_history, metric_key=metric_key)
    metric_name = str(payload["metric_key"])
    points = payload["points"]
    steps = [point["step"] for point in points]
    values = [point["value"] for point in points]

    output_path = ensure_output_dir(output_dir) / f"{metric_name.replace('/', '_')}_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, marker="o", linewidth=1.8)
    plt.xlabel("Trainer step")
    plt.ylabel(metric_name)
    plt.title(title or f"Training curve: {metric_name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def summarize_metric_progress(
    log_history: list[dict[str, Any]],
    metric_key: str | None = None,
) -> dict[str, Any]:
    """Summarize improvement across a trainer metric series."""

    payload = build_metric_series(log_history, metric_key=metric_key)
    points = payload["points"]
    values = [point["value"] for point in points]
    best_index = max(range(len(values)), key=values.__getitem__)
    return {
        "metric_key": payload["metric_key"],
        "start": values[0],
        "end": values[-1],
        "best": max(values),
        "best_step": points[best_index]["step"],
        "delta": round(values[-1] - values[0], 10),
        "best_delta": round(max(values) - values[0], 10),
        "num_points": len(values),
    }


def _baseline_delta_summary(baseline_payload: dict[str, Any]) -> dict[str, float]:
    heuristic = baseline_payload["heuristic"]["summary"]
    random = baseline_payload["random"]["summary"]
    return {
        "score_delta_vs_random": round(heuristic["score"] - random["score"], 4),
        "return_delta_vs_random": round(heuristic["total_return"] - random["total_return"], 4),
        "compliance_delta_vs_random": round(
            heuristic["compliance_score"] - random["compliance_score"],
            4,
        ),
        "risk_response_delta_vs_random": round(
            heuristic["risk_response"] - random["risk_response"],
            4,
        ),
    }


def build_judging_report(
    output_dir: str | Path,
    *,
    model_name: str,
    colab_account_email: str,
    baseline_payload: dict[str, Any],
    log_history: list[dict[str, Any]] | None = None,
    metric_key: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build a compact judging report from baseline and training artifacts."""

    report: dict[str, Any] = {
        "project": "AI Investment Committee Environment",
        "model_name": model_name,
        "colab_account_email": colab_account_email,
        "output_dir": str(Path(output_dir)),
        "baseline": baseline_payload,
        "baseline_deltas": _baseline_delta_summary(baseline_payload),
    }
    if log_history:
        report["training_metric"] = summarize_metric_progress(log_history, metric_key=metric_key)
    if notes:
        report["notes"] = notes
    return report


def export_judging_report(
    output_dir: str | Path,
    *,
    model_name: str,
    colab_account_email: str,
    baseline_payload: dict[str, Any],
    log_history: list[dict[str, Any]] | None = None,
    metric_key: str | None = None,
    notes: str | None = None,
) -> tuple[Path, Path]:
    """Write compact JSON and Markdown judging reports."""

    output_dir_path = ensure_output_dir(output_dir)
    report = build_judging_report(
        output_dir_path,
        model_name=model_name,
        colab_account_email=colab_account_email,
        baseline_payload=baseline_payload,
        log_history=log_history,
        metric_key=metric_key,
        notes=notes,
    )
    json_path = _write_json(output_dir_path / "judging_report.json", report)

    heuristic = report["baseline"]["heuristic"]["summary"]
    random = report["baseline"]["random"]["summary"]
    lines = [
        "# Round 2 Judging Report",
        "",
        f"- Project: {report['project']}",
        f"- Model: {report['model_name']}",
        f"- Colab account: {report['colab_account_email']}",
        "",
        "## Baseline Summary",
        "",
        f"- Heuristic overall score: {heuristic['score']:.4f}",
        f"- Random overall score: {random['score']:.4f}",
        f"- Score delta vs random: {report['baseline_deltas']['score_delta_vs_random']:.4f}",
        f"- Return delta vs random: {report['baseline_deltas']['return_delta_vs_random']:.4f}",
        "",
    ]
    if "training_metric" in report:
        metric = report["training_metric"]
        lines.extend(
            [
                "## Training Metric",
                "",
                f"- Metric key: {metric['metric_key']}",
                f"- Start: {metric['start']:.4f}",
                f"- End: {metric['end']:.4f}",
                f"- Best: {metric['best']:.4f}",
                f"- Best step: {metric['best_step']}",
                f"- Delta: {metric['delta']:.4f}",
                f"- Best delta: {metric['best_delta']:.4f}",
                "",
            ]
        )
    if notes:
        lines.extend(["## Notes", "", notes, ""])

    markdown_path = output_dir_path / "judging_report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def export_onsite_demo_summary(
    output_dir: str | Path,
    *,
    model_name: str,
    colab_account_email: str,
    baseline_payload: dict[str, Any],
    log_history: list[dict[str, Any]] | None = None,
    metric_key: str | None = None,
    notes: str | None = None,
) -> Path:
    """Write a concise onsite-ready demo summary for judges and mentors."""

    output_dir_path = ensure_output_dir(output_dir)
    report = build_judging_report(
        output_dir_path,
        model_name=model_name,
        colab_account_email=colab_account_email,
        baseline_payload=baseline_payload,
        log_history=log_history,
        metric_key=metric_key,
        notes=notes,
    )
    heuristic = report["baseline"]["heuristic"]["summary"]
    random = report["baseline"]["random"]["summary"]

    lines = [
        "# Onsite Demo Summary",
        "",
        "## Project",
        "",
        "- Name: AI Investment Committee Environment",
        "- Primary theme: Multi-Agent Interactions",
        "- Secondary themes: World Modeling (Professional Tasks), Long-Horizon Planning",
        f"- Model: {model_name}",
        f"- Colab account: {colab_account_email}",
        "",
        "## What We Built",
        "",
        "A multi-agent OpenEnv environment where a trainable Portfolio Manager learns to "
        "coordinate with a scripted Research Analyst and Risk Officer while managing a "
        "portfolio through noisy signals, hidden market regimes, and changing constraints.",
        "",
        "## Task Ladder",
        "",
        "- guided_allocation",
        "- research_risk_conflict",
        "- regime_shift_recovery",
        "- mandate_drift",
        "",
        "## Why The Reward Is Hard To Game",
        "",
        "- multiple verifier-style reward functions instead of one fuzzy scalar",
        "- progress-weighted rewards so stopping early does not win",
        "- compliance, risk response, and information usage scored separately from return",
        "",
        "## Baseline Snapshot",
        "",
        f"- Heuristic overall score: {heuristic['score']:.4f}",
        f"- Random overall score: {random['score']:.4f}",
        f"- Score delta vs random: {report['baseline_deltas']['score_delta_vs_random']:.4f}",
        f"- Return delta vs random: {report['baseline_deltas']['return_delta_vs_random']:.4f}",
        "",
    ]
    if "training_metric" in report:
        metric = report["training_metric"]
        lines.extend(
            [
                "## Training Snapshot",
                "",
                f"- Metric key: {metric['metric_key']}",
                f"- Start: {metric['start']:.4f}",
                f"- End: {metric['end']:.4f}",
                f"- Best: {metric['best']:.4f}",
                f"- Best step: {metric['best_step']}",
                f"- Final delta: {metric['delta']:.4f}",
                f"- Best delta: {metric['best_delta']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## What To Show Live",
            "",
            "- reward_curve.png",
            "- judging_report.md",
            "- baseline_report.json",
            "- one quick walkthrough of the four tasks and the PM actions",
            "",
            "## 3-Minute Demo Order",
            "",
            "1. Explain the committee roles and why this is not a single-step allocator.",
            "2. Show the four tasks and the verifier-style reward design.",
            "3. Show heuristic vs random baselines.",
            "4. Show the Colab training curve and best-step improvement.",
            "5. Close with why this environment is useful for training financial workflow agents.",
            "",
        ]
    )
    if notes:
        lines.extend(["## Notes", "", notes, ""])

    markdown_path = output_dir_path / "onsite_demo_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Round 2 training artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baselines", help="Export heuristic/random baselines.")
    baseline_parser.add_argument("--output-dir", default="outputs/committee-grpo")
    baseline_parser.add_argument("--seeds", default="7,11,13")

    plot_parser = subparsers.add_parser("plot", help="Plot a metric from training log history.")
    plot_parser.add_argument("--log-file", required=True)
    plot_parser.add_argument("--output-dir", default="outputs/committee-grpo")
    plot_parser.add_argument("--metric-key", default=None)

    series_parser = subparsers.add_parser("series", help="Export a metric series as JSON.")
    series_parser.add_argument("--log-file", required=True)
    series_parser.add_argument("--output-dir", default="outputs/committee-grpo")
    series_parser.add_argument("--metric-key", default=None)

    report_parser = subparsers.add_parser("report", help="Export a compact judging report.")
    report_parser.add_argument("--output-dir", default="outputs/committee-grpo")
    report_parser.add_argument("--model-name", required=True)
    report_parser.add_argument("--colab-email", required=True)
    report_parser.add_argument("--baseline-file", default=None)
    report_parser.add_argument("--log-file", default=None)
    report_parser.add_argument("--metric-key", default=None)
    report_parser.add_argument("--notes", default=None)

    demo_parser = subparsers.add_parser("demo", help="Export an onsite demo summary.")
    demo_parser.add_argument("--output-dir", default="outputs/committee-grpo")
    demo_parser.add_argument("--model-name", required=True)
    demo_parser.add_argument("--colab-email", required=True)
    demo_parser.add_argument("--baseline-file", default=None)
    demo_parser.add_argument("--log-file", default=None)
    demo_parser.add_argument("--metric-key", default=None)
    demo_parser.add_argument("--notes", default=None)

    args = parser.parse_args()

    if args.command == "baselines":
        output_path = export_baseline_report(args.output_dir, seeds=_parse_seeds(args.seeds))
        print(output_path)
        return

    if args.command == "report":
        baseline_file = args.baseline_file
        if baseline_file is None:
            baseline_file = export_baseline_report(args.output_dir)
        baseline_payload = load_json(baseline_file)
        log_history = load_training_log_history(args.log_file) if args.log_file else None
        json_path, markdown_path = export_judging_report(
            args.output_dir,
            model_name=args.model_name,
            colab_account_email=args.colab_email,
            baseline_payload=baseline_payload,
            log_history=log_history,
            metric_key=args.metric_key,
            notes=args.notes,
        )
        print(json_path)
        print(markdown_path)
        return

    if args.command == "demo":
        baseline_file = args.baseline_file
        if baseline_file is None:
            baseline_file = export_baseline_report(args.output_dir)
        baseline_payload = load_json(baseline_file)
        log_history = load_training_log_history(args.log_file) if args.log_file else None
        output_path = export_onsite_demo_summary(
            args.output_dir,
            model_name=args.model_name,
            colab_account_email=args.colab_email,
            baseline_payload=baseline_payload,
            log_history=log_history,
            metric_key=args.metric_key,
            notes=args.notes,
        )
        print(output_path)
        return

    log_history = load_training_log_history(args.log_file)

    if args.command == "plot":
        output_path = plot_metric_series(
            log_history,
            output_dir=args.output_dir,
            metric_key=args.metric_key,
        )
        print(output_path)
        return

    output_path = export_metric_series(
        args.output_dir,
        log_history=log_history,
        metric_key=args.metric_key,
    )
    print(output_path)


if __name__ == "__main__":
    main()
