"""Evaluation helpers for the Round 2 committee environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import run_episode
from tasks import TASK_ORDER

DEFAULT_SEEDS = (7, 11, 13)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def evaluate_policy(
    policy_name: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[dict[str, float]]:
    """Run a policy across all tasks and aggregate a few stable metrics."""

    rows: list[dict[str, float]] = []
    for task_id in TASK_ORDER:
        reports = [run_episode(task_id, policy_name, seed=seed) for seed in seeds]
        rows.append(
            {
                "task": task_id,
                "policy": policy_name,
                "score": round(mean(report.score for report in reports), 4),
                "total_return": round(mean(report.metrics.total_return for report in reports), 4),
                "max_drawdown": round(mean(report.metrics.max_drawdown for report in reports), 4),
                "compliance_score": round(
                    mean(report.metrics.compliance_score for report in reports),
                    4,
                ),
                "information_usage": round(
                    mean(report.metrics.information_usage for report in reports),
                    4,
                ),
                "risk_response": round(
                    mean(report.metrics.risk_response for report in reports),
                    4,
                ),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Return an overall summary row across all evaluated tasks."""

    return {
        "task": "overall",
        "policy": rows[0]["policy"] if rows else "n/a",
        "score": round(mean(row["score"] for row in rows), 4) if rows else 0.0,
        "total_return": round(mean(row["total_return"] for row in rows), 4) if rows else 0.0,
        "max_drawdown": round(mean(row["max_drawdown"] for row in rows), 4) if rows else 0.0,
        "compliance_score": round(mean(row["compliance_score"] for row in rows), 4)
        if rows
        else 0.0,
        "information_usage": round(mean(row["information_usage"] for row in rows), 4)
        if rows
        else 0.0,
        "risk_response": round(mean(row["risk_response"] for row in rows), 4) if rows else 0.0,
    }


def print_eval_table(rows: list[dict[str, float]]) -> None:
    """Pretty-print evaluation rows."""

    for row in rows:
        print(
            f"{row['task']:<24} policy={row['policy']:<10} score={row['score']:.3f} "
            f"ret={row['total_return']:.3f} dd={row['max_drawdown']:.3f} "
            f"comp={row['compliance_score']:.3f} info={row['information_usage']:.3f} "
            f"risk={row['risk_response']:.3f}"
        )
    summary = summarize_rows(rows)
    print("-" * 112)
    print(
        f"{summary['task']:<24} policy={summary['policy']:<10} score={summary['score']:.3f} "
        f"ret={summary['total_return']:.3f} dd={summary['max_drawdown']:.3f} "
        f"comp={summary['compliance_score']:.3f} info={summary['information_usage']:.3f} "
        f"risk={summary['risk_response']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate committee policies.")
    parser.add_argument(
        "--policy",
        default="heuristic",
        choices=[
            "random",
            "heuristic",
            "llm",
            "always_cash",
            "concentrated_alpha",
            "query_spam",
        ],
        help="Policy to evaluate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a plain-text table.",
    )
    parser.add_argument(
        "--seeds",
        default="7,11,13",
        help="Comma-separated evaluation seeds. Default: 7,11,13.",
    )
    args = parser.parse_args()

    rows = evaluate_policy(args.policy, seeds=_parse_seeds(args.seeds))
    if args.json:
        print(json.dumps({"rows": rows, "summary": summarize_rows(rows)}, indent=2))
        return
    print_eval_table(rows)


if __name__ == "__main__":
    main()
