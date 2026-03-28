"""Baseline inference runner for the AMC allocator environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable

try:
    from graders import EpisodeMetrics, compute_metrics, grade_episode
    from policies import build_policy
    from server.amc_environment import AmcAllocatorEnvironment
    from tasks import DEFAULT_TASK_ID, TASK_ORDER
except ImportError:  # pragma: no cover
    from .graders import EpisodeMetrics, compute_metrics, grade_episode
    from .policies import build_policy
    from .server.amc_environment import AmcAllocatorEnvironment
    from .tasks import DEFAULT_TASK_ID, TASK_ORDER


@dataclass(frozen=True)
class EpisodeReport:
    task_id: str
    policy: str
    score: float
    metrics: EpisodeMetrics

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


def run_episode(task_id: str, policy_name: str, seed: int = 7) -> EpisodeReport:
    env = AmcAllocatorEnvironment(task_id=task_id)
    observation = env.reset(seed=seed)
    policy = build_policy(policy_name, seed=seed)

    while not observation.done:
        action = policy(observation)
        observation = env.step(action)

    metrics = compute_metrics(env.state)
    score = grade_episode(metrics)
    return EpisodeReport(task_id=task_id, policy=policy_name, score=score, metrics=metrics)


def run_benchmark(
    policy_name: str, tasks: Iterable[str] | None = None, seed: int = 7
) -> list[EpisodeReport]:
    selected_tasks = list(tasks or TASK_ORDER)
    if policy_name == "all":
        reports: list[EpisodeReport] = []
        for name in ("random", "heuristic", "llm"):
            reports.extend(run_benchmark(name, tasks=selected_tasks, seed=seed))
        return reports
    return [run_episode(task_id, policy_name, seed=seed) for task_id in selected_tasks]


def _print_reports(reports: list[EpisodeReport], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
        return

    grouped: dict[str, list[EpisodeReport]] = {}
    for report in reports:
        grouped.setdefault(report.policy, []).append(report)

    for policy_name, policy_reports in grouped.items():
        print(f"policy={policy_name}")
        total_score = 0.0
        for report in policy_reports:
            total_score += report.score
            print(
                f"  task={report.task_id:<16} "
                f"score={report.score:.3f} "
                f"return={report.metrics.total_return:.3%} "
                f"drawdown={report.metrics.max_drawdown:.3%} "
                f"turnover={report.metrics.average_turnover:.3f}"
            )
        print(f"  aggregate_score={total_score / len(policy_reports):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run allocator baseline inference.")
    parser.add_argument(
        "--policy",
        default="heuristic",
        choices=["heuristic", "random", "llm", "all"],
        help="Policy to evaluate. Default is heuristic for reproducible local runs.",
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", *TASK_ORDER],
        help="Optional single task to run. Default runs all tasks.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random seed.")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of text."
    )
    args = parser.parse_args()

    selected_tasks = TASK_ORDER if args.task == "all" else (args.task,)
    reports = run_benchmark(args.policy, tasks=selected_tasks, seed=args.seed)
    _print_reports(reports, as_json=args.json)


if __name__ == "__main__":
    main()
