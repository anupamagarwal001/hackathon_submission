"""Baseline inference runner for the AMC allocator environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Callable
from typing import Iterable

from openai import OpenAI

try:
    from graders import EpisodeMetrics, compute_metrics, grade_episode
    from policies import LLMAllocatorPolicy, build_policy
    from server.amc_environment import AmcAllocatorEnvironment
    from tasks import DEFAULT_TASK_ID, TASK_ORDER
except ImportError:  # pragma: no cover
    from .graders import EpisodeMetrics, compute_metrics, grade_episode
    from .policies import LLMAllocatorPolicy, build_policy
    from .server.amc_environment import AmcAllocatorEnvironment
    from .tasks import DEFAULT_TASK_ID, TASK_ORDER


# Mandatory evaluator-facing configuration.
# Defaults are set only for API_BASE_URL and MODEL_NAME, not HF_TOKEN.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional if a local docker image workflow is used by the runner.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


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


def _format_structured_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def emit_structured_stdout(marker: str, **payload: object) -> None:
    ordered = " ".join(
        f"{key}={_format_structured_value(value)}" for key, value in payload.items()
    )
    print(f"[{marker}] {ordered}".rstrip(), flush=True)


def _validator_api_key() -> str | None:
    return os.getenv("API_KEY")


def _build_runtime_policy(policy_name: str, seed: int = 7):
    normalized = policy_name.lower()
    if normalized != "llm":
        return build_policy(normalized, seed=seed)

    validator_api_key = _validator_api_key()
    if validator_api_key:
        client = OpenAI(
            api_key=validator_api_key,
            base_url=os.getenv("API_BASE_URL", API_BASE_URL),
        )
        return LLMAllocatorPolicy(client=client, model_name=MODEL_NAME)

    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set API_KEY, HF_TOKEN, or OPENAI_API_KEY before running --policy llm")

    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    return LLMAllocatorPolicy(client=client, model_name=MODEL_NAME)


def _resolve_policy_name(requested_policy: str | None) -> str:
    if requested_policy:
        return requested_policy
    if _validator_api_key():
        return "llm"
    return "heuristic"


def run_episode(
    task_id: str,
    policy_name: str,
    seed: int = 7,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> EpisodeReport:
    env = AmcAllocatorEnvironment(task_id=task_id)
    observation = env.reset(seed=seed)
    policy = _build_runtime_policy(policy_name, seed=seed)
    if event_callback is not None:
        event_callback(
            "START",
            {
                "task": task_id,
            },
        )

    while not observation.done:
        action = policy(observation)
        observation = env.step(action)
        if event_callback is not None:
            event_callback(
                "STEP",
                {
                    "step": observation.step_index,
                    "reward": observation.reward,
                },
            )

    metrics = compute_metrics(env.state)
    score = grade_episode(metrics)
    report = EpisodeReport(task_id=task_id, policy=policy_name, score=score, metrics=metrics)
    if event_callback is not None:
        event_callback(
            "END",
            {
                "task": task_id,
                "score": score,
                "steps": metrics.steps,
            },
        )
    return report


def run_benchmark(
    policy_name: str,
    tasks: Iterable[str] | None = None,
    seed: int = 7,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> list[EpisodeReport]:
    selected_tasks = list(tasks or TASK_ORDER)
    if policy_name == "all":
        reports: list[EpisodeReport] = []
        for name in ("random", "heuristic", "llm"):
            reports.extend(
                run_benchmark(
                    name,
                    tasks=selected_tasks,
                    seed=seed,
                    event_callback=event_callback,
                )
            )
        return reports
    return [
        run_episode(task_id, policy_name, seed=seed, event_callback=event_callback)
        for task_id in selected_tasks
    ]


def _print_pretty_reports(reports: list[EpisodeReport]) -> None:
    grouped: dict[str, list[EpisodeReport]] = {}
    for report in reports:
        grouped.setdefault(report.policy, []).append(report)

    for policy_name, policy_reports in grouped.items():
        print(f"policy={policy_name}", file=sys.stderr)
        total_score = 0.0
        for report in policy_reports:
            total_score += report.score
            print(
                f"  task={report.task_id:<16} "
                f"score={report.score:.3f} "
                f"return={report.metrics.total_return:.3%} "
                f"drawdown={report.metrics.max_drawdown:.3%} "
                f"turnover={report.metrics.average_turnover:.3f}",
                file=sys.stderr,
            )
        print(f"  aggregate_score={total_score / len(policy_reports):.3f}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run allocator baseline inference.")
    parser.add_argument(
        "--policy",
        default=None,
        choices=["heuristic", "random", "llm", "all"],
        help="Policy to evaluate. Defaults to llm when API_KEY is injected, otherwise heuristic.",
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
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also print the human-readable summary to stderr.",
    )
    args = parser.parse_args()

    selected_policy = _resolve_policy_name(args.policy)
    selected_tasks = TASK_ORDER if args.task == "all" else (args.task,)
    if args.json:
        reports = run_benchmark(selected_policy, tasks=selected_tasks, seed=args.seed)
        print(json.dumps([report.to_dict() for report in reports], indent=2))
        return

    reports = run_benchmark(
        selected_policy,
        tasks=selected_tasks,
        seed=args.seed,
        event_callback=lambda marker, payload: emit_structured_stdout(marker, **payload),
    )
    if args.pretty:
        _print_pretty_reports(reports)


if __name__ == "__main__":
    main()
