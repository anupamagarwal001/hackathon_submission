"""Minimal TRL training scaffold for the Round 2 committee environment.

This script is intentionally narrow:

- trains only the Portfolio Manager
- keeps Research and Risk scripted inside the environment
- uses TRL's `environment_factory` pattern so the PM interacts through tools
- is meant for Colab / GPU execution, not for this local macOS shell

Official references used for this scaffold:
- https://huggingface.co/docs/trl/openenv
- https://huggingface.co/docs/trl/grpo_trainer
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Optional training dependencies for Colab / GPU environments.
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
except ImportError:  # pragma: no cover
    Dataset = None
    GRPOConfig = None
    GRPOTrainer = None

try:  # Optional PEFT dependency for LoRA fine-tuning.
    from peft import LoraConfig
except ImportError:  # pragma: no cover
    LoraConfig = None

from graders import compute_metrics, grade_episode
from models import PortfolioAction
from server.amc_environment import ALLOCATION_TEMPLATE_DESCRIPTIONS, AmcAllocatorEnvironment
from tasks import DEFAULT_TASK_ID, TASK_ORDER
from training.committee_artifacts import (
    export_baseline_report,
    export_metric_series,
    export_judging_report,
    export_training_log_history,
    load_json,
    plot_metric_series,
)


TASK_PROMPTS = {
    "guided_allocation": (
        "You are the Portfolio Manager in an investment committee. Use the available tools to "
        "gather information and manage the portfolio while preserving capital discipline."
    ),
    "research_risk_conflict": (
        "You are the Portfolio Manager. Research may look attractive but risk conditions are "
        "fragile. Use tools carefully and avoid reckless concentration."
    ),
    "regime_shift_recovery": (
        "You are the Portfolio Manager. Detect changes in market conditions, de-risk when "
        "necessary, and recover without taking avoidable drawdowns."
    ),
    "mandate_drift": (
        "You are the Portfolio Manager. The mandate can tighten during the episode. Keep track "
        "of changing constraints and stay compliant while still searching for return."
    ),
}

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")


def _require_training_deps() -> None:
    if Dataset is None or GRPOConfig is None or GRPOTrainer is None:
        raise ImportError(
            "Training dependencies are not installed. Run this in Colab or a GPU environment "
            "with TRL and datasets available."
        )


def _require_peft() -> None:
    if LoraConfig is None:
        raise ImportError(
            "PEFT is not installed. Install `peft` in Colab before using --use-lora."
        )


def _format_weights(weights: Dict[str, float]) -> str:
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    if not ordered or max(weight for _, weight in ordered) <= 0.0:
        return "all cash"
    return ", ".join(f"{asset}:{weight:.2f}" for asset, weight in ordered if weight > 0.0)


class CommitteeToolEnv:
    """TRL tool environment wrapper around the committee OpenEnv environment."""

    def __init__(self) -> None:
        self.task_id = DEFAULT_TASK_ID
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.component_totals: Dict[str, float] = {}
        self.inner_env = AmcAllocatorEnvironment(task_id=self.task_id)
        self.last_observation = self.inner_env.reset(task_id=self.task_id)

    def reset(self, task_id: str = DEFAULT_TASK_ID, **kwargs) -> str | None:
        """Reset the environment for a new rollout."""

        self.task_id = task_id or DEFAULT_TASK_ID
        self.inner_env = AmcAllocatorEnvironment(task_id=self.task_id)
        self.last_observation = self.inner_env.reset(task_id=self.task_id)
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.component_totals = {}
        return self._render_observation(prefix="Episode reset")

    def query_research(self, target: str = "SECTOR") -> str:
        """
        Request a research update from the Research Analyst.

        Args:
            target: Asset ticker or `SECTOR` for a broad research view.

        Returns:
            Updated environment observation text.
        """

        return self._apply_action(
            PortfolioAction(
                action_type="query_research",
                query_target=target,
                reason=f"Research query for {target}.",
            )
        )

    def query_risk(self) -> str:
        """
        Request a risk review from the Risk Officer.

        Returns:
            Updated environment observation text.
        """

        return self._apply_action(
            PortfolioAction(action_type="query_risk", reason="Request risk review.")
        )

    def allocate(self, template: str) -> str:
        """
        Rebalance the portfolio using one of the supported allocation templates.

        Args:
            template: Template name such as `balanced_top3`, `top2_conviction`,
                `defensive_quality`, or `concentrated_alpha`.

        Returns:
            Updated environment observation text.
        """

        if template not in ALLOCATION_TEMPLATE_DESCRIPTIONS:
            raise ValueError(
                f"Unknown template {template!r}. Expected one of: "
                f"{', '.join(sorted(ALLOCATION_TEMPLATE_DESCRIPTIONS))}"
            )
        return self._apply_action(
            PortfolioAction(
                action_type="allocate",
                allocation_template=template,
                reason=f"Allocate using template {template}.",
            )
        )

    def move_to_cash(self) -> str:
        """
        Exit risky positions and move the portfolio into cash.

        Returns:
            Updated environment observation text.
        """

        return self._apply_action(
            PortfolioAction(action_type="move_to_cash", reason="Preserve capital.")
        )

    def hold(self) -> str:
        """
        Keep the current portfolio unchanged for this step.

        Returns:
            Updated environment observation text.
        """

        return self._apply_action(PortfolioAction(action_type="hold", reason="Hold position."))

    def _apply_action(self, action: PortfolioAction) -> str:
        if self.done:
            raise ValueError("Episode already finished. Stop calling tools.")

        self.last_observation = self.inner_env.step(action)
        self.last_reward = self.last_observation.reward
        self.total_reward += self.last_reward
        self.done = self.last_observation.done
        for name, value in self.last_observation.reward_components.items():
            self.component_totals[name] = self.component_totals.get(name, 0.0) + float(value)
        return self._render_observation(prefix=f"Action applied: {action.action_type}")

    def _render_observation(self, prefix: str) -> str:
        observation = self.last_observation
        risk_pressure = observation.risk_metrics.get("risk_pressure", 0.0)
        top_signals = sorted(
            observation.signals.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        top_signal_text = ", ".join(f"{asset}:{signal:+.2f}" for asset, signal in top_signals)
        latest_research = observation.research_notes[-1] if observation.research_notes else "none"
        latest_risk = observation.risk_notes[-1] if observation.risk_notes else "none"
        return (
            f"{prefix}\n"
            f"task={observation.task_id}\n"
            f"step={observation.step_index}/{observation.step_index + observation.steps_remaining}\n"
            f"cash={observation.cash_weight:.2f}\n"
            f"weights={_format_weights(observation.current_weights)}\n"
            f"top_signals={top_signal_text}\n"
            f"risk_pressure={risk_pressure:.2f}\n"
            f"constraints={json.dumps(observation.active_constraints, sort_keys=True)}\n"
            f"queries_remaining={observation.queries_remaining}\n"
            f"latest_research={latest_research}\n"
            f"latest_risk={latest_risk}\n"
            f"last_reward={observation.reward:.4f}\n"
            f"done={observation.done}"
        )


def _completion_ratio(env: CommitteeToolEnv) -> float:
    total_steps = max(env.inner_env.state.total_steps, 1)
    return min(1.0, env.inner_env.state.current_step / total_steps)


def _progress_weighted(env: CommitteeToolEnv, reward: float) -> float:
    return reward * _completion_ratio(env)


def task_score_reward(environments: List[CommitteeToolEnv], **kwargs) -> List[float]:
    """Primary environment reward based on the deterministic episode grader."""

    rewards: List[float] = []
    for env in environments:
        metrics = compute_metrics(env.inner_env.state)
        rewards.append(_progress_weighted(env, grade_episode(metrics)))
    return rewards


def compliance_reward(environments: List[CommitteeToolEnv], **kwargs) -> List[float]:
    """Reward cleaner mandate and compliance behavior."""

    rewards: List[float] = []
    for env in environments:
        metrics = compute_metrics(env.inner_env.state)
        rewards.append(_progress_weighted(env, 0.20 * metrics.compliance_score))
    return rewards


def risk_response_reward(environments: List[CommitteeToolEnv], **kwargs) -> List[float]:
    """Reward timely de-risking and adaptation under pressure."""

    rewards: List[float] = []
    for env in environments:
        metrics = compute_metrics(env.inner_env.state)
        rewards.append(_progress_weighted(env, 0.30 * max(0.0, metrics.risk_response)))
    return rewards


def information_usage_reward(environments: List[CommitteeToolEnv], **kwargs) -> List[float]:
    """Reward constructive use of research signals and committee information."""

    rewards: List[float] = []
    for env in environments:
        metrics = compute_metrics(env.inner_env.state)
        rewards.append(_progress_weighted(env, 0.25 * max(0.0, metrics.information_usage)))
    return rewards


def build_reward_functions():
    """Return the verifier-style reward function stack used for training."""

    return [
        task_score_reward,
        compliance_reward,
        risk_response_reward,
        information_usage_reward,
    ]


def build_training_dataset(repeats_per_task: int = 8) -> "Dataset":
    """Create a small prompt dataset for environment training."""

    _require_training_deps()
    prompts: list[list[dict[str, str]]] = []
    task_ids: list[str] = []
    for task_id in TASK_ORDER:
        for _ in range(repeats_per_task):
            prompts.append([{"role": "user", "content": TASK_PROMPTS[task_id]}])
            task_ids.append(task_id)
    return Dataset.from_dict({"prompt": prompts, "task_id": task_ids})


def build_grpo_config(output_dir: str, max_steps: int = 30) -> "GRPOConfig":
    """Return a compact GRPO config suitable for a Colab smoke run."""

    _require_training_deps()
    use_cpu = bool(torch is not None and not torch.cuda.is_available())
    generation_count = 2
    candidate_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": generation_count,
        "gradient_accumulation_steps": 1,
        "num_generations": generation_count,
        "max_prompt_length": 1024,
        "max_completion_length": 384,
        "max_steps": max_steps,
        "learning_rate": 1e-6,
        "logging_steps": 1,
        "log_completions": True,
        "report_to": "none",
        "bf16": False,
        "fp16": False,
        "use_cpu": use_cpu,
        "no_cuda": use_cpu,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    supported = inspect.signature(GRPOConfig.__init__).parameters
    filtered_kwargs = {
        key: value for key, value in candidate_kwargs.items() if key in supported
    }
    return GRPOConfig(**filtered_kwargs)


def build_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> "LoraConfig":
    """Return a compact LoRA configuration for Qwen-style PM fine-tuning."""

    _require_peft()
    candidate_kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules or list(DEFAULT_LORA_TARGET_MODULES),
        "inference_mode": False,
    }
    supported = inspect.signature(LoraConfig.__init__).parameters
    filtered_kwargs = {
        key: value for key, value in candidate_kwargs.items() if key in supported
    }
    return LoraConfig(**filtered_kwargs)


def build_trainer(
    model_name: str,
    output_dir: str,
    repeats_per_task: int = 8,
    max_steps: int = 30,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
) -> "GRPOTrainer":
    """Build the GRPO trainer for the committee environment."""

    _require_training_deps()
    dataset = build_training_dataset(repeats_per_task=repeats_per_task)
    return GRPOTrainer(
        model=model_name,
        args=build_grpo_config(output_dir=output_dir, max_steps=max_steps),
        train_dataset=dataset,
        reward_funcs=build_reward_functions(),
        environment_factory=CommitteeToolEnv,
        peft_config=(
            build_lora_config(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            if use_lora
            else None
        ),
    )


def baseline_summary() -> list[dict[str, float]]:
    """Return baseline metrics for quick before/after comparison."""

    from inference import run_episode

    rows: list[dict[str, float]] = []
    for task_id in TASK_ORDER:
        heuristic = run_episode(task_id, "heuristic", seed=7)
        random_baseline = run_episode(task_id, "random", seed=7)
        rows.append(
            {
                "task": task_id,
                "heuristic_score": round(heuristic.score, 4),
                "random_score": round(random_baseline.score, 4),
                "heuristic_return": round(heuristic.metrics.total_return, 4),
                "heuristic_drawdown": round(heuristic.metrics.max_drawdown, 4),
            }
        )
    return rows


def print_baseline_summary() -> None:
    """Print a compact baseline table for notebook users."""

    rows = baseline_summary()
    print("Baseline summary")
    for row in rows:
        print(
            f"{row['task']:<24} heuristic={row['heuristic_score']:.3f} "
            f"random={row['random_score']:.3f} return={row['heuristic_return']:.3f} "
            f"drawdown={row['heuristic_drawdown']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO scaffold for committee training.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model to fine-tune.")
    parser.add_argument(
        "--output-dir",
        default="outputs/committee-grpo",
        help="Training output directory.",
    )
    parser.add_argument(
        "--repeats-per-task",
        type=int,
        default=8,
        help="How many prompt rows to create per task.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Short smoke-run step count for hackathon iteration.",
    )
    parser.add_argument(
        "--print-baselines",
        action="store_true",
        help="Print heuristic vs random baselines before training.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Wrap the PM model with a compact LoRA adapter via PEFT.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank used when --use-lora is enabled.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha used when --use-lora is enabled.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout used when --use-lora is enabled.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated target modules for LoRA adapters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset/config only and skip trainer.train().",
    )
    parser.add_argument(
        "--colab-email",
        default=None,
        help="Optional Colab account email for the judging report artifact.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional note to embed into the judging report artifact.",
    )
    args = parser.parse_args()

    if args.print_baselines:
        print_baseline_summary()
        if not args.dry_run:
            return

    _require_training_deps()
    use_cpu = bool(torch is not None and not torch.cuda.is_available())
    if use_cpu and not args.dry_run:
        print(
            "Warning: CPU runtime detected. Full GRPO training will be slow. "
            "Prefer a GPU runtime and enable --use-lora."
        )
    trainer = build_trainer(
        model_name=args.model,
        output_dir=args.output_dir,
        repeats_per_task=args.repeats_per_task,
        max_steps=args.max_steps,
        use_lora=args.use_lora,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[
            module.strip()
            for module in args.lora_target_modules.split(",")
            if module.strip()
        ],
    )
    baseline_report_path = export_baseline_report(args.output_dir)
    baseline_payload = load_json(baseline_report_path)

    if args.dry_run:
        judging_paths = None
        if args.colab_email:
            judging_paths = export_judging_report(
                args.output_dir,
                model_name=args.model,
                colab_account_email=args.colab_email,
                baseline_payload=baseline_payload,
                notes=args.notes,
            )
        dataset_size = len(trainer.train_dataset) if trainer.train_dataset is not None else 0
        print(
            "Dry run complete. "
            f"trainer={type(trainer).__name__} "
            f"dataset_rows={dataset_size} "
            f"reward_funcs={len(build_reward_functions())} "
            f"baseline_report={baseline_report_path}"
            + (
                f" judging_report={judging_paths[0]} judging_markdown={judging_paths[1]}"
                if judging_paths
                else ""
            )
        )
        return

    trainer.train()
    trainer.save_model(args.output_dir)
    training_log_path = export_training_log_history(args.output_dir, trainer.state.log_history)
    metric_series_path = export_metric_series(args.output_dir, trainer.state.log_history)
    metric_plot_path = None
    try:
        metric_plot_path = plot_metric_series(trainer.state.log_history, args.output_dir)
    except ImportError:
        metric_plot_path = None
    judging_paths = None
    if args.colab_email:
        judging_paths = export_judging_report(
            args.output_dir,
            model_name=args.model,
            colab_account_email=args.colab_email,
            baseline_payload=baseline_payload,
            log_history=trainer.state.log_history,
            notes=args.notes,
        )
    print(
        "Training complete. "
        f"model_dir={args.output_dir} "
        f"training_log={training_log_path} "
        f"metric_series={metric_series_path} "
        f"baseline_report={baseline_report_path}"
        + (f" metric_plot={metric_plot_path}" if metric_plot_path else "")
        + (
            f" judging_report={judging_paths[0]} judging_markdown={judging_paths[1]}"
            if judging_paths
            else ""
        )
    )


if __name__ == "__main__":
    main()
