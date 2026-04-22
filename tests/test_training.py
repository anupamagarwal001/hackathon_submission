from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tasks import TASK_ORDER
from training.committee_artifacts import (
    build_metric_series,
    export_judging_report,
    export_baseline_report,
    export_onsite_demo_summary,
    export_training_log_history,
    load_training_log_history,
    summarize_metric_progress,
)
from training.committee_eval import evaluate_policy, summarize_rows
from training.committee_grpo_train import (
    CommitteeToolEnv,
    build_reward_functions,
)


def test_committee_tool_env_reset_renders_committee_context() -> None:
    env = CommitteeToolEnv()
    rendered = env.reset(task_id="guided_allocation")

    assert rendered is not None
    assert "task=guided_allocation" in rendered
    assert "queries_remaining=" in rendered
    assert "latest_research=none" in rendered


def test_committee_tool_env_query_research_updates_notes() -> None:
    env = CommitteeToolEnv()
    env.reset(task_id="guided_allocation")

    rendered = env.query_research("SECTOR")

    assert "Action applied: query_research" in rendered
    assert env.inner_env.state.query_count == 1
    assert env.last_observation.research_notes


def test_training_rewards_do_not_pay_for_zero_progress() -> None:
    env = CommitteeToolEnv()
    env.reset(task_id="guided_allocation")

    rewards = [reward_func([env])[0] for reward_func in build_reward_functions()]

    assert rewards == [0.0, 0.0, 0.0, 0.0]


def test_evaluate_policy_and_summary_cover_all_tasks() -> None:
    rows = evaluate_policy("heuristic", seeds=(7,))
    summary = summarize_rows(rows)

    assert len(rows) == len(TASK_ORDER)
    assert all(row["task"] in TASK_ORDER for row in rows)
    assert summary["task"] == "overall"
    assert summary["policy"] == "heuristic"
    assert summary["score"] > 0.0


def test_artifact_helpers_export_and_reload_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        baseline_path = export_baseline_report(output_dir, seeds=(7,))
        assert baseline_path.exists()
        assert baseline_path.name == "baseline_report.json"

        log_history = [
            {"step": 1, "reward": 0.1, "loss": 0.9},
            {"step": 2, "reward": 0.2, "loss": 0.7},
        ]
        log_path = export_training_log_history(output_dir, log_history)
        loaded = load_training_log_history(log_path)
        series = build_metric_series(loaded, metric_key="reward")

        assert loaded == log_history
        assert series["metric_key"] == "reward"
        assert series["points"] == [
            {"step": 1, "value": 0.1},
            {"step": 2, "value": 0.2},
        ]


def test_judging_report_exports_json_and_markdown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        baseline_path = export_baseline_report(output_dir, seeds=(7,))
        baseline_payload = json.loads(baseline_path.read_text())
        log_history = [
            {"step": 1, "reward": 0.1},
            {"step": 2, "reward": 0.4},
        ]

        json_path, markdown_path = export_judging_report(
            output_dir,
            model_name="Qwen/Qwen3-0.6B",
            colab_account_email="anuagar@groww.in",
            baseline_payload=baseline_payload,
            log_history=log_history,
            notes="Smoke test report.",
        )

        json_text = json_path.read_text()
        markdown_text = markdown_path.read_text()
        assert json_path.exists()
        assert markdown_path.exists()
        assert "anuagar@groww.in" in json_text
        assert "Qwen/Qwen3-0.6B" in markdown_text
        assert "Smoke test report." in markdown_text


def test_summarize_metric_progress_tracks_best_step_and_delta() -> None:
    log_history = [
        {"step": 1, "reward": 0.1},
        {"step": 2, "reward": 0.05},
        {"step": 3, "reward": 0.4},
        {"step": 4, "reward": 0.2},
    ]

    summary = summarize_metric_progress(log_history, metric_key="reward")

    assert summary["metric_key"] == "reward"
    assert summary["start"] == 0.1
    assert summary["end"] == 0.2
    assert summary["best"] == 0.4
    assert summary["best_step"] == 3
    assert summary["delta"] == 0.1
    assert summary["best_delta"] == 0.3


def test_onsite_demo_summary_exports_markdown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        baseline_path = export_baseline_report(output_dir, seeds=(7,))
        baseline_payload = json.loads(baseline_path.read_text())
        log_history = [
            {"step": 1, "reward": 0.1},
            {"step": 2, "reward": 0.4},
        ]

        markdown_path = export_onsite_demo_summary(
            output_dir,
            model_name="Qwen/Qwen3-0.6B",
            colab_account_email="anuagar@groww.in",
            baseline_payload=baseline_payload,
            log_history=log_history,
            notes="Use the curve and the task ladder in the demo.",
        )

        text = markdown_path.read_text()
        assert markdown_path.exists()
        assert "Onsite Demo Summary" in text
        assert "Best step: 2" in text
        assert "Use the curve and the task ladder in the demo." in text
