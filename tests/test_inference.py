from __future__ import annotations

from inference import run_episode


def test_scores_are_bounded_for_all_tasks() -> None:
    for task_id in ("signal_following", "noisy_market", "regime_shift"):
        report = run_episode(task_id, "heuristic", seed=7)
        assert 0.0 <= report.score <= 1.0


def test_heuristic_beats_random_on_all_tasks() -> None:
    for task_id in ("signal_following", "noisy_market", "regime_shift"):
        heuristic_report = run_episode(task_id, "heuristic", seed=7)
        random_report = run_episode(task_id, "random", seed=7)
        assert heuristic_report.score > random_report.score
