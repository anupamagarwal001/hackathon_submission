from __future__ import annotations

from inference import emit_structured_stdout, run_episode


def test_scores_are_bounded_for_all_tasks() -> None:
    for task_id in ("signal_following", "noisy_market", "regime_shift"):
        report = run_episode(task_id, "heuristic", seed=7)
        assert 0.0 <= report.score <= 1.0


def test_heuristic_beats_random_on_all_tasks() -> None:
    for task_id in ("signal_following", "noisy_market", "regime_shift"):
        heuristic_report = run_episode(task_id, "heuristic", seed=7)
        random_report = run_episode(task_id, "random", seed=7)
        assert heuristic_report.score > random_report.score


def test_run_episode_emits_required_structured_events() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    report = run_episode(
        "signal_following",
        "heuristic",
        seed=7,
        event_callback=lambda marker, payload: events.append((marker, payload)),
    )

    assert events[0][0] == "START"
    assert events[-1][0] == "END"
    step_events = [payload for marker, payload in events if marker == "STEP"]
    assert len(step_events) == report.metrics.steps
    assert all("reward" in payload for payload in step_events)


def test_emit_structured_stdout_uses_required_markers(capsys) -> None:
    emit_structured_stdout("END", task="signal_following", score=0.95, steps=30)
    output = capsys.readouterr().out.strip()
    assert output.startswith("[END] ")
    assert "task=signal_following" in output
    assert "score=0.950000" in output
    assert "steps=30" in output


def test_run_episode_structured_events_match_expected_fields() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    run_episode(
        "signal_following",
        "heuristic",
        seed=7,
        event_callback=lambda marker, payload: events.append((marker, payload)),
    )

    assert events[0] == ("START", {"task": "signal_following"})
    assert events[1][0] == "STEP"
    assert list(events[1][1].keys()) == ["step", "reward"]
    assert events[-1][0] == "END"
    assert list(events[-1][1].keys()) == ["task", "score", "steps"]
