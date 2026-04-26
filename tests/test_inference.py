from __future__ import annotations

from importlib import reload

import inference
from inference import emit_structured_stdout, run_episode
from policies import LLMAllocatorPolicy, heuristic_policy
from server.amc_environment import AmcAllocatorEnvironment


TASK_IDS = (
    "guided_allocation",
    "research_risk_conflict",
    "regime_shift_recovery",
    "mandate_drift",
)


def test_scores_are_bounded_for_all_tasks() -> None:
    for task_id in TASK_IDS:
        report = run_episode(task_id, "heuristic", seed=7)
        assert 0.0 <= report.score <= 1.0


def test_heuristic_beats_random_on_all_tasks() -> None:
    for task_id in TASK_IDS:
        heuristic_report = run_episode(task_id, "heuristic", seed=7)
        random_report = run_episode(task_id, "random", seed=7)
        assert heuristic_report.score > random_report.score


def test_reward_hacking_probes_do_not_beat_heuristic() -> None:
    heuristic_report = run_episode("research_risk_conflict", "heuristic", seed=7)
    for policy_name in ("always_cash", "concentrated_alpha", "query_spam"):
        probe_report = run_episode("research_risk_conflict", policy_name, seed=7)
        assert probe_report.score < heuristic_report.score


def test_run_episode_emits_required_structured_events() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    report = run_episode(
        "guided_allocation",
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
    emit_structured_stdout("END", task="guided_allocation", score=0.95, steps=30)
    output = capsys.readouterr().out.strip()
    assert output.startswith("[END] ")
    assert "task=guided_allocation" in output
    assert "score=0.950000" in output
    assert "steps=30" in output


def test_run_episode_structured_events_match_expected_fields() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    run_episode(
        "guided_allocation",
        "heuristic",
        seed=7,
        event_callback=lambda marker, payload: events.append((marker, payload)),
    )

    assert events[0] == ("START", {"task": "guided_allocation"})
    assert events[1][0] == "STEP"
    assert list(events[1][1].keys()) == ["step", "reward"]
    assert events[-1][0] == "END"
    assert list(events[-1][1].keys()) == ["task", "score", "steps"]


def test_default_policy_prefers_llm_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-key")
    reloaded = reload(inference)
    assert reloaded._resolve_policy_name(None) == "llm"


def test_default_policy_falls_back_to_heuristic_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    reloaded = reload(inference)
    assert reloaded._resolve_policy_name(None) == "heuristic"


def test_llm_policy_falls_back_to_heuristic_on_client_error() -> None:
    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object) -> object:
            self.calls += 1
            raise RuntimeError("proxy unavailable")

    class FailingChat:
        def __init__(self) -> None:
            self.completions = FailingCompletions()

    class FailingClient:
        def __init__(self) -> None:
            self.chat = FailingChat()

    env = AmcAllocatorEnvironment(task_id="guided_allocation")
    observation = env.reset(seed=7)
    policy = LLMAllocatorPolicy(client=FailingClient(), model_name="test-model")

    first_action = policy(observation)
    expected = heuristic_policy(observation)
    assert first_action.action_type == expected.action_type
    assert first_action.allocation_template == expected.allocation_template
    assert first_action.target_weights == expected.target_weights
    assert "LLM fallback activated" in (first_action.reason or "")
    assert policy.client.chat.completions.calls == 1

    second_action = policy(observation)
    assert second_action.action_type == expected.action_type
    assert policy.client.chat.completions.calls == 1
