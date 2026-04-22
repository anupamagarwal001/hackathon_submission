from __future__ import annotations

from models import PortfolioAction
from server.amc_environment import AmcAllocatorEnvironment


def test_reset_exposes_committee_observation_shape() -> None:
    env = AmcAllocatorEnvironment(task_id="guided_allocation")
    observation = env.reset()

    assert observation.task_id == "guided_allocation"
    assert observation.step_index == 0
    assert observation.steps_remaining == 30
    assert len(observation.prices) == 5
    assert observation.cash_weight == 1.0
    assert "balanced_top3" in observation.available_allocation_templates
    assert observation.queries_remaining > 0


def test_query_research_adds_notes_without_rebalancing() -> None:
    env = AmcAllocatorEnvironment(task_id="guided_allocation")
    env.reset()
    observation = env.step(
        PortfolioAction(
            action_type="query_research",
            query_target="SECTOR",
            reason="Need initial analyst view.",
        )
    )

    assert observation.current_weights == {asset: 0.0 for asset in env.state.assets}
    assert observation.cash_weight == 1.0
    assert observation.research_notes
    assert observation.queries_remaining == env._scenario.max_queries - 1  # noqa: SLF001


def test_template_allocation_respects_constraints() -> None:
    env = AmcAllocatorEnvironment(task_id="mandate_drift")
    env.reset()
    observation = env.step(
        PortfolioAction(
            action_type="allocate",
            allocation_template="concentrated_alpha",
            reason="Test concentration handling.",
        )
    )

    max_single = observation.active_constraints["max_single_asset_weight"]
    assert max(observation.current_weights.values(), default=0.0) <= max_single + 1e-9
    assert round(sum(observation.current_weights.values()) + observation.cash_weight, 6) == 1.0


def test_state_tracks_committee_histories() -> None:
    env = AmcAllocatorEnvironment(task_id="research_risk_conflict")
    observation = env.reset()
    actions = [
        PortfolioAction(action_type="query_research", query_target="SECTOR"),
        PortfolioAction(action_type="query_risk"),
        PortfolioAction(action_type="allocate", allocation_template="balanced_top3"),
    ]
    for action in actions:
        observation = env.step(action)

    state = env.state
    assert state.current_step == 3
    assert len(state.nav_history) == 4
    assert len(state.reward_history) == 3
    assert len(state.action_history) == 3
    assert len(state.reward_component_history) == 3
    assert state.portfolio_value == observation.portfolio_value
