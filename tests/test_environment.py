from __future__ import annotations

from server.amc_environment import AmcAllocatorEnvironment
from models import PortfolioAction


def test_reset_exposes_allocator_observation_shape() -> None:
    env = AmcAllocatorEnvironment(task_id="signal_following")
    observation = env.reset()

    assert observation.task_id == "signal_following"
    assert observation.step_index == 0
    assert observation.steps_remaining == 30
    assert len(observation.prices) == 5
    assert observation.cash_weight == 1.0


def test_step_normalizes_invalid_weights() -> None:
    env = AmcAllocatorEnvironment(task_id="signal_following")
    env.reset()
    observation = env.step(
        PortfolioAction(
            target_weights={"INFY": 0.9, "TCS": 0.9, "WIPRO": -0.2},
            reason="force normalization",
        )
    )

    assert round(sum(observation.current_weights.values()) + observation.cash_weight, 6) == 1.0
    assert observation.current_weights["WIPRO"] == 0.0
    assert observation.turnover > 0.0


def test_state_tracks_nav_and_history() -> None:
    env = AmcAllocatorEnvironment(task_id="noisy_market")
    observation = env.reset()
    for _ in range(5):
        observation = env.step(PortfolioAction(target_weights={"INFY": 0.4, "TCS": 0.3}))

    state = env.state
    assert state.current_step == 5
    assert len(state.nav_history) == 6
    assert len(state.reward_history) == 5
    assert state.portfolio_value == observation.portfolio_value
