"""Deterministic graders for allocator episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return _clamp((value - minimum) / (maximum - minimum))


@dataclass(frozen=True)
class EpisodeMetrics:
    """Episode-level metrics consumed by graders and reporting."""

    task_id: str
    final_nav: float
    total_return: float
    reward_sum: float
    max_drawdown: float
    average_turnover: float
    signal_alignment: float
    positive_step_ratio: float
    volatility: float
    invested_ratio: float
    steps: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(state) -> EpisodeMetrics:
    """Build deterministic episode metrics from the final environment state."""

    nav_history = list(state.nav_history)
    reward_history = list(state.reward_history)
    turnover_history = list(state.turnover_history)
    signal_alignment_history = list(state.signal_alignment_history)
    return_history = list(state.portfolio_return_history)

    final_nav = nav_history[-1] if nav_history else float(state.portfolio_value)
    total_return = final_nav - 1.0
    max_drawdown = float(state.risk_metrics.get("max_drawdown", 0.0))
    average_turnover = mean(turnover_history) if turnover_history else 0.0
    signal_alignment = mean(signal_alignment_history) if signal_alignment_history else 0.0
    positive_step_ratio = (
        sum(1 for value in reward_history if value > 0.0) / len(reward_history)
        if reward_history
        else 0.0
    )
    volatility = mean(abs(value) for value in return_history) if return_history else 0.0
    invested_ratio = 1.0 - mean(
        [1.0 - sum(weights.values()) for weights in [state.current_weights]]
    )
    if state.current_weights:
        invested_ratio = 1.0 - float(state.cash_weight)

    return EpisodeMetrics(
        task_id=state.task_id,
        final_nav=final_nav,
        total_return=total_return,
        reward_sum=sum(reward_history),
        max_drawdown=max_drawdown,
        average_turnover=average_turnover,
        signal_alignment=signal_alignment,
        positive_step_ratio=positive_step_ratio,
        volatility=volatility,
        invested_ratio=invested_ratio,
        steps=state.current_step,
    )


def score_signal_following(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.50 * _normalize(metrics.total_return, 0.0, 0.12)
        + 0.25 * _normalize(metrics.signal_alignment, 0.25, 0.75)
        + 0.15 * _normalize(metrics.positive_step_ratio, 0.40, 0.80)
        + 0.10 * _normalize(0.35 - metrics.average_turnover, 0.0, 0.35)
    )


def score_noisy_market(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.42 * _normalize(metrics.total_return, -0.01, 0.10)
        + 0.25 * _normalize(0.18 - metrics.average_turnover, 0.0, 0.18)
        + 0.18 * _normalize(0.10 - metrics.max_drawdown, 0.0, 0.10)
        + 0.15 * _normalize(metrics.signal_alignment, 0.10, 0.55)
    )


def score_regime_shift(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.38 * _normalize(metrics.total_return, -0.02, 0.10)
        + 0.27 * _normalize(0.12 - metrics.max_drawdown, 0.0, 0.12)
        + 0.20 * _normalize(metrics.signal_alignment, 0.05, 0.45)
        + 0.15 * _normalize(metrics.positive_step_ratio, 0.35, 0.70)
    )


TASK_GRADERS = {
    "signal_following": score_signal_following,
    "noisy_market": score_noisy_market,
    "regime_shift": score_regime_shift,
}


def grade_episode(metrics: EpisodeMetrics) -> float:
    """Score an episode on the hackathon's required 0.0–1.0 scale."""

    return TASK_GRADERS[metrics.task_id](metrics)
