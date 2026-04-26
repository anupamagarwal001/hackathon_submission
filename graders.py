"""Deterministic graders for committee-style episodes."""

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
    information_usage: float
    risk_response: float
    positive_step_ratio: float
    volatility: float
    invested_ratio: float
    compliance_score: float
    steps: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(state) -> EpisodeMetrics:
    """Build deterministic episode metrics from the final environment state."""

    nav_history = list(state.nav_history)
    reward_history = list(state.reward_history)
    turnover_history = list(state.turnover_history)
    signal_alignment_history = list(state.signal_alignment_history)
    info_usage_history = list(state.information_usage_history)
    risk_response_history = list(state.risk_response_history)
    return_history = list(state.portfolio_return_history)

    final_nav = nav_history[-1] if nav_history else float(state.portfolio_value)
    total_return = final_nav - 1.0
    max_drawdown = float(state.risk_metrics.get("max_drawdown", 0.0))
    average_turnover = mean(turnover_history) if turnover_history else 0.0
    signal_alignment = mean(signal_alignment_history) if signal_alignment_history else 0.0
    information_usage = mean(info_usage_history) if info_usage_history else 0.0
    risk_response = mean(risk_response_history) if risk_response_history else 0.0
    positive_step_ratio = (
        sum(1 for value in reward_history if value > 0.0) / len(reward_history)
        if reward_history
        else 0.0
    )
    volatility = mean(abs(value) for value in return_history) if return_history else 0.0
    invested_ratio = 1.0 - float(state.cash_weight)
    compliance_score = 1.0
    if state.current_step:
        compliance_score = _clamp(1.0 - (state.compliance_violations / state.current_step))

    return EpisodeMetrics(
        task_id=state.task_id,
        final_nav=final_nav,
        total_return=total_return,
        reward_sum=sum(reward_history),
        max_drawdown=max_drawdown,
        average_turnover=average_turnover,
        signal_alignment=signal_alignment,
        information_usage=information_usage,
        risk_response=risk_response,
        positive_step_ratio=positive_step_ratio,
        volatility=volatility,
        invested_ratio=invested_ratio,
        compliance_score=compliance_score,
        steps=state.current_step,
    )


def score_guided_allocation(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.40 * _normalize(metrics.total_return, 0.0, 0.12)
        + 0.20 * _normalize(metrics.information_usage, 0.05, 0.55)
        + 0.15 * _normalize(metrics.positive_step_ratio, 0.40, 0.80)
        + 0.15 * _normalize(metrics.compliance_score, 0.70, 1.0)
        + 0.10 * _normalize(0.35 - metrics.average_turnover, 0.0, 0.35)
    )


def score_research_risk_conflict(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.30 * _normalize(metrics.total_return, -0.01, 0.09)
        + 0.25 * _normalize(metrics.compliance_score, 0.65, 1.0)
        + 0.20 * _normalize(metrics.risk_response, 0.0, 0.18)
        + 0.15 * _normalize(0.12 - metrics.max_drawdown, 0.0, 0.12)
        + 0.10 * _normalize(metrics.information_usage, 0.0, 0.35)
    )


def score_regime_shift_recovery(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.33 * _normalize(metrics.total_return, -0.01, 0.12)
        + 0.27 * _normalize(metrics.risk_response, 0.0, 0.22)
        + 0.20 * _normalize(0.15 - metrics.max_drawdown, 0.0, 0.15)
        + 0.10 * _normalize(metrics.positive_step_ratio, 0.35, 0.75)
        + 0.10 * _normalize(metrics.compliance_score, 0.70, 1.0)
    )


def score_mandate_drift(metrics: EpisodeMetrics) -> float:
    return _clamp(
        0.30 * _normalize(metrics.compliance_score, 0.60, 1.0)
        + 0.25 * _normalize(0.12 - metrics.max_drawdown, 0.0, 0.12)
        + 0.20 * _normalize(metrics.total_return, -0.01, 0.09)
        + 0.15 * _normalize(metrics.information_usage, 0.0, 0.30)
        + 0.10 * _normalize(0.30 - metrics.average_turnover, 0.0, 0.30)
    )


TASK_GRADERS = {
    "guided_allocation": score_guided_allocation,
    "research_risk_conflict": score_research_risk_conflict,
    "regime_shift_recovery": score_regime_shift_recovery,
    "mandate_drift": score_mandate_drift,
}


def apply_reward_hacking_guardrails(metrics: EpisodeMetrics, score: float) -> float:
    """Cap scores for common degenerate strategies.

    These guardrails keep conservative no-op behavior from winning purely through
    low drawdown/compliance, while still leaving the component metrics visible.
    """

    adjusted = score
    if metrics.invested_ratio < 0.05 and metrics.information_usage < 0.02:
        adjusted = min(adjusted, 0.20)
    if metrics.compliance_score < 0.20:
        adjusted = min(adjusted, 0.35)
    if (
        metrics.information_usage < 0.02
        and metrics.risk_response < 0.02
        and metrics.total_return <= 0.001
    ):
        adjusted = min(adjusted, 0.25)
    return _clamp(adjusted)


def grade_episode(metrics: EpisodeMetrics) -> float:
    """Score an episode on the hackathon's required 0.0–1.0 scale."""

    return apply_reward_hacking_guardrails(metrics, TASK_GRADERS[metrics.task_id](metrics))
