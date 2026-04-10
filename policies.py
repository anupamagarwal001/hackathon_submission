"""Baseline policies for allocator evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

try:
    from .models import AllocatorObservation, PortfolioAction
except ImportError:  # pragma: no cover
    from models import AllocatorObservation, PortfolioAction


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalize_weights(
    scores: dict[str, float], cash_buffer: float = 0.0
) -> dict[str, float]:
    positive_scores = {asset: max(0.0, score) for asset, score in scores.items()}
    total_score = sum(positive_scores.values())
    if total_score <= 0.0:
        return {}
    investable = 1.0 - _clamp(cash_buffer, 0.0, 0.8)
    return {
        asset: investable * score / total_score
        for asset, score in positive_scores.items()
        if score > 0.0
    }


def heuristic_policy(observation: AllocatorObservation) -> PortfolioAction:
    """Simple risk-aware signal following policy."""

    market_signal = observation.risk_metrics.get("market_signal", 0.0)
    dispersion = observation.risk_metrics.get("signal_dispersion", 0.0)
    drawdown = observation.risk_metrics.get("max_drawdown", 0.0)
    base_cash_buffer = observation.risk_metrics.get("cash_buffer_hint", 0.05)

    scores = {}
    for asset, signal in observation.signals.items():
        current_weight = observation.current_weights.get(asset, 0.0)
        score = max(0.0, signal) ** 1.6
        if observation.task_id == "noisy_market":
            score = 0.75 * score + 0.25 * current_weight
        elif observation.task_id == "regime_shift":
            score = 0.70 * score + 0.30 * current_weight
        scores[asset] = score

    if max(scores.values(), default=0.0) < 0.05 or market_signal < -0.18:
        return PortfolioAction(target_weights={}, reason="Signals weak; stay defensive in cash.")

    cash_buffer = base_cash_buffer
    cash_buffer += max(0.0, -market_signal) * 0.25
    cash_buffer += max(0.0, drawdown - 0.03) * 1.5
    cash_buffer += max(0.0, 0.18 - dispersion) * 0.10

    weights = _normalize_weights(scores, cash_buffer=cash_buffer)
    return PortfolioAction(
        target_weights=weights,
        reason="Risk-aware heuristic allocation from positive signals.",
    )


def random_policy(
    observation: AllocatorObservation, rng: random.Random | None = None
) -> PortfolioAction:
    """Random baseline with explicit cash in the draw."""

    generator = rng or random.Random()
    sample_count = len(observation.signals) + 1
    draws = [generator.random() for _ in range(sample_count)]
    total = sum(draws) or 1.0
    normalized = [value / total for value in draws]
    weights = {
        asset: normalized[index]
        for index, asset in enumerate(observation.signals.keys())
    }
    return PortfolioAction(target_weights=weights, reason="Random baseline.")


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model response did not include a JSON object")
    return json.loads(text[start : end + 1])


@dataclass
class LLMAllocatorPolicy:
    """LLM-backed policy using the OpenAI Python client."""

    client: OpenAI
    model_name: str

    def __call__(self, observation: AllocatorObservation) -> PortfolioAction:
        prompt = (
            "You are allocating an AMC portfolio for one decision step.\n"
            f"Task: {observation.task_id} - {observation.task_description}\n"
            f"Step: {observation.step_index} / {observation.step_index + observation.steps_remaining}\n"
            f"Prices: {json.dumps(observation.prices, sort_keys=True)}\n"
            f"Signals: {json.dumps(observation.signals, sort_keys=True)}\n"
            f"Current weights: {json.dumps(observation.current_weights, sort_keys=True)}\n"
            f"Cash weight: {observation.cash_weight:.4f}\n"
            f"Risk metrics: {json.dumps(observation.risk_metrics, sort_keys=True)}\n"
            "Respond with JSON only: "
            '{"target_weights":{"INFY":0.2,"TCS":0.3},"reason":"short explanation"}.\n'
            "Constraints: no shorting, omit cash, total invested weight can be below 1.0."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a portfolio allocation model. Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = _extract_json_object(content)
        weights = payload.get("target_weights", {})
        reason = payload.get("reason")
        if not isinstance(weights, dict):
            raise ValueError("LLM response target_weights must be an object")
        cleaned = {str(asset).upper(): float(weight) for asset, weight in weights.items()}
        return PortfolioAction(target_weights=cleaned, reason=reason)


def build_policy(
    policy_name: str,
    seed: int = 7,
) -> Callable[[AllocatorObservation], PortfolioAction]:
    """Return a policy callable for the requested policy name."""

    normalized = policy_name.lower()
    if normalized == "heuristic":
        return heuristic_policy
    if normalized == "random":
        rng = random.Random(seed)
        return lambda observation: random_policy(observation, rng=rng)
    raise KeyError(f"Unknown policy {policy_name!r}")
