"""Baseline policies for committee-style evaluation."""

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


def heuristic_policy(observation: AllocatorObservation) -> PortfolioAction:
    """Committee-aware heuristic baseline for the Portfolio Manager."""

    market_signal = observation.risk_metrics.get("market_signal", 0.0)
    risk_pressure = observation.risk_metrics.get("risk_pressure", 0.0)
    drawdown = observation.risk_metrics.get("max_drawdown", 0.0)
    dispersion = observation.risk_metrics.get("signal_dispersion", 0.0)
    min_cash = observation.active_constraints.get("min_cash_weight", 0.0)

    if observation.queries_remaining > 0:
        if observation.step_index == 0 and not observation.research_notes:
            return PortfolioAction(
                action_type="query_research",
                query_target="SECTOR",
                reason="Build initial analyst view before taking risk.",
            )
        if observation.outstanding_risk_alert and not observation.risk_notes:
            return PortfolioAction(
                action_type="query_risk",
                reason="Need an explicit risk read before reallocating.",
            )
        if (
            observation.task_id == "research_risk_conflict"
            and observation.step_index in (0, 10, 24)
            and len(observation.research_notes) <= 2
        ):
            return PortfolioAction(
                action_type="query_research",
                query_target="SECTOR",
                reason="Refresh research view during conflict-heavy regime.",
            )

    if market_signal < -0.16 or drawdown > 0.08:
        return PortfolioAction(
            action_type="move_to_cash",
            reason="Preserve capital under deteriorating conditions.",
        )

    if observation.outstanding_risk_alert or risk_pressure > 0.58 or min_cash >= 0.15:
        return PortfolioAction(
            action_type="allocate",
            allocation_template="defensive_quality",
            reason="Shift into defensive quality allocation under elevated risk.",
        )

    best_signal = max(observation.signals.values(), default=0.0)
    if best_signal > 0.58 and dispersion > 0.18:
        template = "top2_conviction"
        if observation.active_constraints.get("max_single_asset_weight", 1.0) < 0.40:
            template = "balanced_top3"
        return PortfolioAction(
            action_type="allocate",
            allocation_template=template,
            reason="Lean into the strongest cross-sectional opportunities.",
        )

    if observation.task_id == "mandate_drift" and min_cash >= 0.12:
        return PortfolioAction(
            action_type="allocate",
            allocation_template="defensive_quality",
            reason="Respect tighter mandate and preserve flexibility.",
        )

    if best_signal > 0.25 or market_signal > 0.04:
        return PortfolioAction(
            action_type="allocate",
            allocation_template="balanced_top3",
            reason="Express the current research and signal mix with balanced exposure.",
        )

    return PortfolioAction(
        action_type="hold",
        reason="No edge is strong enough to justify a fresh rebalance.",
    )


def random_policy(
    observation: AllocatorObservation,
    rng: random.Random | None = None,
) -> PortfolioAction:
    """Random committee baseline."""

    generator = rng or random.Random()
    action_pool = ["hold", "move_to_cash", "allocate"]
    if observation.queries_remaining > 0:
        action_pool.extend(["query_research", "query_risk"])

    action_type = generator.choice(action_pool)
    if action_type == "query_research":
        query_target = generator.choice(["SECTOR", *observation.signals.keys()])
        return PortfolioAction(
            action_type="query_research",
            query_target=query_target,
            reason="Random research query.",
        )
    if action_type == "query_risk":
        return PortfolioAction(action_type="query_risk", reason="Random risk query.")
    if action_type == "move_to_cash":
        return PortfolioAction(action_type="move_to_cash", reason="Random defensive shift.")
    if action_type == "allocate":
        template = generator.choice(list(observation.available_allocation_templates))
        return PortfolioAction(
            action_type="allocate",
            allocation_template=template,
            reason="Random allocation template.",
        )
    return PortfolioAction(action_type="hold", reason="Random hold.")


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model response did not include a JSON object")
    return json.loads(text[start : end + 1])


@dataclass
class LLMAllocatorPolicy:
    """LLM-backed Portfolio Manager policy using the OpenAI Python client."""

    client: OpenAI
    model_name: str
    llm_available: bool = True

    def __call__(self, observation: AllocatorObservation) -> PortfolioAction:
        if not self.llm_available:
            return self._fallback_action(observation)

        prompt = (
            "You are the Portfolio Manager in a multi-agent investment committee.\n"
            f"Task: {observation.task_id} - {observation.task_description}\n"
            f"Step: {observation.step_index} / {observation.step_index + observation.steps_remaining}\n"
            f"Prices: {json.dumps(observation.prices, sort_keys=True)}\n"
            f"Signals: {json.dumps(observation.signals, sort_keys=True)}\n"
            f"Current weights: {json.dumps(observation.current_weights, sort_keys=True)}\n"
            f"Cash weight: {observation.cash_weight:.4f}\n"
            f"Constraints: {json.dumps(observation.active_constraints, sort_keys=True)}\n"
            f"Risk metrics: {json.dumps(observation.risk_metrics, sort_keys=True)}\n"
            f"Recent research notes: {json.dumps(observation.research_notes)}\n"
            f"Recent risk notes: {json.dumps(observation.risk_notes)}\n"
            f"Templates: {json.dumps(observation.available_allocation_templates, sort_keys=True)}\n"
            f"Queries remaining: {observation.queries_remaining}\n"
            "Respond with JSON only. Valid examples:\n"
            '{"action_type":"query_research","query_target":"SECTOR","reason":"need a fresh view"}\n'
            '{"action_type":"query_risk","reason":"risk is elevated"}\n'
            '{"action_type":"allocate","allocation_template":"balanced_top3","reason":"broad participation"}\n'
            '{"action_type":"allocate","target_weights":{"INFY":0.3,"TCS":0.2},"reason":"custom allocation"}\n'
            '{"action_type":"move_to_cash","reason":"preserve capital"}\n'
            '{"action_type":"hold","reason":"wait for clarity"}'
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an institutional Portfolio Manager. Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = _extract_json_object(content)
            return PortfolioAction(
                action_type=str(payload.get("action_type", "hold")).lower(),
                allocation_template=payload.get("allocation_template"),
                query_target=payload.get("query_target"),
                target_weights=payload.get("target_weights", {}),
                reason=payload.get("reason"),
            )
        except Exception:
            self.llm_available = False
            return self._fallback_action(observation)

    def _fallback_action(self, observation: AllocatorObservation) -> PortfolioAction:
        fallback = heuristic_policy(observation)
        reason = fallback.reason or "Heuristic fallback."
        return PortfolioAction(
            action_type=fallback.action_type,
            allocation_template=fallback.allocation_template,
            query_target=fallback.query_target,
            target_weights=fallback.target_weights,
            reason=f"{reason} LLM fallback activated.",
        )


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
