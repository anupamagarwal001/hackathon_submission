# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Round 2 AI Investment Committee environment implementation."""

from __future__ import annotations

import math
from statistics import mean
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import AllocatorObservation, AllocatorState, PortfolioAction
    from ..tasks import DEFAULT_TASK_ID, TaskScenario, get_task_scenario
except ImportError:  # pragma: no cover
    from models import AllocatorObservation, AllocatorState, PortfolioAction
    from tasks import DEFAULT_TASK_ID, TaskScenario, get_task_scenario


ALLOCATION_TEMPLATE_DESCRIPTIONS = {
    "balanced_top3": "Equal-weight the three strongest ideas while preserving mandate cash.",
    "top2_conviction": "Concentrate into the two strongest opportunities.",
    "defensive_quality": "Rotate into quality IT names with higher residual cash.",
    "concentrated_alpha": "Take a sharper alpha bet in the best idea plus a supporting name.",
}

DEFENSIVE_ASSETS = ("INFY", "HCLTECH")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _compute_drawdown(nav_history: list[float]) -> float:
    peak = nav_history[0] if nav_history else 1.0
    max_drawdown = 0.0
    for nav in nav_history:
        peak = max(peak, nav)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, 1.0 - nav / peak)
    return max_drawdown


def _normalize_requested_weights(raw_weights: dict[str, float], assets: list[str]) -> dict[str, float]:
    normalized_assets = [asset.upper() for asset in assets]
    return {
        asset: float(raw_weights.get(asset, raw_weights.get(asset.upper(), 0.0)))
        for asset in normalized_assets
    }


def _apply_weight_constraints(
    raw_weights: dict[str, float],
    assets: list[str],
    constraints: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    max_single = constraints.get("max_single_asset_weight", 1.0)
    min_cash = constraints.get("min_cash_weight", 0.0)
    max_invested = max(0.0, 1.0 - min_cash)

    normalized = _normalize_requested_weights(raw_weights, assets)
    invalid_negative = sum(abs(weight) for weight in normalized.values() if weight < 0.0)
    capped = {
        asset: min(max_single, max(0.0, weight))
        for asset, weight in normalized.items()
    }
    max_single_excess = sum(max(0.0, normalized[asset] - max_single) for asset in capped)

    total = sum(capped.values())
    if total > max_invested and total > 0.0:
        scale = max_invested / total
        capped = {asset: weight * scale for asset, weight in capped.items()}
        total = sum(capped.values())

    cash_weight = max(0.0, 1.0 - total)
    cash_shortfall = max(0.0, min_cash - cash_weight)
    violation = invalid_negative + max_single_excess + cash_shortfall
    return capped, cash_weight, violation


class AmcAllocatorEnvironment(Environment[PortfolioAction, AllocatorObservation, AllocatorState]):
    """Committee-style portfolio environment for Round 2."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = DEFAULT_TASK_ID):
        super().__init__()
        self._scenario = get_task_scenario(task_id)
        self._price_index = 0
        self._state = AllocatorState()
        self.reset()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="amc_allocator_env",
            description=(
                "Multi-agent investment committee environment with Portfolio Manager, "
                "Research Analyst, and Risk Officer interactions."
            ),
            version="2.0.0",
        )

    def set_task(self, task_id: str) -> None:
        self._scenario = get_task_scenario(task_id)

    @property
    def state(self) -> AllocatorState:
        return self._state

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> AllocatorObservation:
        if kwargs.get("task_id"):
            self.set_task(str(kwargs["task_id"]))
        self._price_index = 0
        assets = list(self._scenario.assets)
        self._state = AllocatorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._scenario.task_id,
            task_description=self._scenario.summary,
            total_steps=self._scenario.steps,
            current_step=0,
            assets=assets,
            holdings={asset: 0.0 for asset in assets},
            current_weights={asset: 0.0 for asset in assets},
            cash_weight=1.0,
            portfolio_value=1.0,
            nav_history=[1.0],
            turnover_history=[],
            reward_history=[],
            portfolio_return_history=[],
            signal_alignment_history=[],
            information_usage_history=[],
            risk_response_history=[],
            compliance_penalty_history=[],
            reward_component_history=[],
            risk_metrics={},
            active_constraints=self._constraints_for_step(0),
            research_notes_history=[],
            risk_notes_history=[],
            action_history=[],
            allocation_template_history=[],
            hidden_regime_history=[],
            latest_research_view={},
            latest_risk_view={},
            query_count=0,
            outstanding_risk_alert=False,
            compliance_violations=0,
            last_turnover=0.0,
            last_reward=0.0,
            last_action_type="reset",
            last_action_reason="reset",
        )
        self._state.risk_metrics = self._build_risk_metrics()
        self._state.outstanding_risk_alert = (
            self._state.risk_metrics["risk_pressure"] >= self._scenario.risk_alert_threshold
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            turnover=0.0,
            reason="Environment reset.",
            reward_components={},
        )

    def step(
        self,
        action: PortfolioAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AllocatorObservation:
        if self._price_index >= self._scenario.steps:
            return self._build_observation(
                reward=0.0,
                done=True,
                turnover=0.0,
                reason="Episode already complete.",
                reward_components=self._state.reward_component_history[-1]
                if self._state.reward_component_history
                else {},
            )

        self._state.active_constraints = self._constraints_for_step(self._price_index)
        assets = self._scenario.assets
        current_signal_row = self._scenario.signals[self._price_index]
        current_price_row = self._scenario.prices[self._price_index]
        next_price_row = self._scenario.prices[self._price_index + 1]
        previous_weights = dict(self._state.current_weights)
        previous_cash = float(self._state.cash_weight)
        previous_concentration = sum(weight * weight for weight in previous_weights.values())

        action_type = action.action_type
        action_template = action.allocation_template
        target_weights = dict(previous_weights)
        cash_weight = previous_cash
        query_penalty = 0.0
        invalid_action_penalty = 0.0
        compliance_penalty = 0.0

        if action_type == "query_research":
            query_penalty = self._consume_query_budget()
            if query_penalty >= 0.02:
                invalid_action_penalty += 0.02
            else:
                note, view = self._generate_research_note(action.query_target, current_signal_row)
                self._state.research_notes_history.append(note)
                self._state.latest_research_view = view
        elif action_type == "query_risk":
            query_penalty = self._consume_query_budget()
            if query_penalty >= 0.02:
                invalid_action_penalty += 0.02
            else:
                note, view, alert = self._generate_risk_note(current_signal_row)
                self._state.risk_notes_history.append(note)
                self._state.latest_risk_view = view
                self._state.outstanding_risk_alert = alert
        elif action_type in {"allocate", "revise_allocation"}:
            if action.target_weights:
                raw_weights = dict(action.target_weights)
                action_template = action_template or "custom_weights"
            elif action.allocation_template:
                raw_weights = self._resolve_template(action.allocation_template, current_signal_row)
            else:
                raw_weights = {}
                invalid_action_penalty += 0.02

            target_weights, cash_weight, weight_violation = _apply_weight_constraints(
                raw_weights,
                assets,
                self._state.active_constraints,
            )
            turnover = sum(
                abs(target_weights[asset] - previous_weights.get(asset, 0.0)) for asset in assets
            )
            turnover_violation = max(
                0.0,
                turnover - self._state.active_constraints.get("max_turnover", 1.0),
            )
            compliance_penalty = 0.04 * (weight_violation + turnover_violation)
        elif action_type == "move_to_cash":
            target_weights = {asset: 0.0 for asset in assets}
            cash_weight = 1.0
        elif action_type == "hold":
            pass
        else:
            invalid_action_penalty += 0.02

        turnover = sum(
            abs(target_weights[asset] - previous_weights.get(asset, 0.0)) for asset in assets
        )
        if action_type not in {"allocate", "revise_allocation"}:
            turnover_violation = 0.0
        else:
            turnover_violation = max(
                0.0,
                turnover - self._state.active_constraints.get("max_turnover", 1.0),
            )
            compliance_penalty = max(compliance_penalty, 0.04 * turnover_violation)

        transaction_cost = turnover * self._scenario.transaction_cost_bps / 10_000.0
        asset_returns = {
            asset: next_price_row[asset] / current_price_row[asset] - 1.0
            for asset in assets
        }
        gross_return = sum(target_weights[asset] * asset_returns[asset] for asset in assets)
        average_asset_return = mean(asset_returns.values())
        variance_proxy = mean(
            [(asset_return - average_asset_return) ** 2 for asset_return in asset_returns.values()]
        )
        concentration = sum(weight * weight for weight in target_weights.values())
        signal_alignment = sum(target_weights[asset] * current_signal_row[asset] for asset in assets)
        information_usage = self._score_information_usage(target_weights)
        risk_response = self._score_risk_response(
            previous_weights,
            target_weights,
            previous_cash,
            cash_weight,
        )
        risk_penalty = self._scenario.risk_aversion * variance_proxy * (1.0 + concentration)
        provisional_nav = max(
            0.01,
            self._state.portfolio_value * (1.0 + gross_return - transaction_cost),
        )
        projected_nav_history = self._state.nav_history + [provisional_nav]
        max_drawdown = _compute_drawdown(projected_nav_history)
        drawdown_penalty = self._scenario.drawdown_penalty * max(0.0, max_drawdown - 0.04)
        signal_alignment_bonus = 0.002 * signal_alignment
        information_usage_bonus = 0.003 * information_usage
        risk_response_bonus = 0.003 * risk_response
        reward = (
            gross_return
            + signal_alignment_bonus
            + information_usage_bonus
            + risk_response_bonus
            - transaction_cost
            - query_penalty
            - risk_penalty
            - drawdown_penalty
            - compliance_penalty
            - invalid_action_penalty
        )
        reward_components = {
            "portfolio_return": round(gross_return, 6),
            "transaction_cost_penalty": round(transaction_cost, 6),
            "query_penalty": round(query_penalty, 6),
            "variance_penalty": round(risk_penalty, 6),
            "drawdown_penalty": round(drawdown_penalty, 6),
            "compliance_penalty": round(compliance_penalty, 6),
            "invalid_action_penalty": round(invalid_action_penalty, 6),
            "signal_alignment_bonus": round(signal_alignment_bonus, 6),
            "information_usage_bonus": round(information_usage_bonus, 6),
            "risk_response_bonus": round(risk_response_bonus, 6),
        }

        self._price_index += 1
        self._state.step_count = self._price_index
        self._state.current_step = self._price_index
        self._state.current_weights = target_weights
        self._state.cash_weight = cash_weight
        self._state.portfolio_value = provisional_nav
        self._state.nav_history.append(provisional_nav)
        self._state.turnover_history.append(turnover)
        self._state.reward_history.append(reward)
        self._state.portfolio_return_history.append(gross_return)
        self._state.signal_alignment_history.append(signal_alignment)
        self._state.information_usage_history.append(information_usage)
        self._state.risk_response_history.append(risk_response)
        self._state.compliance_penalty_history.append(compliance_penalty + invalid_action_penalty)
        self._state.reward_component_history.append(reward_components)
        self._state.last_turnover = turnover
        self._state.last_reward = reward
        self._state.last_action_type = action_type
        self._state.last_action_reason = action.reason
        self._state.holdings = {
            asset: (provisional_nav * target_weights[asset]) / next_price_row[asset]
            for asset in assets
        }
        self._state.action_history.append(action_type)
        self._state.allocation_template_history.append(action_template or "")
        self._state.hidden_regime_history.append(self._regime_label(self._price_index - 1))
        if compliance_penalty + invalid_action_penalty > 0.0:
            self._state.compliance_violations += 1

        self._apply_constraint_updates_for_next_step()
        self._state.risk_metrics = self._build_risk_metrics()
        self._state.outstanding_risk_alert = (
            self._state.risk_metrics["risk_pressure"] >= self._scenario.risk_alert_threshold
        )

        done = self._price_index >= self._scenario.steps
        return self._build_observation(
            reward=reward,
            done=done,
            turnover=turnover,
            reason=action.reason or action_type,
            reward_components=reward_components,
        )

    def _consume_query_budget(self) -> float:
        if self._state.query_count >= self._scenario.max_queries:
            return 0.02
        self._state.query_count += 1
        return self._scenario.query_cost

    def _constraints_for_step(self, step_index: int) -> dict[str, float]:
        constraints = dict(self._scenario.base_constraints)
        for schedule_step in sorted(self._scenario.constraint_schedule):
            if schedule_step <= step_index:
                constraints.update(self._scenario.constraint_schedule[schedule_step])
        return constraints

    def _apply_constraint_updates_for_next_step(self) -> None:
        if self._price_index >= self._scenario.steps:
            return
        previous = dict(self._state.active_constraints)
        updated = self._constraints_for_step(self._price_index)
        self._state.active_constraints = updated
        if (
            self._price_index in self._scenario.constraint_schedule
            and previous != updated
        ):
            note = (
                "Mandate update: "
                f"max single {updated['max_single_asset_weight']:.2f}, "
                f"min cash {updated['min_cash_weight']:.2f}, "
                f"max turnover {updated['max_turnover']:.2f}."
            )
            self._state.risk_notes_history.append(note)
            self._state.latest_risk_view = {
                "recommended_cash_weight": updated["min_cash_weight"],
                "recommended_max_single_weight": updated["max_single_asset_weight"],
                "risk_pressure": max(
                    self._state.latest_risk_view.get("risk_pressure", 0.0),
                    0.45,
                ),
            }

    def _regime_label(self, step_index: int) -> str:
        bounded = min(step_index, self._scenario.steps - 1)
        return self._scenario.regimes[bounded]

    def _estimate_risk_pressure(self, signal_row: dict[str, float]) -> float:
        market_signal = mean(signal_row.values())
        max_drawdown = _compute_drawdown(self._state.nav_history or [1.0])
        regime = self._regime_label(min(self._price_index, self._scenario.steps - 1))
        regime_pressure = {
            "steady": 0.05,
            "baseline": 0.08,
            "crowded": 0.12,
            "fragile": 0.22,
            "expansion": 0.08,
            "stress": 0.35,
            "repair": 0.15,
            "tightening": 0.26,
            "oversight": 0.18,
        }.get(regime, 0.12)
        pressure = max(0.0, -market_signal) * 0.7
        pressure += max(0.0, max_drawdown - 0.03) * 2.0
        pressure += regime_pressure
        return _clamp(pressure, 0.0, 1.0)

    def _generate_research_note(
        self,
        query_target: str | None,
        signal_row: dict[str, float],
    ) -> tuple[str, dict[str, float]]:
        view: dict[str, float] = {}
        for index, asset in enumerate(self._scenario.assets):
            noise = self._scenario.research_noise * math.sin(
                (self._price_index + 1) * (index + 1) / 2.9
            )
            view[asset] = round(_clamp(signal_row[asset] + noise, -1.0, 1.0), 4)

        ranked = sorted(view, key=view.get, reverse=True)
        focus = query_target if query_target in self._scenario.assets else "SECTOR"
        focus_line = (
            f"Research focus {focus}: " if focus != "SECTOR" else "Research sector view: "
        )
        top_assets = ", ".join(
            f"{asset} ({view[asset]:+.2f})" for asset in ranked[:2]
        )
        confidence = mean(abs(view[asset]) for asset in ranked[:2])
        confidence_label = "high" if confidence > 0.55 else "medium" if confidence > 0.3 else "low"
        note = f"{focus_line}top ideas are {top_assets}; confidence {confidence_label}."
        return note, view

    def _generate_risk_note(
        self,
        signal_row: dict[str, float],
    ) -> tuple[str, dict[str, float], bool]:
        risk_pressure = self._estimate_risk_pressure(signal_row)
        recommended_cash = _clamp(
            self._state.active_constraints["min_cash_weight"]
            + self._scenario.cash_bias
            + 0.25 * risk_pressure,
            0.05,
            0.75,
        )
        recommended_max_single = min(
            self._state.active_constraints["max_single_asset_weight"],
            max(0.18, self._state.active_constraints["max_single_asset_weight"] - 0.12 * risk_pressure),
        )
        alert = risk_pressure >= self._scenario.risk_alert_threshold
        tone = "elevated risk" if alert else "contained risk"
        note = (
            f"Risk review: {tone}; keep cash near {recommended_cash:.0%} and "
            f"single-name exposure below {recommended_max_single:.0%}."
        )
        return note, {
            "recommended_cash_weight": round(recommended_cash, 6),
            "recommended_max_single_weight": round(recommended_max_single, 6),
            "risk_pressure": round(risk_pressure, 6),
        }, alert

    def _resolve_template(
        self,
        template_name: str,
        signal_row: dict[str, float],
    ) -> dict[str, float]:
        source_scores = self._state.latest_research_view or signal_row
        ranked = sorted(source_scores, key=source_scores.get, reverse=True)
        investable = max(0.0, 1.0 - self._state.active_constraints.get("min_cash_weight", 0.0))

        if template_name == "balanced_top3":
            selected = ranked[:3]
            if not selected:
                return {}
            weight = investable / len(selected)
            return {asset: weight for asset in selected}
        if template_name == "top2_conviction":
            selected = ranked[:2]
            if not selected:
                return {}
            if len(selected) == 1:
                return {selected[0]: investable * 0.82}
            return {
                selected[0]: investable * 0.58,
                selected[1]: investable * 0.32,
            }
        if template_name == "defensive_quality":
            defensive = [asset for asset in DEFENSIVE_ASSETS if asset in self._scenario.assets]
            if not defensive:
                return {}
            if len(defensive) == 1:
                return {defensive[0]: investable * 0.45}
            return {
                defensive[0]: investable * 0.35,
                defensive[1]: investable * 0.25,
            }
        if template_name == "concentrated_alpha":
            selected = ranked[:2]
            if not selected:
                return {}
            if len(selected) == 1:
                return {selected[0]: investable * 0.88}
            return {
                selected[0]: investable * 0.72,
                selected[1]: investable * 0.16,
            }
        return {}

    def _score_information_usage(self, target_weights: dict[str, float]) -> float:
        if not self._state.latest_research_view:
            return 0.0
        return sum(
            target_weights[asset] * self._state.latest_research_view.get(asset, 0.0)
            for asset in self._scenario.assets
        )

    def _score_risk_response(
        self,
        previous_weights: dict[str, float],
        target_weights: dict[str, float],
        previous_cash: float,
        cash_weight: float,
    ) -> float:
        risk_pressure = float(
            self._state.latest_risk_view.get(
                "risk_pressure",
                self._state.risk_metrics.get("risk_pressure", 0.0),
            )
        )
        previous_concentration = sum(weight * weight for weight in previous_weights.values())
        new_concentration = sum(weight * weight for weight in target_weights.values())
        defensive_shift = max(0.0, cash_weight - previous_cash) + max(
            0.0,
            previous_concentration - new_concentration,
        )
        aggressive_shift = max(0.0, previous_cash - cash_weight) + max(
            0.0,
            new_concentration - previous_concentration,
        )
        return defensive_shift * risk_pressure - 0.5 * aggressive_shift * max(0.0, risk_pressure - 0.2)

    def _build_risk_metrics(self) -> dict[str, float]:
        signal_index = min(self._price_index, self._scenario.steps - 1)
        signal_row = self._scenario.signals[signal_index]
        weights = self._state.current_weights or {asset: 0.0 for asset in self._scenario.assets}
        market_signal = mean(signal_row.values())
        signal_dispersion = mean(abs(signal - market_signal) for signal in signal_row.values())
        concentration = sum(weight * weight for weight in weights.values())
        max_drawdown = _compute_drawdown(self._state.nav_history or [1.0])
        average_turnover = (
            mean(self._state.turnover_history) if self._state.turnover_history else 0.0
        )
        cash_buffer_hint = _clamp(
            self._scenario.cash_bias + max(0.0, -market_signal) * 0.18 + max_drawdown * 0.85,
            self._state.active_constraints.get("min_cash_weight", 0.02),
            0.75,
        )
        max_single = max(weights.values(), default=0.0)
        compliance_headroom = (
            self._state.active_constraints.get("max_single_asset_weight", 1.0) - max_single
        )
        risk_pressure = self._estimate_risk_pressure(signal_row)
        return {
            "market_signal": round(market_signal, 6),
            "signal_dispersion": round(signal_dispersion, 6),
            "concentration": round(concentration, 6),
            "average_turnover": round(average_turnover, 6),
            "max_drawdown": round(max_drawdown, 6),
            "cash_buffer_hint": round(cash_buffer_hint, 6),
            "risk_pressure": round(risk_pressure, 6),
            "compliance_headroom": round(compliance_headroom, 6),
        }

    def _build_observation(
        self,
        reward: float,
        done: bool,
        turnover: float,
        reason: str,
        reward_components: dict[str, float],
    ) -> AllocatorObservation:
        signal_index = min(self._price_index, self._scenario.steps - 1)
        price_row = self._scenario.prices[self._price_index]
        signal_row = self._scenario.signals[signal_index]
        metadata = {
            "sector": self._scenario.sector,
            "task_name": self._scenario.display_name,
            "reason": reason,
            "transaction_cost_bps": self._scenario.transaction_cost_bps,
        }
        return AllocatorObservation(
            task_id=self._scenario.task_id,
            task_description=self._scenario.summary,
            step_index=self._price_index,
            steps_remaining=max(0, self._scenario.steps - self._price_index),
            prices=price_row,
            signals=signal_row,
            current_weights=self._state.current_weights,
            cash_weight=self._state.cash_weight,
            portfolio_value=self._state.portfolio_value,
            turnover=turnover,
            risk_metrics=self._state.risk_metrics,
            active_constraints=self._state.active_constraints,
            reward_components=reward_components,
            research_notes=self._state.research_notes_history[-3:],
            risk_notes=self._state.risk_notes_history[-3:],
            available_allocation_templates=ALLOCATION_TEMPLATE_DESCRIPTIONS,
            outstanding_risk_alert=self._state.outstanding_risk_alert,
            queries_remaining=max(0, self._scenario.max_queries - self._state.query_count),
            reward=reward,
            done=done,
            metadata=metadata,
        )
