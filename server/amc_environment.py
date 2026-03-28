# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AMC allocator environment implementation."""

from __future__ import annotations

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


def _normalize_weights(
    raw_weights: dict[str, float], assets: list[str]
) -> tuple[dict[str, float], float]:
    normalized_assets = [asset.upper() for asset in assets]
    cleaned = {
        asset: max(0.0, float(raw_weights.get(asset, raw_weights.get(asset.upper(), 0.0))))
        for asset in normalized_assets
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return {asset: 0.0 for asset in normalized_assets}, 1.0
    if total > 1.0:
        cleaned = {asset: weight / total for asset, weight in cleaned.items()}
        total = 1.0
    cash_weight = max(0.0, 1.0 - total)
    return cleaned, cash_weight


class AmcAllocatorEnvironment(Environment[PortfolioAction, AllocatorObservation, AllocatorState]):
    """Single-agent portfolio allocator environment."""

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
                "Risk-aware portfolio allocation environment with deterministic "
                "signal-following, noisy-market, and regime-shift tasks."
            ),
            version="1.0.0",
        )

    def set_task(self, task_id: str) -> None:
        self._scenario = get_task_scenario(task_id)

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
            risk_metrics={},
            last_turnover=0.0,
            last_reward=0.0,
            last_action_reason="reset",
        )
        risk_metrics = self._build_risk_metrics()
        self._state.risk_metrics = risk_metrics
        return self._build_observation(
            reward=0.0,
            done=False,
            turnover=0.0,
            reason="Environment reset.",
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
            )

        assets = self._scenario.assets
        current_signal_row = self._scenario.signals[self._price_index]
        current_price_row = self._scenario.prices[self._price_index]
        next_price_row = self._scenario.prices[self._price_index + 1]

        target_weights, cash_weight = _normalize_weights(action.target_weights, assets)
        previous_weights = dict(self._state.current_weights)

        turnover = sum(
            abs(target_weights[asset] - previous_weights.get(asset, 0.0))
            for asset in assets
        )
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
        signal_alignment = sum(
            target_weights[asset] * current_signal_row[asset] for asset in assets
        )
        risk_penalty = self._scenario.risk_aversion * variance_proxy * (1.0 + concentration)
        shaping_bonus = 0.0025 * signal_alignment
        provisional_nav = max(
            0.01,
            self._state.portfolio_value * (1.0 + gross_return - transaction_cost),
        )
        projected_nav_history = self._state.nav_history + [provisional_nav]
        max_drawdown = _compute_drawdown(projected_nav_history)
        drawdown_penalty = self._scenario.drawdown_penalty * max(0.0, max_drawdown - 0.05)
        reward = gross_return + shaping_bonus - transaction_cost - risk_penalty - drawdown_penalty

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
        self._state.last_turnover = turnover
        self._state.last_reward = reward
        self._state.last_action_reason = action.reason
        self._state.holdings = {
            asset: (provisional_nav * target_weights[asset]) / next_price_row[asset]
            for asset in assets
        }
        self._state.risk_metrics = self._build_risk_metrics()

        done = self._price_index >= self._scenario.steps
        return self._build_observation(
            reward=reward,
            done=done,
            turnover=turnover,
            reason=action.reason or "",
        )

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
            self._scenario.cash_bias + max(0.0, -market_signal) * 0.22 + max_drawdown * 0.75,
            0.02,
            0.7,
        )
        return {
            "market_signal": round(market_signal, 6),
            "signal_dispersion": round(signal_dispersion, 6),
            "concentration": round(concentration, 6),
            "average_turnover": round(average_turnover, 6),
            "max_drawdown": round(max_drawdown, 6),
            "cash_buffer_hint": round(cash_buffer_hint, 6),
        }

    def _build_observation(
        self,
        reward: float,
        done: bool,
        turnover: float,
        reason: str,
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
            reward=reward,
            done=done,
            metadata=metadata,
        )

    @property
    def state(self) -> AllocatorState:
        return self._state
