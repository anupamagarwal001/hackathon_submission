# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the AMC allocator environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import AllocatorObservation, AllocatorState, PortfolioAction


class AmcAllocatorEnv(EnvClient[PortfolioAction, AllocatorObservation, AllocatorState]):
    """Persistent client for allocator episodes over WebSocket."""

    def _step_payload(self, action: PortfolioAction) -> Dict:
        return {
            "target_weights": action.target_weights,
            "reason": action.reason,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AllocatorObservation]:
        obs_data = payload.get("observation", {})
        observation = AllocatorObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            step_index=obs_data.get("step_index", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            prices=obs_data.get("prices", {}),
            signals=obs_data.get("signals", {}),
            current_weights=obs_data.get("current_weights", {}),
            cash_weight=obs_data.get("cash_weight", 1.0),
            portfolio_value=obs_data.get("portfolio_value", 1.0),
            turnover=obs_data.get("turnover", 0.0),
            risk_metrics=obs_data.get("risk_metrics", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AllocatorState:
        return AllocatorState.model_validate(payload)
