"""Task definitions and scenario loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    from .data import MARKET_SCENARIOS
except ImportError:  # pragma: no cover
    from data import MARKET_SCENARIOS

DEFAULT_TASK_ID = "signal_following"
TASK_ORDER = ("signal_following", "noisy_market", "regime_shift")


@dataclass(frozen=True)
class TaskScenario:
    """Deterministic scenario configuration for one allocator task."""

    task_id: str
    display_name: str
    summary: str
    sector: str
    assets: List[str]
    steps: int
    transaction_cost_bps: float
    risk_aversion: float
    drawdown_penalty: float
    cash_bias: float
    prices: List[Dict[str, float]]
    signals: List[Dict[str, float]]

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"{self.task_id} must define at least one step")
        if len(self.prices) != self.steps + 1:
            raise ValueError(
                f"{self.task_id} requires {self.steps + 1} price rows, got {len(self.prices)}"
            )
        if len(self.signals) != self.steps:
            raise ValueError(
                f"{self.task_id} requires {self.steps} signal rows, got {len(self.signals)}"
            )


def load_task_scenarios() -> Dict[str, TaskScenario]:
    """Load deterministic task scenarios from the checked-in dataset."""

    sector = str(MARKET_SCENARIOS["sector"])
    assets = [str(asset) for asset in MARKET_SCENARIOS["assets"]]
    tasks = {}
    raw_tasks = MARKET_SCENARIOS["tasks"]
    for task_id in TASK_ORDER:
        raw = raw_tasks[task_id]
        tasks[task_id] = TaskScenario(
            task_id=task_id,
            display_name=str(raw["display_name"]),
            summary=str(raw["summary"]),
            sector=sector,
            assets=assets,
            steps=int(raw["steps"]),
            transaction_cost_bps=float(raw["transaction_cost_bps"]),
            risk_aversion=float(raw["risk_aversion"]),
            drawdown_penalty=float(raw["drawdown_penalty"]),
            cash_bias=float(raw["cash_bias"]),
            prices=[{k: float(v) for k, v in row.items()} for row in raw["prices"]],
            signals=[{k: float(v) for k, v in row.items()} for row in raw["signals"]],
        )
    return tasks


TASK_SCENARIOS = load_task_scenarios()


def get_task_scenario(task_id: str) -> TaskScenario:
    """Return the requested task scenario or raise on unknown task ids."""

    normalized_task_id = task_id.strip().lower()
    if normalized_task_id not in TASK_SCENARIOS:
        available = ", ".join(TASK_ORDER)
        raise KeyError(f"Unknown task_id={task_id!r}. Expected one of: {available}")
    return TASK_SCENARIOS[normalized_task_id]
