"""Round 2 task definitions and scenario loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from .data import MARKET_SCENARIOS
except ImportError:  # pragma: no cover
    from data import MARKET_SCENARIOS

DEFAULT_TASK_ID = "guided_allocation"
TASK_ORDER = (
    "guided_allocation",
    "research_risk_conflict",
    "regime_shift_recovery",
    "mandate_drift",
)


@dataclass(frozen=True)
class TaskScenario:
    """Deterministic committee-style scenario configuration."""

    task_id: str
    display_name: str
    summary: str
    sector: str
    assets: list[str]
    steps: int
    transaction_cost_bps: float
    risk_aversion: float
    drawdown_penalty: float
    cash_bias: float
    query_cost: float
    max_queries: int
    research_noise: float
    risk_alert_threshold: float
    base_constraints: Dict[str, float]
    constraint_schedule: Dict[int, Dict[str, float]]
    prices: list[Dict[str, float]]
    signals: list[Dict[str, float]]
    regimes: list[str]

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
        if len(self.regimes) != self.steps:
            raise ValueError(
                f"{self.task_id} requires {self.steps} regime labels, got {len(self.regimes)}"
            )


def _clone_rows(rows: list[Dict[str, float]]) -> list[Dict[str, float]]:
    return [{asset: float(value) for asset, value in row.items()} for row in rows]


def _regimes(steps: int, labels: list[tuple[int, str]]) -> list[str]:
    output: list[str] = []
    cursor = 0
    for stop, label in labels:
        bounded_stop = min(stop, steps)
        while cursor < bounded_stop:
            output.append(label)
            cursor += 1
    while cursor < steps:
        output.append(labels[-1][1])
        cursor += 1
    return output


def load_task_scenarios() -> Dict[str, TaskScenario]:
    """Build the Round 2 task set from the offline market dataset."""

    sector = str(MARKET_SCENARIOS["sector"])
    assets = [str(asset) for asset in MARKET_SCENARIOS["assets"]]
    raw_tasks = MARKET_SCENARIOS["tasks"]

    signal_following = raw_tasks["signal_following"]
    noisy_market = raw_tasks["noisy_market"]
    regime_shift = raw_tasks["regime_shift"]

    scenarios = {
        "guided_allocation": TaskScenario(
            task_id="guided_allocation",
            display_name="Guided Allocation",
            summary=(
                "The Portfolio Manager learns the committee workflow in a stable market "
                "with clean research input and light risk pressure."
            ),
            sector=sector,
            assets=assets,
            steps=int(signal_following["steps"]),
            transaction_cost_bps=float(signal_following["transaction_cost_bps"]),
            risk_aversion=float(signal_following["risk_aversion"]),
            drawdown_penalty=float(signal_following["drawdown_penalty"]),
            cash_bias=float(signal_following["cash_bias"]),
            query_cost=0.0012,
            max_queries=8,
            research_noise=0.12,
            risk_alert_threshold=0.52,
            base_constraints={
                "max_single_asset_weight": 0.50,
                "min_cash_weight": 0.05,
                "max_turnover": 0.65,
            },
            constraint_schedule={},
            prices=_clone_rows(signal_following["prices"]),
            signals=_clone_rows(signal_following["signals"]),
            regimes=_regimes(int(signal_following["steps"]), [(30, "steady")]),
        ),
        "research_risk_conflict": TaskScenario(
            task_id="research_risk_conflict",
            display_name="Research vs Risk Conflict",
            summary=(
                "Bullish analyst views collide with tighter risk constraints and "
                "fragile market internals."
            ),
            sector=sector,
            assets=assets,
            steps=int(noisy_market["steps"]),
            transaction_cost_bps=float(noisy_market["transaction_cost_bps"]),
            risk_aversion=float(noisy_market["risk_aversion"]),
            drawdown_penalty=float(noisy_market["drawdown_penalty"]),
            cash_bias=float(noisy_market["cash_bias"]),
            query_cost=0.0018,
            max_queries=10,
            research_noise=0.22,
            risk_alert_threshold=0.46,
            base_constraints={
                "max_single_asset_weight": 0.38,
                "min_cash_weight": 0.08,
                "max_turnover": 0.45,
            },
            constraint_schedule={
                12: {"max_single_asset_weight": 0.32},
                28: {"min_cash_weight": 0.14},
            },
            prices=_clone_rows(noisy_market["prices"]),
            signals=_clone_rows(noisy_market["signals"]),
            regimes=_regimes(
                int(noisy_market["steps"]),
                [(15, "crowded"), (30, "fragile"), (45, "fragile")],
            ),
        ),
        "regime_shift_recovery": TaskScenario(
            task_id="regime_shift_recovery",
            display_name="Regime Shift Recovery",
            summary=(
                "The committee must respond to hidden regime deterioration, de-risk, "
                "and then selectively re-risk as conditions improve."
            ),
            sector=sector,
            assets=assets,
            steps=int(regime_shift["steps"]),
            transaction_cost_bps=float(regime_shift["transaction_cost_bps"]),
            risk_aversion=float(regime_shift["risk_aversion"]),
            drawdown_penalty=float(regime_shift["drawdown_penalty"]),
            cash_bias=float(regime_shift["cash_bias"]),
            query_cost=0.0016,
            max_queries=10,
            research_noise=0.18,
            risk_alert_threshold=0.42,
            base_constraints={
                "max_single_asset_weight": 0.42,
                "min_cash_weight": 0.10,
                "max_turnover": 0.40,
            },
            constraint_schedule={
                20: {"min_cash_weight": 0.18},
                40: {"min_cash_weight": 0.08},
            },
            prices=_clone_rows(regime_shift["prices"]),
            signals=_clone_rows(regime_shift["signals"]),
            regimes=_regimes(
                int(regime_shift["steps"]),
                [(20, "expansion"), (40, "stress"), (60, "repair")],
            ),
        ),
        "mandate_drift": TaskScenario(
            task_id="mandate_drift",
            display_name="Mandate Drift",
            summary=(
                "The PM must manage a long-horizon allocation process while compliance "
                "rules and cash mandates tighten mid-episode."
            ),
            sector=sector,
            assets=assets,
            steps=int(noisy_market["steps"]),
            transaction_cost_bps=float(noisy_market["transaction_cost_bps"]) + 2.0,
            risk_aversion=float(noisy_market["risk_aversion"]) + 0.08,
            drawdown_penalty=float(noisy_market["drawdown_penalty"]) + 0.05,
            cash_bias=float(noisy_market["cash_bias"]) + 0.03,
            query_cost=0.0015,
            max_queries=9,
            research_noise=0.18,
            risk_alert_threshold=0.44,
            base_constraints={
                "max_single_asset_weight": 0.45,
                "min_cash_weight": 0.05,
                "max_turnover": 0.50,
            },
            constraint_schedule={
                15: {"max_single_asset_weight": 0.28, "min_cash_weight": 0.12},
                30: {"max_single_asset_weight": 0.35, "min_cash_weight": 0.18, "max_turnover": 0.35},
            },
            prices=_clone_rows(noisy_market["prices"]),
            signals=_clone_rows(noisy_market["signals"]),
            regimes=_regimes(
                int(noisy_market["steps"]),
                [(15, "baseline"), (30, "tightening"), (45, "oversight")],
            ),
        ),
    }
    return scenarios


TASK_SCENARIOS = load_task_scenarios()


def get_task_scenario(task_id: str) -> TaskScenario:
    """Return the requested task scenario or raise on unknown task ids."""

    normalized_task_id = task_id.strip().lower()
    if normalized_task_id not in TASK_SCENARIOS:
        available = ", ".join(TASK_ORDER)
        raise KeyError(f"Unknown task_id={task_id!r}. Expected one of: {available}")
    return TASK_SCENARIOS[normalized_task_id]
