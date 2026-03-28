"""Deterministic offline market scenarios for the allocator tasks."""

from __future__ import annotations

import math
from typing import Dict

ASSET_BLUEPRINTS = [
    ("INFY", 1580.0, 0.1, 0.6),
    ("TCS", 3725.0, 0.7, 1.1),
    ("HCLTECH", 1485.0, 1.3, 1.7),
    ("WIPRO", 505.0, 1.9, 2.2),
    ("TECHM", 1180.0, 2.5, 2.8),
]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _build_signal_following(steps: int = 30) -> Dict[str, object]:
    prices = [{name: round(start, 4) for name, start, _, _ in ASSET_BLUEPRINTS}]
    signals = []
    for step in range(steps):
        signal_row = {}
        previous_prices = prices[-1]
        next_prices = {}
        sector_wave = 0.35 * math.sin((step + 1) / 4.0) + 0.2 * math.cos((step + 1) / 7.0)
        for index, (name, _, phase1, phase2) in enumerate(ASSET_BLUEPRINTS):
            alpha = 0.75 * math.sin((step + 1) / 3.2 + phase1)
            alpha += 0.35 * math.cos((step + 1) / 5.8 + phase2)
            signal = _clamp(alpha + 0.15 * sector_wave + 0.05 * (index - 2), -1.0, 1.0)
            asset_return = _clamp(
                0.0068 * signal
                + 0.0015 * sector_wave
                + 0.0009 * math.sin((step + index + 2) / 2.7),
                -0.03,
                0.03,
            )
            signal_row[name] = round(signal, 4)
            next_prices[name] = round(previous_prices[name] * (1.0 + asset_return), 4)
        signals.append(signal_row)
        prices.append(next_prices)

    return {
        "display_name": "Signal Following",
        "summary": "Clean cross-sectional alpha with minimal trading friction.",
        "steps": steps,
        "transaction_cost_bps": 2.0,
        "risk_aversion": 0.15,
        "drawdown_penalty": 0.0,
        "cash_bias": 0.05,
        "prices": prices,
        "signals": signals,
    }


def _build_noisy_market(steps: int = 45) -> Dict[str, object]:
    prices = [{name: round(start * 1.05, 4) for name, start, _, _ in ASSET_BLUEPRINTS}]
    signals = []
    for step in range(steps):
        signal_row = {}
        previous_prices = prices[-1]
        next_prices = {}
        sector_wave = 0.25 * math.sin((step + 1) / 4.8) - 0.18 * math.cos((step + 1) / 3.6)
        turbulence = 0.25 * math.sin((step + 1) / 1.9)
        for index, (name, _, phase1, phase2) in enumerate(ASSET_BLUEPRINTS):
            alpha = 0.55 * math.sin((step + 1) / 4.1 + phase1)
            alpha += 0.28 * math.cos((step + 1) / 6.2 + phase2)
            noise = 0.6 * math.cos((step + 2) / 2.4 + phase2)
            noise -= 0.35 * math.sin((step + 1) / 1.8 + phase1)
            signal = _clamp(0.7 * alpha + 0.3 * noise + 0.1 * sector_wave, -1.0, 1.0)
            asset_return = _clamp(
                0.0045 * alpha
                + 0.0012 * sector_wave
                - 0.0012 * abs(noise)
                + 0.0009 * turbulence * ((index % 2) * 2 - 1),
                -0.028,
                0.028,
            )
            signal_row[name] = round(signal, 4)
            next_prices[name] = round(previous_prices[name] * (1.0 + asset_return), 4)
        signals.append(signal_row)
        prices.append(next_prices)

    return {
        "display_name": "Noisy Market",
        "summary": "Conflicting signals with meaningful transaction costs and false positives.",
        "steps": steps,
        "transaction_cost_bps": 14.0,
        "risk_aversion": 0.4,
        "drawdown_penalty": 0.15,
        "cash_bias": 0.12,
        "prices": prices,
        "signals": signals,
    }


def _build_regime_shift(steps: int = 60) -> Dict[str, object]:
    prices = [{name: round(start * 0.98, 4) for name, start, _, _ in ASSET_BLUEPRINTS}]
    signals = []
    for step in range(steps):
        signal_row = {}
        previous_prices = prices[-1]
        next_prices = {}
        if step < 20:
            regime = 0.8
        elif step < 40:
            regime = -1.0
        else:
            regime = 0.45
        for index, (name, _, phase1, phase2) in enumerate(ASSET_BLUEPRINTS):
            rotation = math.sin((step + 1) / 7.2 + phase1)
            rotation += 0.65 * math.cos((step + 1) / 5.1 + phase2)
            delayed = math.sin(max(step - 2, 0) / 6.0 + phase1)
            defensive_bias = 1.0 if index in (0, 2) else -1.0
            signal = _clamp(
                0.45 * delayed + 0.35 * regime * defensive_bias + 0.18 * rotation,
                -1.0,
                1.0,
            )
            asset_return = _clamp(
                0.0042 * delayed
                + 0.0028 * regime * defensive_bias
                + 0.0011 * rotation
                - 0.0015 * (1.0 if regime < 0 and index in (3, 4) else 0.0),
                -0.035,
                0.03,
            )
            signal_row[name] = round(signal, 4)
            next_prices[name] = round(previous_prices[name] * (1.0 + asset_return), 4)
        signals.append(signal_row)
        prices.append(next_prices)

    return {
        "display_name": "Regime Shift",
        "summary": "Delayed signals and defensive rotation through hidden market regimes.",
        "steps": steps,
        "transaction_cost_bps": 8.0,
        "risk_aversion": 0.55,
        "drawdown_penalty": 0.45,
        "cash_bias": 0.2,
        "prices": prices,
        "signals": signals,
    }


MARKET_SCENARIOS = {
    "sector": "Indian IT Services",
    "assets": [name for name, *_ in ASSET_BLUEPRINTS],
    "tasks": {
        "signal_following": _build_signal_following(),
        "noisy_market": _build_noisy_market(),
        "regime_shift": _build_regime_shift(),
    },
}
