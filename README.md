---
title: AMC Allocator Environment
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# AMC Allocator Environment

`amc_allocator_env` is a deterministic OpenEnv environment for the Scaler Meta PyTorch Hackathon. It models a realistic AMC-style allocation problem: choosing portfolio weights across five Indian IT services stocks while balancing alpha capture, turnover, cash management, and drawdown control.

## Overview

- Real-world portfolio allocation, not a toy game
- Typed OpenEnv `Action`, `Observation`, and extended `State`
- Three deterministic tasks with fixed offline data
- Deterministic graders returning scores in `0.0–1.0`
- Root-level `inference.py` with `heuristic`, `random`, and `llm` policies
- Ready for OpenEnv validation, Docker deployment, and Hugging Face Spaces

## Task Suite

### `signal_following`
- 30 decision steps
- clean predictive signals
- minimal transaction cost
- objective: capture obvious cross-sectional alpha efficiently

### `noisy_market`
- 45 decision steps
- mixed and partially misleading signals
- high transaction cost
- objective: avoid overtrading while still extracting usable alpha

### `regime_shift`
- 60 decision steps
- latent market break with delayed signal usefulness
- explicit drawdown pressure
- objective: stay defensive through deterioration, then re-risk selectively

## Action, Observation, and State

### `PortfolioAction`
- `target_weights: dict[str, float]`
- `reason: str | None`

Weights refer only to risky assets. Any leftover allocation remains in cash automatically. Negative values are clipped to `0.0`, and overweight portfolios are normalized back to a maximum invested weight of `1.0`.

### `AllocatorObservation`
- `task_id`
- `task_description`
- `step_index`
- `steps_remaining`
- `prices`
- `signals`
- `current_weights`
- `cash_weight`
- `portfolio_value`
- `turnover`
- `risk_metrics`
- OpenEnv-native `reward`, `done`, and `metadata`

### `AllocatorState`

The state endpoint exposes episode bookkeeping including NAV history, turnover history, realized returns, signal alignment history, holdings, and active task metadata.

## Reward and Grading

Per-step reward is shaped to reflect practical allocator behavior:

```text
reward
= realized portfolio return
+ signal alignment bonus
- transaction cost
- variance proxy penalty
- drawdown penalty
```

This keeps trajectories informative while rewarding risk-aware behavior instead of raw return chasing.

Deterministic grading lives in [`graders.py`](./graders.py). Each episode is mapped into a bounded score in `0.0–1.0` using:

- total return
- max drawdown
- average turnover
- signal alignment
- positive reward step ratio

## Baseline Performance

Current deterministic heuristic benchmark:

```text
signal_following  score=0.603  return=9.317%
noisy_market      score=0.436  return=4.071%
regime_shift      score=0.954  return=17.845%
aggregate_score   score=0.664
```

Expected behavior:

- `heuristic` beats `random` on all three tasks
- all scores remain in `0.0–1.0`
- runtime stays comfortably below the hackathon limit

## Running Locally

Use Python 3.11 and the bundled lockfile.

```bash
uv sync --python python3.11 --extra dev
```

Run the deterministic heuristic baseline:

```bash
python3.11 inference.py --policy heuristic
```

Compare baselines:

```bash
python3.11 inference.py --policy random
python3.11 inference.py --policy heuristic
```

Run all policies:

```bash
python3.11 inference.py --policy all
```

Run one task only:

```bash
python3.11 inference.py --policy heuristic --task noisy_market
```

## LLM Policy

The `llm` policy uses the OpenAI Python client and supports both direct OpenAI usage and OpenAI-compatible providers.

Supported environment variables:

- `MODEL_NAME`
- `API_BASE_URL`
- `HF_TOKEN`
- `OPENAI_API_KEY`

Resolution order:

- auth token: `OPENAI_API_KEY`, otherwise `HF_TOKEN`
- base URL: `API_BASE_URL` when set, otherwise the default OpenAI endpoint

Example:

```bash
export MODEL_NAME=gpt-4.1-mini
export OPENAI_API_KEY=...
python3.11 inference.py --policy llm --task signal_following
```

## Server and Deployment

Run the FastAPI environment locally:

```bash
AMC_TASK_ID=signal_following uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Available task ids:

- `signal_following`
- `noisy_market`
- `regime_shift`

OpenEnv validation:

```bash
openenv validate --verbose
openenv build
openenv validate --url http://localhost:8000
```

Docker paths:

- [`Dockerfile`](./Dockerfile): Hugging Face Space runtime
- [`server/Dockerfile`](./server/Dockerfile): OpenEnv-oriented container build

## Project Layout

```text
amc_allocator_env/
├── README.md
├── Dockerfile
├── inference.py
├── openenv.yaml
├── tasks.py
├── graders.py
├── policies.py
├── models.py
├── data/
├── server/
└── tests/
```
