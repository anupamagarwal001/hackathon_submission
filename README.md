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

`amc_allocator_env` is a deterministic OpenEnv environment for the Scaler Meta PyTorch hackathon. It simulates an AMC-style allocator choosing portfolio weights across five Indian IT services stocks under three progressively harder tasks.

The environment is submission-oriented:

- real-world portfolio allocation instead of a toy game
- typed OpenEnv `Action`, `Observation`, and extended `State`
- three deterministic tasks with grader scores in `0.0–1.0`
- root-level `inference.py` with `heuristic`, `random`, and `llm` policies
- Docker and Hugging Face Space friendly layout

## Tasks

### 1. `signal_following`
- 30 decision steps
- clean predictive signals
- near-zero transaction cost
- objective: exploit straightforward cross-sectional alpha

### 2. `noisy_market`
- 45 decision steps
- conflicting and partially misleading signals
- high transaction cost
- objective: avoid overtrading while still capturing alpha

### 3. `regime_shift`
- 60 decision steps
- delayed signals with defensive rotation during a hidden downturn
- explicit drawdown penalty
- objective: protect capital through the regime break, then re-risk selectively

## Action and Observation Spaces

### Action: `PortfolioAction`
- `target_weights: dict[str, float]`
- `reason: str | None`

Weights represent asset allocations only. Any unallocated weight becomes cash automatically. The environment clips negative weights to `0.0` and normalizes overweight portfolios back to `1.0`.

### Observation: `AllocatorObservation`
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

### State: `AllocatorState`

The state endpoint exposes episode bookkeeping such as NAV history, turnover history, realized returns, signal alignment, holdings, and the active task metadata.

## Reward Design

Per-step reward is shaped, not just terminal:

```text
reward
= realized portfolio return
+ signal alignment bonus
- transaction cost
- variance proxy penalty
- drawdown penalty (task dependent)
```

This keeps the trajectory informative while still pushing the agent toward risk-adjusted performance.

## Graders

Each task has a deterministic grader in [`graders.py`](/Users/anuagar/Desktop/dev/amc_allocator_env/graders.py). Every grader maps episode metrics into `0.0–1.0` using hard-coded thresholds.

Metrics used:
- total return
- max drawdown
- average turnover
- signal alignment
- positive reward step ratio

## Local Setup

Use Python 3.11 and the bundled `uv.lock`.

```bash
cd /Users/anuagar/Desktop/dev/amc_allocator_env
/Users/anuagar/Library/Python/3.11/bin/uv sync --python python3.11 --extra dev
```

## Run the Baseline Inference

Default mode is the reproducible heuristic baseline:

```bash
cd /Users/anuagar/Desktop/dev/amc_allocator_env
python3.11 inference.py --policy heuristic
```

Compare deterministic baselines:

```bash
python3.11 inference.py --policy random
python3.11 inference.py --policy heuristic
```

Run all configured policies:

```bash
python3.11 inference.py --policy all
```

Run a single task:

```bash
python3.11 inference.py --policy heuristic --task noisy_market
```

Expected behavior:
- `heuristic` should beat `random` on all three tasks
- all task scores stay within `0.0–1.0`
- runtime stays well under the hackathon’s 20 minute limit

## LLM Policy Configuration

The `llm` policy uses the OpenAI Python client and accepts the dashboard’s mixed configuration conventions.

Supported environment variables:
- `MODEL_NAME`
- `API_BASE_URL`
- `HF_TOKEN`
- `OPENAI_API_KEY`

Resolution order:
- auth token: `OPENAI_API_KEY`, otherwise `HF_TOKEN`
- base URL: `API_BASE_URL` when set, otherwise default OpenAI endpoint

Example:

```bash
export MODEL_NAME=gpt-4.1-mini
export OPENAI_API_KEY=...
python3.11 inference.py --policy llm --task signal_following
```

## Run the Server Locally

```bash
cd /Users/anuagar/Desktop/dev/amc_allocator_env
AMC_TASK_ID=signal_following uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Available task ids:
- `signal_following`
- `noisy_market`
- `regime_shift`

## Docker

Build:

```bash
cd /Users/anuagar/Desktop/dev/amc_allocator_env
docker build -t amc_allocator_env:latest -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 -e AMC_TASK_ID=signal_following amc_allocator_env:latest
```

## OpenEnv Validation

```bash
cd /Users/anuagar/Desktop/dev/amc_allocator_env
PATH="/Users/anuagar/Library/Python/3.11/bin:$PATH" openenv validate --verbose
PATH="/Users/anuagar/Library/Python/3.11/bin:$PATH" openenv build
PATH="/Users/anuagar/Library/Python/3.11/bin:$PATH" openenv validate --url http://localhost:8000
```

## Hugging Face Space

After local validation passes:

```bash
cd /Users/anuagar/Desktop/dev/amc_allocator_env
PATH="/Users/anuagar/Library/Python/3.11/bin:$PATH" openenv push --repo-id <hf-user>/amc_allocator_env
```

## Project Layout

```text
amc_allocator_env/
├── README.md
├── __init__.py
├── client.py
├── data/
│   ├── __init__.py
│   └── market_scenarios.py
├── graders.py
├── inference.py
├── models.py
├── openenv.yaml
├── policies.py
├── pyproject.toml
├── server/
│   ├── amc_environment.py
│   ├── app.py
│   └── Dockerfile
├── tasks.py
└── tests/
```
