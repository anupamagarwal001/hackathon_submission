---
title: AI Investment Committee Environment
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - finance
  - multi-agent
---

# AI Investment Committee Environment

`amc_allocator_env` is a deterministic OpenEnv environment for the Meta PyTorch OpenEnv Hackathon. It simulates a small institutional investment workflow where a trainable Portfolio Manager interacts with a scripted Research Analyst and Risk Officer while managing a portfolio of Indian IT services stocks.

## Overview

- committee-style environment, not a single-step toy allocator
- one trainable Portfolio Manager with scripted Research and Risk actors
- four deterministic Round 2 tasks with hidden market regimes and changing constraints
- typed OpenEnv `Action`, `Observation`, and extended `State`
- root-level `inference.py` for validator-compatible structured output
- minimal TRL training scaffold for the Portfolio Manager in [`training/`](./training)

## Task Suite

### `guided_allocation`
- stable regime and cleaner research signal
- objective: learn the basic committee workflow and sensible allocation

### `research_risk_conflict`
- bullish research with tighter risk limits
- objective: balance alpha-seeking against mandate discipline

### `regime_shift_recovery`
- hidden deterioration followed by repair
- objective: de-risk early and selectively re-risk later

### `mandate_drift`
- compliance rules tighten mid-episode
- objective: maintain long-horizon performance while staying compliant

## Committee Roles

### Portfolio Manager
- the only trainable policy
- chooses when to query Research, query Risk, rebalance, hold, or move to cash

### Research Analyst
- scripted environment actor
- provides noisy but useful asset or sector views

### Risk Officer
- scripted environment actor
- highlights concentration, drawdown, and mandate pressure

## Action, Observation, and State

### `PortfolioAction`
- `action_type`
  - `query_research`
  - `query_risk`
  - `allocate`
  - `revise_allocation`
  - `hold`
  - `move_to_cash`
- `allocation_template`
- `query_target`
- `target_weights`
- `reason`

### `AllocatorObservation`
- task and step metadata
- prices and signals
- current weights and cash
- active constraints
- recent Research notes and Risk notes
- reward component breakdown
- risk summary
- available allocation templates
- remaining advisory query budget

### `AllocatorState`

The state endpoint exposes:

- NAV and turnover histories
- committee note histories
- reward component history
- compliance events
- information-usage and risk-response histories
- hidden regime labels

## Reward and Grading

Per-step reward is shaped around institutional behavior:

```text
reward
= portfolio_return
+ signal_alignment_bonus
+ information_usage_bonus
+ risk_response_bonus
- transaction_cost
- query_cost
- variance_penalty
- drawdown_penalty
- compliance_penalty
- invalid_action_penalty
```

Deterministic episode grading maps each task to `0.0–1.0` in [`graders.py`](./graders.py). The graders focus on:

- risk-adjusted return
- drawdown control
- compliance quality
- information usage
- regime adaptation

## Baselines

Current deterministic benchmark with seed `7`:

```text
guided_allocation        heuristic=0.346  random=0.286
research_risk_conflict   heuristic=0.407  random=0.262
regime_shift_recovery    heuristic=0.421  random=0.327
mandate_drift            heuristic=0.459  random=0.353
```

Expected behavior:

- `heuristic` beats `random` on all four tasks
- all final scores remain in `0.0–1.0`
- `inference.py` emits `[START]`, `[STEP]`, and `[END]` blocks for the validator

## Training Surface

Round 2 training files:

- [`training/committee_grpo_train.py`](./training/committee_grpo_train.py)
- [`training/committee_eval.py`](./training/committee_eval.py)
- [`training/committee_artifacts.py`](./training/committee_artifacts.py)
- [`training/committee_grpo_colab.ipynb`](./training/committee_grpo_colab.ipynb)
- [`docs/ROUND2_TRAINING_RUNBOOK.md`](./docs/ROUND2_TRAINING_RUNBOOK.md)
- [`docs/ROUND2_DEMO_FLOW.md`](./docs/ROUND2_DEMO_FLOW.md)
- [`docs/ROUND2_ONSITE_CHECKLIST.md`](./docs/ROUND2_ONSITE_CHECKLIST.md)
- [`docs/ROUND2_HF_MINI_BLOG_DRAFT.md`](./docs/ROUND2_HF_MINI_BLOG_DRAFT.md)

The training design is intentionally narrow:

- train only the Portfolio Manager
- keep Research and Risk scripted inside the environment
- use TRL `GRPOTrainer` with `environment_factory`
- use optional LoRA adapters via PEFT for practical fine-tuning
- use multiple verifier-style reward functions

Verified Colab smoke result on April 22, 2026:

- runtime: Google Colab `T4 GPU`
- model: `Qwen/Qwen3-0.6B`
- config: `--use-lora --repeats-per-task 2 --max-steps 4`
- baseline summary: heuristic `0.4084`, random `0.2504`
- trainer reward series: `0.0193 -> 0.0160 -> 0.0006 -> 0.0275`
- final reward delta: `+0.0082`

## Running Locally

Use Python 3.11 and the bundled lockfile.

```bash
uv sync --python python3.11 --extra dev
```

Run the committee baselines:

```bash
python3.11 inference.py --policy heuristic
python3.11 inference.py --policy random
python3.11 inference.py --policy heuristic --pretty
```

Run the evaluation helper:

```bash
python3.11 training/committee_eval.py --policy heuristic
python3.11 training/committee_eval.py --policy random
```

Print the Round 2 baseline table used before training:

```bash
python3.11 training/committee_grpo_train.py --print-baselines
```

Build the trainer without starting real training:

```bash
python3.11 training/committee_grpo_train.py --print-baselines --dry-run --use-lora --colab-email anuagar@groww.in
```

Export the baseline report artifact:

```bash
python3.11 training/committee_artifacts.py baselines --output-dir outputs/committee-grpo
```

Export the compact judge-facing report after a training run:

```bash
python3.11 training/committee_artifacts.py report --output-dir outputs/committee-grpo --model-name Qwen/Qwen3-0.6B --colab-email anuagar@groww.in --log-file outputs/committee-grpo/training_log_history.json
```

Export the onsite judge-facing one-pager:

```bash
python3.11 training/committee_artifacts.py demo --output-dir outputs/committee-grpo --model-name Qwen/Qwen3-0.6B --colab-email anuagar@groww.in --log-file outputs/committee-grpo/training_log_history.json
```

The training script also auto-exports `reward_series.json` and `reward_curve.png` after a real GPU run.

For the TRL training flow and Colab commands, see [`docs/ROUND2_TRAINING_RUNBOOK.md`](./docs/ROUND2_TRAINING_RUNBOOK.md).

## LLM Policy and Validator Environment Variables

Validator-facing path:

- `API_BASE_URL`
- `API_KEY`
- `MODEL_NAME`

Local fallback path:

- `HF_TOKEN`
- `OPENAI_API_KEY`

The runtime prefers the injected validator credentials when `API_KEY` is present.

## Server and Deployment

Run the FastAPI environment locally:

```bash
AMC_TASK_ID=guided_allocation uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Available task ids:

- `guided_allocation`
- `research_risk_conflict`
- `regime_shift_recovery`
- `mandate_drift`

Validation:

```bash
openenv validate --verbose
openenv build
openenv validate --url http://localhost:8000
```

## Project Layout

```text
amc_allocator_env/
├── README.md
├── inference.py
├── tasks.py
├── graders.py
├── policies.py
├── models.py
├── data/
├── server/
├── tests/
├── training/
└── docs/
```
