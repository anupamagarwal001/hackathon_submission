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

## Links

- Hugging Face Space: [anupamagarwal001/amc_allocator_env](https://huggingface.co/spaces/anupamagarwal001/amc_allocator_env)
- Live app: [anupamagarwal001-amc-allocator-env.hf.space](https://anupamagarwal001-amc-allocator-env.hf.space)
- Public GitHub mirror: [anupamagarwal001/hackathon_submission](https://github.com/anupamagarwal001/hackathon_submission)
- Colab training notebook: [`training/committee_grpo_colab.ipynb`](./training/committee_grpo_colab.ipynb)
- HF Jobs launcher: [`training/launch_hf_job.py`](./training/launch_hf_job.py)
- Mini-blog draft: [`docs/ROUND2_HF_MINI_BLOG_DRAFT.md`](./docs/ROUND2_HF_MINI_BLOG_DRAFT.md)
- Short slide deck: [`docs/AI_Investment_Committee_Deck.html`](./docs/AI_Investment_Committee_Deck.html)
- Pitch script: [`docs/ROUND2_PITCH_SCRIPT.md`](./docs/ROUND2_PITCH_SCRIPT.md)
- Demo flow: [`docs/ROUND2_DEMO_FLOW.md`](./docs/ROUND2_DEMO_FLOW.md)
- On-site checklist: [`docs/ROUND2_ONSITE_CHECKLIST.md`](./docs/ROUND2_ONSITE_CHECKLIST.md)
- Presentation outline: [`docs/ROUND2_PRESENTATION_SLIDES.md`](./docs/ROUND2_PRESENTATION_SLIDES.md)

## Overview

- committee-style environment, not a single-step toy allocator
- one trainable Portfolio Manager with scripted Research and Risk actors
- four deterministic Round 2 tasks with hidden market regimes and changing constraints
- typed OpenEnv `Action`, `Observation`, and extended `State`
- root-level `inference.py` for validator-compatible structured output
- minimal TRL training scaffold for the Portfolio Manager in [`training/`](./training)

## Why This Is A Fresh Theme-1 Environment

The judges explicitly ask whether the environment teaches an LLM something it currently cannot do well, whether the domain is underexplored, and whether the setup could support research. This environment is built around those questions.

- **Not a game clone:** the agent is not solving chess, snake, or a grid world. It is learning an institutional workflow with conflicting incentives.
- **Real multi-agent dynamics:** the Portfolio Manager depends on Research for opportunity discovery and on Risk for mandate enforcement. Those two actors can disagree, which forces negotiation-like behavior rather than single-step prediction.
- **Partially observable incentives:** the PM never sees the hidden market regime directly, and it cannot optimize purely for return because constraint pressure and risk alerts change the value of each action.
- **Research-worthy framing:** the core problem is whether verifier-driven RL can train an LLM to resolve strategic disagreement under changing incentives in a professional workflow. That is a more interesting research question than “can an allocator follow a signal.”

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

Multi-seed on-site baseline snapshot from the verified Colab run:

| policy | score | total_return | max_drawdown | compliance_score | information_usage | risk_response |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | 0.4084 | 0.0217 | 0.0153 | 0.6611 | 0.2716 | 0.0181 |
| random | 0.2504 | -0.0061 | 0.0209 | 0.7584 | 0.0436 | 0.0307 |

![Deterministic baseline comparison](./docs/assets/baseline_score_comparison.png)

The heuristic does not dominate every component. That is intentional. Random can be slightly more conservative on compliance-only dimensions, but it loses on the objective that matters: the multi-objective overall score produced by return, drawdown, information usage, and risk-aware decision quality together.

## Training Evidence

The current training story is a real smoke run, not a mocked chart:

- environment: `AI Investment Committee Environment`
- runtime: Google Colab `T4 GPU`
- model: `Qwen/Qwen3-0.6B`
- trainer: HF `TRL` `GRPOTrainer`
- adapter path: `LoRA`
- config: `use_lora=True`, `repeats_per_task=2`, `max_steps=4`

Verified outputs from the run:

- reward start: `0.0193`
- reward end: `0.0275`
- reward delta: `+0.0082`
- best step: `4`
- committed artifacts:
  - `baseline_report.json`
  - `training_log_history.json`
  - `judging_report.json`
  - `judging_report.md`
  - `onsite_demo_summary.md`
  - `reward_series.json`
  - `reward_curve.png`

![Smoke training reward curve](./docs/assets/reward_curve.png)

The point of this run is not to claim a fully converged PM. The point is to show end-to-end trainability: a real environment, real verifier-style rewards, a real trainer, and measurable positive movement in a repeatable short-horizon on-site run.

## Training Surface

Round 2 training files:

- [`training/committee_grpo_train.py`](./training/committee_grpo_train.py)
- [`training/committee_eval.py`](./training/committee_eval.py)
- [`training/committee_artifacts.py`](./training/committee_artifacts.py)
- [`training/committee_grpo_colab.ipynb`](./training/committee_grpo_colab.ipynb)
- [`training/hf_jobs_smoke.py`](./training/hf_jobs_smoke.py)
- [`training/launch_hf_job.py`](./training/launch_hf_job.py)
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

To regenerate the committed README plots from the verified smoke metrics:

```bash
python3 training/generate_readme_assets.py
```

To spend the claimed HF credits on a repeatable GPU smoke run instead of Colab:

```bash
python3 training/launch_hf_job.py launch
```

This launches the default `t4-small` PM smoke training job through Hugging Face Jobs using the public repo mirror.

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

## Supporting Materials

Judge-facing support artifacts linked from this README:

- mini-blog draft: [`docs/ROUND2_HF_MINI_BLOG_DRAFT.md`](./docs/ROUND2_HF_MINI_BLOG_DRAFT.md)
- slide deck: [`docs/AI_Investment_Committee_Deck.html`](./docs/AI_Investment_Committee_Deck.html)
- pitch and Q&A script: [`docs/ROUND2_PITCH_SCRIPT.md`](./docs/ROUND2_PITCH_SCRIPT.md)
- on-site demo flow: [`docs/ROUND2_DEMO_FLOW.md`](./docs/ROUND2_DEMO_FLOW.md)
- on-site execution checklist: [`docs/ROUND2_ONSITE_CHECKLIST.md`](./docs/ROUND2_ONSITE_CHECKLIST.md)
- short slide-deck outline: [`docs/ROUND2_PRESENTATION_SLIDES.md`](./docs/ROUND2_PRESENTATION_SLIDES.md)

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
