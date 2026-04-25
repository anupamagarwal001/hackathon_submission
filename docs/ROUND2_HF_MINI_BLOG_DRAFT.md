# AI Investment Committee Environment

## What we built

For the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology Grand Finale, we built **AI Investment Committee Environment**, a multi-agent OpenEnv environment for training a Portfolio Manager agent inside a realistic investment workflow.

Instead of treating portfolio allocation as a single-step prediction problem, we modeled it as a committee process:

- a **Research Analyst** provides noisy but useful views
- a **Risk Officer** enforces constraints and flags fragility
- a trainable **Portfolio Manager** decides when to query, allocate, hold, or move to cash

This turns portfolio management into a proper agent environment with state, actions, delayed outcomes, partial observability, and measurable rewards.

## Why this fits OpenEnv

The environment is built around the OpenEnv interaction model:

- `reset()` starts a fresh episode
- `step(action)` applies one committee decision
- the observation exposes signals, notes, constraints, and portfolio state
- reward is computed from multiple verifier-style components

The design goal was not just to simulate returns, but to capture professional workflow behavior:

- decision quality
- information usage
- risk response
- compliance discipline
- adaptation to regime shifts

## Themes covered

Primary theme:

- **Multi-Agent Interactions**

Secondary themes:

- **World Modeling, Professional Tasks**
- **Long-Horizon Planning**

The environment is intentionally professional rather than game-like. It models a small institutional workflow with changing mandates and hidden market conditions.

## Task ladder

We designed four tasks with increasing difficulty:

1. `guided_allocation`
2. `research_risk_conflict`
3. `regime_shift_recovery`
4. `mandate_drift`

These tasks are deterministic and programmatically scorable, which is important for reinforcement learning with verifier-based rewards.

## Reward design

A strong lesson from RL is that the reward is the task specification. So we avoided one fuzzy score and used multiple reward components:

- task score
- compliance quality
- risk response
- information usage

We also progress-weight the reward so the model cannot game training by ending episodes early or doing nothing.

## Training setup

To keep the environment stable and the results interpretable, we train only the **Portfolio Manager**. The Research Analyst and Risk Officer remain scripted environment actors.

This gives us:

- cleaner reward attribution
- easier debugging
- a clearer before/after story

For the minimal training script, we used:

- **TRL**
- **LoRA adapters**
- **Qwen/Qwen3-0.6B**
- a short Google Colab **T4 GPU** smoke run

## Baseline and training results

Deterministic baseline snapshot:

- heuristic overall score: `0.4084`
- random overall score: `0.2504`
- score delta vs random: `0.1580`

Verified T4 smoke training run:

- reward start: `0.0193`
- reward end: `0.0275`
- best step: `4`
- final delta: `+0.0082`

We also tested a longer `8`-step run and observed regression after the mid-run peak. That was useful because it gave us a realistic picture of where the smoke configuration is stable and where it starts to overfit.

## Why this matters

This project is not a trading system. It is a **trainable multi-agent workflow environment**.

The point is to help LLMs learn behaviors that matter in real professional settings:

- asking for the right information
- balancing opportunity with risk
- respecting constraints
- adapting over time

That is why we think the environment is a strong fit for OpenEnv and for the broader direction of RL post-training for agent systems.

## Why this is a fresh angle

The hackathon theme is intentionally open-ended, and the judges explicitly warn against well-worn toy benchmarks. We leaned into that.

- This is not a board game, a grid world, or a synthetic negotiation toy.
- The environment teaches a real capability gap: handling conflicting incentives across multiple roles inside a partially observable workflow.
- The domain is underexplored in RL-for-LLM training. Plenty of environments teach solving a puzzle; far fewer teach a model how to operate inside a professional committee that must trade off opportunity, compliance, and regime uncertainty.
- There is a clear research question here: can verifier-driven RL improve strategic coordination and risk-aware decision making in a realistic investment workflow?

## Repo and training artifacts

The repo contains:

- the OpenEnv environment
- deterministic task suite
- GRPO training scaffold
- Colab notebook
- artifact exporters for judge-facing summaries

The main artifacts used in the demo are:

- `baseline_report.json`
- `training_log_history.json`
- `reward_curve.png`
- `judging_report.md`
- `onsite_demo_summary.md`

## Next step

The next step is to scale beyond the smoke run while preserving the same properties:

- objective evaluation
- multi-agent realism
- stable reward behavior
- clear before/after improvement

## Links

- Hugging Face Space: [anupamagarwal001/amc_allocator_env](https://huggingface.co/spaces/anupamagarwal001/amc_allocator_env)
- Public GitHub mirror: [anupamagarwal001/hackathon_submission](https://github.com/anupamagarwal001/hackathon_submission)
- Slide deck: [`AI_Investment_Committee_Deck.pptx`](./AI_Investment_Committee_Deck.pptx)
- Pitch script: [`docs/ROUND2_PITCH_SCRIPT.md`](./ROUND2_PITCH_SCRIPT.md)
- Slide outline: [`docs/ROUND2_PRESENTATION_SLIDES.md`](./ROUND2_PRESENTATION_SLIDES.md)
