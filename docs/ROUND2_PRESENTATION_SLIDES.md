# AI Investment Committee Environment: Short Slide Deck Outline

## Slide 1: Problem

- LLMs are still weak at resolving structured disagreement across multiple actors with different incentives.
- Real portfolio decisions are committee decisions, not single-step predictions.
- We want an environment that can train an LLM to ask for information, balance opportunity against risk, and adapt under changing constraints.

## Slide 2: Environment

- OpenEnv environment with three roles:
  - Portfolio Manager (trainable)
  - Research Analyst (scripted)
  - Risk Officer (scripted)
- Hidden market regimes, noisy signals, transaction costs, and mandate updates.
- Four deterministic tasks:
  - guided allocation
  - research-risk conflict
  - regime-shift recovery
  - mandate drift

## Slide 3: Why This Is Novel

- Not a toy game or grid-world clone.
- Multi-agent professional workflow with partially observable incentives.
- The PM must resolve conflicts between research conviction and risk discipline.
- The research question is whether verifier-driven RL can improve strategic coordination inside a realistic investment committee workflow.

## Slide 4: Reward + Training

- Reward is decomposed into verifier-style components:
  - return
  - drawdown
  - compliance
  - information usage
  - risk response
- Training stack:
  - OpenEnv
  - HF TRL GRPO
  - LoRA adapters
  - Qwen/Qwen3-0.6B
  - Colab T4 smoke run

## Slide 5: Results

- Baseline overall score:
  - heuristic `0.4084`
  - random `0.2504`
- Smoke training reward:
  - start `0.0193`
  - end `0.0275`
  - delta `+0.0082`
  - best step `4`
- Key judge-facing artifacts:
  - `reward_curve.png`
  - `judging_report.md`
  - `onsite_demo_summary.md`

## Slide 6: Why It Matters

- This environment teaches structured multi-agent reasoning, not just asset ranking.
- It is useful for training LLMs to operate in professional decision loops with partial observability and changing incentives.
- The result is a trainable, research-grade workflow benchmark rather than a finance-themed toy problem.
