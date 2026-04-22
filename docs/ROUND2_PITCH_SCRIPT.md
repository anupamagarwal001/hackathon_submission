# Round 2 Pitch Script

## 30-Second Version

We built AI Investment Committee Environment, a multi-agent OpenEnv environment where a trainable Portfolio Manager learns to make investment decisions by coordinating with a Research Analyst and a Risk Officer. The environment simulates realistic institutional workflows with noisy signals, hidden market regimes, changing constraints, and measurable rewards. Our goal is to train the PM to improve not just returns, but also risk control, adaptation, and decision quality.

## Full 3-Minute Script

Hi, we built **AI Investment Committee Environment**, a multi-agent OpenEnv environment for training AI systems to behave like a real investment committee.

In real fund management, decisions are not made by one isolated allocator. Research proposes opportunities, risk enforces constraints, and the portfolio manager has to make sequential decisions under uncertainty. Our environment models exactly that workflow.

We define three roles: a **Research Analyst**, a **Risk Officer**, and a **Portfolio Manager**. The Portfolio Manager is the trainable agent. Research provides noisy but useful recommendations. Risk monitors exposure, concentration, and mandate violations. The environment also includes hidden market regimes, transaction costs, and changing constraints.

The goal is not simply to maximize return. The agent must learn to use information well, adapt to regime changes, avoid compliance mistakes, and manage drawdown over a long horizon.

We designed four tasks of increasing difficulty:

- guided allocation
- research-versus-risk conflict
- regime shift recovery
- mandate drift

Each task measures how well the Portfolio Manager coordinates with the other actors and manages the portfolio over time.

Our reward combines return with penalties for drawdown, overtrading, and violations, plus bonuses for using research effectively and responding appropriately to risk signals.

For training, we keep the Analyst and Risk Officer fixed and train only the Portfolio Manager using a minimal HF TRL pipeline. This keeps the environment stable and makes improvement easy to measure.

We evaluate performance before and after training using:

- average reward
- final task score
- drawdown
- compliance violations
- turnover
- regime-shift recovery quality

The core result we want to show is simple: after training, the Portfolio Manager does not just chase returns. It becomes better at coordinating with other agents, handling uncertainty, respecting constraints, and adapting to changes in the market.

So our contribution is not just a stock simulator. It is a realistic, trainable multi-agent workflow environment for financial decision-making.

## Measured Result Version

If you want to use the concrete verified smoke numbers in the pitch, replace the training paragraph with this:

"For training, we keep the Analyst and Risk Officer fixed and train only the Portfolio Manager using a minimal HF TRL plus LoRA pipeline. On a verified T4 Colab smoke run with Qwen3-0.6B, our baseline heuristic scored `0.4084` overall versus `0.2504` for random. In training, the reward started at `0.0193` and ended at `0.0275`, with the best step at step `4`. That gives us a clean positive reward delta in a short on-site-friendly run."

## 2-Minute Backup Version

We built AI Investment Committee Environment, a multi-agent OpenEnv environment for training a Portfolio Manager agent inside a realistic financial workflow.

The key difference from a toy allocator is that the PM does not act alone. It coordinates with a scripted Research Analyst and Risk Officer, while the environment changes through noisy signals, hidden market regimes, and evolving constraints.

We designed four deterministic tasks: guided allocation, research-risk conflict, regime-shift recovery, and mandate drift. The reward is not just return. It also scores compliance, risk response, and information usage, which makes the training signal harder to game and closer to a real professional setting.

For training, we keep only the Portfolio Manager trainable and use a minimal TRL plus LoRA setup. That gives us a stable environment and a clean learning signal. In our verified T4 smoke run, the heuristic baseline scored `0.4084` versus `0.2504` for random, and the training reward improved from `0.0193` to `0.0275`.

So the contribution is a trainable multi-agent workflow world, not just a stock simulator.

## Judge-Friendly Closing Line

This project turns portfolio management into a real multi-agent learning problem instead of a single-step allocation problem.

## Likely Q&A Answers

### Why only train one agent?

Training only the Portfolio Manager keeps the environment stable and gives a clean before/after learning signal. Research and Risk remain fixed environment actors.

### Why is this more than a toy simulator?

Because the PM must coordinate with multiple actors, handle partial observability, adapt to regime shifts, and respect changing constraints over time.

### Why is the reward credible?

Because it is decomposed into verifier-style components: task score, compliance quality, risk response, and information usage. We also use progress-weighting so degenerate short episodes do not win.

### What does improvement mean here?

Higher reward, better task scores, lower drawdown, fewer compliance violations, and better recovery after regime changes.

### Why only show a short training run?

Because for the on-site setting we optimize for a truthful and repeatable signal, not a flashy but unstable run. Our verified 4-step T4 run ends above its starting reward, while a longer 8-step run regressed after peaking mid-run.

### Why is this useful for AI training?

It teaches structured decision-making in a realistic professional workflow rather than isolated one-step prediction.
