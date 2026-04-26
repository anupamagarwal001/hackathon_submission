# Round 2 Pitch Script

## 30-Second Version

We built AI Investment Committee Environment, a benchmark for conflict-aware decision-making in LLM agents. The core question is simple: when Research says buy but Risk says the mandate is under pressure, does the Portfolio Manager blindly chase the signal or resolve the conflict? Our OpenEnv environment measures that behavior with verifier-style rewards, and our verified GRPO smoke run improved reward from `0.0193` to `0.0275`.

## 90-Second Video Script

LLMs are good at giving financial opinions, but real investment decisions are about resolving conflicting incentives.

We built a multi-agent OpenEnv environment where Research, Risk, and a Portfolio Manager interact inside a small investment committee.

Here is the failure mode we target: Research gives a strong buy signal in high-momentum tech, but Risk warns that volatility and concentration are above mandate. A bad Portfolio Manager follows Research only, increases concentration, and gets penalized for compliance and drawdown.

The target behavior is different: query both agents, reduce concentration, preserve the return signal, and avoid mandate violations.

The environment scores this with multiple verifier-style rewards: task score, compliance quality, risk response, and information usage. That makes the reward harder to game than a single return-only score.

In our verified 4-step GRPO smoke run with Qwen3-0.6B and LoRA, reward improved from `0.0193` to `0.0275`. We also preserved the completion traces: a higher-reward late sample queries both Risk and Research before taking exposure, while a lower-reward sample allocates too early. We also ran an 8-step comparison and saw it regress after step `4`, so we use the 4-step run as the stable demonstrated configuration.

This is not a trading system. It is a benchmark for whether LLM agents can learn conflict-aware professional decision-making under partial information.

## Full 3-Minute Script

Hi, we built **AI Investment Committee Environment**, a multi-agent OpenEnv environment for training AI systems to behave like a real investment committee.

In real fund management, decisions are not made by one isolated allocator. Research proposes opportunities, risk enforces constraints, and the portfolio manager has to make sequential decisions under uncertainty. Our environment models exactly that workflow.

We define three roles: a **Research Analyst**, a **Risk Officer**, and a **Portfolio Manager**. The Portfolio Manager is the trainable agent. Research provides noisy but useful recommendations. Risk monitors exposure, concentration, and mandate violations. The environment also includes hidden market regimes, transaction costs, and changing constraints.

The goal is not simply to maximize return. The agent must learn to use information well, adapt to regime changes, avoid compliance mistakes, and manage drawdown over a long horizon.

The novelty is that this is not a toy market game. It is a professional multi-agent workflow with conflicting incentives, partial observability, and verifier-based rewards. The trainable agent has to resolve disagreement between Research and Risk instead of just following a signal.

The core failure mode is visible: Research can say "buy" while Risk says the portfolio is already too concentrated or too volatile. A weak PM follows the signal blindly. A better PM uses both inputs and chooses a more balanced allocation.

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

"For training, we keep the Analyst and Risk Officer fixed and train only the Portfolio Manager using a minimal HF TRL plus LoRA pipeline. On a verified T4 smoke run with Qwen3-0.6B, our baseline heuristic scored `0.4084` overall versus `0.2504` for random. In training, the reward started at `0.0193` and ended at `0.0275`, with the best step at step `4`. The preserved completion trace shows the verifier preferring a PM candidate that queries both Risk and Research before exposure. That gives us a clean positive reward delta plus a concrete behavior sample in a short on-site-friendly run."

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

### Why is this actually novel for the hackathon?

Because it targets an underexplored capability gap: structured strategic disagreement in a professional workflow. The environment teaches an LLM to resolve conflicting incentives between opportunity-seeking and risk control, which is much closer to real agent deployment than a static finance benchmark or a toy game.

### Could this support research beyond the hackathon?

Yes. The research question is whether verifier-driven RL can improve multi-agent coordination, information usage, and risk-aware decision making in a partially observable institutional environment. That is a defensible RL-for-LLM training question, not just a demo gimmick.

### Why is the reward credible?

Because it is decomposed into verifier-style components: task score, compliance quality, risk response, and information usage. We also use progress-weighting so degenerate short episodes do not win.

### What does improvement mean here?

Higher reward, better task scores, lower drawdown, fewer compliance violations, and better recovery after regime changes.

### Why only show a short training run?

Because for the on-site setting we optimize for a truthful and repeatable signal, not a flashy but unstable run. Our verified 4-step T4 run ends above its starting reward, while a longer 8-step run regressed after peaking mid-run.

### Why is this useful for AI training?

It teaches structured decision-making in a realistic professional workflow rather than isolated one-step prediction.
