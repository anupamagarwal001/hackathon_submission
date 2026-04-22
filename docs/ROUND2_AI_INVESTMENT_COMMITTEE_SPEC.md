# Round 2 Spec: AI Investment Committee Environment

## 1. One-line Summary

Build a multi-agent OpenEnv environment where a trainable Portfolio Manager agent coordinates with a Research Analyst and a Risk Officer to manage a portfolio through noisy signals, hidden regime shifts, and changing constraints.

## 2. In Simple Terms

Think of this as a small investment firm simulated for AI training.

- The Research Analyst says what looks interesting.
- The Risk Officer says what looks dangerous.
- The Portfolio Manager decides what to do.
- The market changes over time.
- The Portfolio Manager should learn to make better decisions.

Round 1 was a single-agent allocator.

Round 2 evolves that into a more realistic world:

- multiple actors
- imperfect information
- long-horizon decisions
- changing rules
- visible training improvement

## 3. Official Alignment Update

The latest participant update and support docs add a few important constraints and recommendations:

- You may continue with a Round 1-style idea if it still aligns with the Round 2 themes.
- Solo participants must remain solo. No on-ground team formation is allowed.
- The official build stack is: environment -> verifier/reward functions -> TRL trainer -> Unsloth for efficiency -> deployment on OpenEnv / Hugging Face Spaces.
- The official guidance strongly recommends:
  - objective verification
  - multiple independent reward functions
  - reward-hacking prevention
  - curriculum from easier tasks to harder tasks
  - deploying the environment early
  - monitoring actual generations, not just one reward number

This concept remains aligned after those updates.

Why it still fits:

- It extends the Round 1 AMC allocator idea rather than discarding it.
- It is still strongly aligned with the Multi-Agent and World Modeling themes.
- It supports crisp programmatic verification.
- It can be trained with the recommended OpenEnv + TRL + Unsloth stack.

## 4. Why This Idea

This idea fits the Round 2 brief well because it covers three themes at once.

Primary fit:

- Multi-Agent Interactions

Secondary fit:

- World Modeling, Professional Tasks
- Long-Horizon Planning and Instruction Following

Why it is strong:

- It builds on the Round 1 AMC allocator concept instead of starting from zero.
- It feels real and professional instead of toy-like.
- It is easy to explain in a 3-minute pitch.
- It has a clear reward story.
- It gives a clean before/after training story.

## 5. Problem Statement

Modern portfolio decisions are not made by one isolated actor. They emerge from interaction between research, portfolio construction, and risk control. Existing agent benchmarks rarely capture these structured professional workflows. We propose an OpenEnv environment where a Portfolio Manager agent must make sequential investment decisions by interacting with a Research Analyst and Risk Officer while adapting to regime shifts, transaction costs, and changing mandates.

Short version for judges:

"We built a trainable multi-agent investment committee environment where one agent learns to make portfolio decisions by coordinating with research and risk under uncertainty."

## 6. Design Goals

What this environment should prove:

- An AI agent can improve inside a realistic professional workflow.
- Agent performance depends on coordination, not just raw signal following.
- Reward should reflect institutional quality, not only return.
- Training should visibly improve behavior.
- The task should be verifiable enough for RL with verifiable rewards.
- The task should be easy enough initially that the model can achieve non-zero reward.

## 7. Non-Goals

What we are not trying to build:

- a real trading system
- live market execution
- a huge multi-agent society
- a complex front-end product
- a finance-grade risk engine

This hackathon submission should optimize for:

- clear environment design
- strong demo story
- measurable improvement
- fast onsite execution

## 8. Environment Overview

The environment simulates a small institutional portfolio workflow over a sequence of steps.

The world contains:

- 6 to 8 tradable assets
- portfolio NAV and cash
- transaction costs
- latent market regimes
- noisy analyst signals
- changing risk limits
- communication history
- compliance state

Each episode lasts 12 to 20 steps.

At each step:

1. The market state updates.
2. The Portfolio Manager receives an observation.
3. The PM may ask the Analyst or Risk Officer for input.
4. The PM chooses an allocation-related action.
5. The environment simulates execution and updates the portfolio.
6. Reward is computed.
7. The episode continues until the horizon ends.

## 9. Agents

### 9.1 Portfolio Manager

This is the only trainable agent.

Responsibilities:

- decide whether to ask Research for more information
- decide whether to ask Risk for a review
- propose or revise allocations
- decide when to stay defensive in cash
- adapt to changing market conditions

Why this is the trainable agent:

- It is the main decision-maker.
- It is where the most important behavior lives.
- Training one agent is much easier to debug and explain than training three.

### 9.2 Research Analyst

This is a scripted environment actor.

Responsibilities:

- provide asset or sector recommendations
- provide confidence levels
- give short rationales
- sometimes be noisy or wrong

Why keep it scripted:

- It acts like a stable source of advice.
- It keeps the world realistic without increasing training complexity.
- It makes improvement easier to attribute to the PM.

### 9.3 Risk Officer

This is a scripted environment actor.

Responsibilities:

- enforce exposure and concentration limits
- flag risky allocations
- warn about fragility
- request revisions
- penalize violations

Why keep it scripted:

- It acts like a rules-and-constraints layer in the environment.
- It keeps training stable.
- It makes the PM's improvement easier to measure.

## 10. Why Only Train One Agent

This is an important modeling choice.

We train only the Portfolio Manager and keep the other two agents fixed.

Why this is the right choice:

- easier to train within hackathon time limits
- easier to debug
- easier to explain in the pitch
- more stable reward signal
- cleaner before/after comparison

Simple analogy:

- The PM is the student.
- Research and Risk are the teachers and rules of the world.

If all three agents were trainable, it would become hard to answer basic questions like:

- Did reward improve because the PM got better?
- Did reward improve because the Analyst changed behavior?
- Did the training become unstable because all agents changed at once?

For Round 2, clarity matters more than theoretical completeness.

## 11. Action Space

The PM can take one action per step.

High-level actions:

- `query_research(asset_or_sector)`
- `query_risk()`
- `propose_allocation(template_or_weights)`
- `revise_allocation(template_or_weights)`
- `hold`
- `move_to_cash`

### Recommended MVP Representation

Use discrete allocation templates instead of free-form continuous weights in the first version.

Suggested templates:

- overweight top 2 signals
- equal weight top 3
- defensive basket with partial cash
- low-vol basket
- concentrated high-conviction basket
- full cash

Why discrete templates help:

- easier training
- fewer invalid actions
- better sample efficiency
- simpler judging demo

Once the system works, templates can later map to normalized weights internally.

## 12. Observation Space

The PM should see enough to act, but not the hidden truth.

Observation fields:

- current step index
- steps remaining
- current portfolio weights
- cash weight
- portfolio value
- recent asset performance summary
- research notes received so far
- risk notes received so far
- active risk constraints
- previous reward
- turnover from previous step
- market summary features

Important hidden information:

- true market regime
- future returns
- latent fragility that only partially surfaces through Risk

This makes the problem realistic because the PM must infer the state of the world instead of seeing it directly.

## 13. Internal Environment State

The hidden state can include:

- active market regime
- full price path
- all communication history
- portfolio history
- compliance status
- task-specific flags
- whether warnings were ignored

This state is useful for grading and environment transitions even if the PM does not fully observe it.

## 14. Task Suite

Use four tasks with increasing difficulty.

### Task 1: Guided Allocation

Environment characteristics:

- stable regime
- clean analyst signals
- low transaction cost

Goal:

- learn basic coordination and sensible allocation

What good behavior looks like:

- uses research when useful
- allocates into strong signals
- avoids unnecessary churn

### Task 2: Research vs Risk Conflict

Environment characteristics:

- bullish research signals
- elevated underlying fragility
- stricter risk limits

Goal:

- learn judgment under conflicting advice

What good behavior looks like:

- does not blindly trust alpha signals
- requests risk review
- reduces concentration when warned

### Task 3: Regime Shift Recovery

Environment characteristics:

- hidden regime flips mid-episode
- previously successful strategy stops working

Goal:

- detect deterioration and adapt before drawdown becomes too large

What good behavior looks like:

- reduces risk after warning signals
- moves to defensive positions earlier
- avoids prolonged losses from stale beliefs

### Task 4: Mandate Drift

Environment characteristics:

- changing risk budget or allocation limits during the episode
- long-horizon planning pressure

Goal:

- stay aligned with evolving rules over time

What good behavior looks like:

- remembers changed constraints
- revises allocations after policy updates
- balances return with mandate adherence

## 15. Reward Model

The reward should reflect good institutional behavior, not just greed.

Per-step reward:

```text
reward
= return_component
- transaction_cost_penalty
- drawdown_penalty
- compliance_penalty
+ useful_research_bonus
+ timely_risk_response_bonus
```

### Reward Components

`return_component`

- reward for realized portfolio performance

`transaction_cost_penalty`

- discourages overtrading

`drawdown_penalty`

- punishes unstable behavior and slow response to deterioration

`compliance_penalty`

- punishes violating exposure, concentration, or mandate rules

`useful_research_bonus`

- rewards requesting research when it improves decisions

`timely_risk_response_bonus`

- rewards reducing risk when warnings are materially relevant

### Reward Engineering Rules

The official participant guide strongly pushes teams toward verifier-based RL and warns about reward hacking. So this environment should use multiple independent checks, not one single reward term.

Recommended reward/verifier structure:

- portfolio return check
- drawdown control check
- turnover / overtrading check
- compliance / mandate violation check
- information-use check
- anti-cheating or invalid-action check

Important principle:

- Use several smaller independent signals instead of one broad fuzzy reward.

### Anti-Reward-Hacking Safeguards

The PM should not be able to game the environment by:

- using invalid allocation templates
- bypassing risk review through malformed actions
- exploiting hidden state leakage
- spamming research queries without cost
- repeating meaningless actions to farm reward

Guardrails:

- strict action validation
- per-step query cost or budget
- timeout / step caps
- invalid-action penalties
- held-out evaluation scenarios
- manual inspection of sampled rollouts during training

## 16. Final Evaluation Score

Map episodes to a bounded `0.0-1.0` score.

Recommended episode-level weighting:

- 40% risk-adjusted return
- 25% drawdown and compliance quality
- 20% adaptation after regime changes
- 15% decision efficiency and information use

This gives judges a clean story:

- returns matter
- risk matters
- adaptation matters
- coordination matters

## 17. Baselines

Use three baselines.

### Random PM

- picks actions randomly

Purpose:

- establishes a weak floor

### Heuristic PM

- uses rule-based signal following with simple risk overrides

Purpose:

- gives a strong non-learning baseline

### Trained PM

- the actual learning result using TRL

Purpose:

- shows whether training improves behavior

## 18. Training Plan

### Official Stack Interpretation

The latest help guide frames the stack as:

- environment
- verifier / reward functions
- TRL trainer
- Unsloth for efficiency
- deployment on OpenEnv / Spaces

This means our previous choice needs one refinement:

- TRL remains the core training framework
- Unsloth should be treated as the efficiency layer if we need faster or cheaper fine-tuning onsite

So the clean statement is:

"Use TRL as the main RL training framework, and add Unsloth as the acceleration layer when running the actual onsite training jobs."

### Why TRL Still Stays Central

Use HF TRL instead of Unsloth for the main demo because the judging language is about:

- reward improvement
- training loop
- before/after behavior
- measurable progress

TRL fits that story more naturally.

In simple terms:

- TRL is the trainer.
- Unsloth is the speed and memory helper.

That matters in a hackathon demo.

### Why Train Only the Portfolio Manager

Train only the PM.

Keep Research and Risk scripted.

This gives:

- stable environment actors
- cleaner reward attribution
- faster iteration
- easier debugging
- clearer demo story

### Training Stages

#### Stage 1: Benchmark

Run random and heuristic PMs first.

Collect:

- average reward
- task scores
- drawdown
- compliance violations
- turnover

#### Stage 2: Optional Warm Start

If needed, collect good heuristic traces and use them as light supervised warm-start examples.

This is optional but useful if direct RL is too unstable early.

#### Stage 3: TRL-Based Training

Train the PM against environment reward.

Goal:

- increase average task reward
- improve episode scores
- reduce violations and avoidable drawdowns

#### Stage 4: Held-Out Evaluation

Evaluate the trained PM on scenarios not used in the quick training run.

Compare:

- before training
- after training

### Curriculum Strategy

The official guide strongly recommends starting easy and only increasing difficulty after the model starts earning non-zero reward.

So training should follow this task order:

1. Guided Allocation
2. Research vs Risk Conflict
3. Regime Shift Recovery
4. Mandate Drift

Meaning:

- do not start from the hardest task
- first make success possible
- then increase difficulty

### What the Minimal Training Script Should Demonstrate

The notebook or Colab should show:

- environment loads correctly
- reward functions are callable and visible
- at least one short training run completes
- before/after evaluation exists
- sampled rollouts can be inspected

This matches the official guidance better than just "loss went down."

## 19. Minimal Training Script Requirement

The Round 2 brief says to show a minimal training script.

What that means in practice:

- load the environment
- load a small model
- run a short training loop
- log rewards
- evaluate before and after

This does not mean:

- huge training jobs
- perfect convergence
- production infra

It means:

"Prove that your environment can actually be used to train an agent."

## 20. Deployment Strategy

The official guide explicitly recommends deploying early.

Why:

- catch API and packaging bugs before training
- let teammates or judges interact with the same environment artifact
- make demo preparation easier

So the environment plan should be:

1. local reset/step verification
2. remote deployment to Space
3. tiny training run
4. bigger training only after the environment is stable

## 21. Metrics to Show Judges

The strongest metrics are:

- average episode reward
- final task score
- max drawdown
- compliance violation count
- average turnover
- regime-shift recovery score

Best evidence pack:

- one reward curve
- one before/after evaluation table
- one side-by-side rollout example

Also monitor during training:

- overall reward
- each reward component separately
- timeout / invalid action frequency
- actual sampled behaviors

This comes directly from the official build guide and FAQ emphasis on not trusting a single scalar.

## 22. Deliverables

Round 2 should produce:

- a working OpenEnv environment
- a minimal Colab notebook using HF TRL and optionally Unsloth
- one chart showing reward improvement
- one evaluation table
- one short Hugging Face mini-blog or sub-2-minute video
- one 3-minute pitch

## 23. Participation Constraints

Practical constraints from the latest update:

- you are participating as Solo
- no on-ground team formation is allowed
- only registered participants will be allowed on campus
- results will be announced later via livestream, not at the venue

Implication for execution:

- keep the project scoped for one primary builder
- avoid designs that assume extra onsite teammates
- prioritize clean, reliable, explainable progress over ambitious breadth

## 24. Judging Alignment

How this concept maps to the published Round 2 judging criteria:

### Environment Innovation (40%)

- realistic investment committee workflow
- multi-agent interaction
- professional world model
- hidden regime and mandate dynamics

### Storytelling (30%)

- easy to explain with three roles
- obvious real-world analogy
- strong before/after demo potential

### Showing Improvement in Rewards (20%)

- PM before vs PM after is easy to show
- clean metrics and reward curves

### Reward and Training Script/Pipeline (10%)

- reward has business logic
- TRL training loop is easy to demonstrate

## 25. Risks

Main project risks:

- action space becomes too large
- reward becomes too complicated
- no visible improvement in time
- overbuilding multi-agent behavior
- spending too much time on UI instead of training story
- reward hacking or brittle verifier design
- zero-reward training because tasks start too hard

## 26. Mitigations

Recommended responses:

- keep action space discrete in MVP
- train only one agent
- keep reward interpretable
- keep tasks deterministic enough for fast evaluation
- optimize for one strong chart and one clear demo
- use multiple independent reward checks
- add anti-cheat / invalid-action penalties
- start with easier tasks before scaling difficulty

## 27. Implementation Roadmap

### Phase 1: Spec Lock

Finalize:

- agent roles
- task definitions
- action templates
- reward decomposition
- evaluation metrics

### Phase 2: Environment Build

Implement:

- committee-style workflow
- analyst and risk responses
- hidden regime engine
- task configs
- graders
- reward component breakdowns
- anti-hacking checks
- query costs and step limits

### Phase 3: Baselines

Implement:

- random PM
- heuristic PM
- evaluation harness

### Phase 4: Training Notebook

Implement:

- Colab-ready TRL script
- Unsloth integration only if needed for efficiency
- before/after evaluation
- reward logging
- sampled rollout inspection

### Phase 5: Demo Prep

Prepare:

- one reward chart
- one results table
- one sample rollout comparison
- final pitch

### Recommended Solo Timeline

Before travel:

- lock spec
- finish environment MVP
- finish reward functions
- run heuristic baselines
- prepare notebook skeleton

Day 1 onsite:

- verify deployment
- run tiny training first
- inspect generations
- fix reward bugs before scaling

Day 2 onsite:

- scale only after stable improvement
- collect final metrics
- prepare demo and pitch

## 28. Final Positioning Line

Use this sentence repeatedly:

"This is not just a portfolio simulator. It is a trainable multi-agent investment workflow environment where an AI Portfolio Manager learns to coordinate with Research and Risk under uncertainty."
