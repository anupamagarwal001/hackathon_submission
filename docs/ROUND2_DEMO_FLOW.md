# Round 2 Demo Flow

This is the practical live-demo sequence to use on campus.

## Goal

Show four things clearly:

1. this is a real multi-agent environment
2. reward is objective and hard to game
3. the Portfolio Manager is trainable
4. the training result is measurable

## Demo Length

Target: `3 minutes`

## Screen Order

### Screen 1: Problem framing

Show:

- repo README or spec summary

Say:

"Round 1 was a single-agent allocator. For Round 2 we turned that into a multi-agent investment committee where the Portfolio Manager learns by coordinating with Research and Risk."

### Screen 2: Environment structure

Show:

- the task list
- PM actions
- reward breakdown

Say:

"The PM can query Research, query Risk, allocate, hold, or move to cash. We score not only returns, but also compliance, risk response, and information usage."

### Screen 3: Baselines

Show:

- `baseline_report.json` or terminal table

Say:

"Before training, the heuristic policy scores `0.4084` overall while random is `0.2504`. So the environment already has a meaningful difficulty and signal structure."

### Screen 4: Training proof

Show:

- `grpo_behavior_sample.svg`
- `reward_curve.png`
- `judging_report.md`

Say:

"We ran a verified T4 LoRA smoke training job with Qwen3-0.6B. The reward started at `0.0193` and ended at `0.0275`, with the best step at step `4`. The preserved completion trace shows why: the verifier gives higher reward when the PM gathers both Risk and Research before taking exposure."

### Screen 5: Why it matters

Show:

- `onsite_demo_summary.md`

Say:

"This environment is useful because it teaches an agent to act in a structured professional workflow, not just to predict one-step allocations."

## If Judges Ask For More Detail

Open:

- [committee_grpo_train.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/committee_grpo_train.py)
- [committee_artifacts.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/committee_artifacts.py)
- [server/amc_environment.py](/Users/anuagar/Desktop/dev/amc_allocator_env/server/amc_environment.py)

Focus on:

- only the PM is trainable
- Research and Risk are scripted environment actors
- reward uses multiple verifier-style components
- training outputs are exported automatically into judge-facing artifacts

## What Not To Waste Time On

Do not spend live demo time on:

- full code walkthroughs
- raw notebook debugging
- long theoretical RL explanations
- long finance explanations

Keep it on:

- world design
- reward design
- measured improvement
- why this is aligned with the hackathon themes

## Fallback Demo

If compute fails onsite:

1. show the environment and task structure
2. show heuristic vs random baselines
3. show the previously verified T4 smoke result
4. explain the exact training config used

That is still a coherent story.
