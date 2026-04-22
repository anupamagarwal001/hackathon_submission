# Round 2 Alignment Update

This note aligns our Round 2 plan with the latest finalist updates and participant support documents received on April 22, 2026.

## What Changed

### 1. We are allowed to continue the Round 1 idea

Official guidance says participants may continue with their Round 1 problem statement if it aligns with the Round 2 themes.

Our decision:

- keep the AMC allocator foundation
- evolve it into a multi-agent investment committee environment

This is now the official continuity story:

- Round 1: single-agent allocator
- Round 2: multi-agent institutional workflow

### 2. Solo remains solo

Official guidance says solo participants cannot form teams on the ground.

Our implication:

- keep the project scoped for one main builder
- do not depend on separate teammate roles for environment, reward, training, and demo
- optimize for the smallest system that still tells a strong story

### 3. The official stack is clearer now

The help guide frames the stack as:

- environment
- verifier / reward functions
- TRL trainer
- Unsloth for efficiency
- deployment on OpenEnv / Hugging Face Spaces

Our updated interpretation:

- OpenEnv remains the environment interface
- TRL is the main training framework
- Unsloth is optional but recommended as a speed and memory layer

### 4. Reward engineering is now more important than before

The self-serve guide strongly emphasizes:

- multiple independent reward functions
- objective verification
- anti-reward-hacking design
- inspecting model generations

Our implication:

- reward design must be treated as a first-class system
- we should not rely on one broad scalar
- we should expose reward components clearly in logs and evaluation

### 5. Build order matters

The official recommendation is:

1. stabilize environment
2. deploy early
3. run tiny training
4. inspect outputs
5. scale only after the loop is stable

Our implication:

- no large training before the environment is stable
- no fancy scaling before we have sane rewards
- demoability and verifiability come before training scale

## What We Should Not Change

We should not abandon the investment idea.

Why:

- it still fits the allowed themes
- it still feels professional and realistic
- it gives us a strong story rooted in your original idea
- the new guidance does not invalidate it

## What We Should Change

### Change 1: Be explicit about verifiers

Old thinking:

- reward is a sensible weighted formula

Updated thinking:

- reward must be broken into multiple independent checks
- these checks should be inspectable and hard to game

### Change 2: Treat Unsloth as a practical add-on, not a competitor to TRL

Old thinking:

- choose TRL over Unsloth

Updated thinking:

- use TRL as trainer
- use Unsloth if it helps us train faster onsite

### Change 3: Start easier

Old thinking:

- build full multi-agent difficulty quickly

Updated thinking:

- start with easy guided tasks
- only add hard regime and mandate tasks after getting non-zero reward

### Change 4: Prioritize solo execution realism

Old thinking:

- broad multi-agent ambition

Updated thinking:

- the design must be narrow enough that one person can finish and present it well

## Final Aligned Position

We will build:

AI Investment Committee Environment

Definition:

A multi-agent OpenEnv environment where a trainable Portfolio Manager learns to coordinate with a scripted Research Analyst and scripted Risk Officer to make sequential portfolio decisions under uncertainty.

Updated implementation principles:

- one trainable agent
- multiple reward/verifier components
- curriculum from easy to hard
- deploy environment early
- inspect sampled rollouts
- scale training only after stable improvement

## Immediate Next Steps

1. Convert the current single-agent allocator into a committee-style environment skeleton.
2. Define discrete PM action templates.
3. Add scripted Research and Risk response logic.
4. Break reward into separate verifier-style components.
5. Create a minimal TRL notebook, with optional Unsloth integration later.

