# Round 2 Training Runbook

## Goal

Show that the Portfolio Manager can be trained inside the AI Investment Committee environment with:

- OpenEnv-style environment interaction
- TRL as the RL trainer
- optional Unsloth acceleration
- measurable before/after improvement

## Files

- [committee_grpo_train.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/committee_grpo_train.py)
- [committee_eval.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/committee_eval.py)
- [committee_artifacts.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/committee_artifacts.py)
- [committee_grpo_colab.ipynb](/Users/anuagar/Desktop/dev/amc_allocator_env/training/committee_grpo_colab.ipynb)
- [hf_jobs_smoke.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/hf_jobs_smoke.py)
- [launch_hf_job.py](/Users/anuagar/Desktop/dev/amc_allocator_env/training/launch_hf_job.py)

## Training Dependencies

Install these in Colab or the hackathon GPU environment before running the trainer:

```bash
pip install trl datasets accelerate matplotlib peft jmespath "openenv-core[core]>=0.2.2" openai "git+https://github.com/huggingface/transformers.git@main"
```

Optional acceleration layer:

```bash
pip install unsloth
```

## Simple Mental Model

- The Portfolio Manager is the only trainable policy.
- The Research Analyst and Risk Officer are part of the environment.
- TRL calls environment tools through `environment_factory`.
- Reward comes from several independent verifier-style functions.

## Why This Matches The Official Guide

The participant help guide recommends this stack:

- environment
- verifier / reward functions
- TRL trainer
- Unsloth for efficiency
- deployment on OpenEnv / Spaces

That is exactly how this scaffold is structured.

## How To Use It

### 0. Preferred path when using Hugging Face credits

If you claimed the hackathon HF coupon, prefer HF Jobs over Colab for repeatable GPU runs.

Launch the default smoke run:

```bash
python3 training/launch_hf_job.py launch
```

What this does:

- creates an HF Job in namespace `anupamagarwal001`
- uses the public repo mirror as the source of truth
- runs the same PM smoke workflow used in Colab
- defaults to:
- hardware: `t4-small`
- timeout: `2h`
- model: `Qwen/Qwen3-0.6B`
- `--use-lora`
- `--repeats-per-task 2`
- `--max-steps 4`
- artifact upload target:
  - dataset repo: `anupamagarwal001/amc-allocator-job-artifacts`
  - only judge-facing files are uploaded, not full checkpoints

Inspect and stream the job:

```bash
python3 training/launch_hf_job.py inspect <job_id>
python3 training/launch_hf_job.py logs <job_id>
```

List and cancel jobs:

```bash
python3 training/launch_hf_job.py list
python3 training/launch_hf_job.py cancel <job_id>
```

Why this is the right place to spend the `$30` credit:

- it buys repeatable GPU training and eval
- it is cheaper and more targeted than turning the Space into an always-on paid GPU service
- it fits the hackathon requirement of showing real training evidence better than ad hoc notebook-only runs
- it persists reward curves and reports into a dedicated HF dataset repo instead of losing them in ephemeral job logs

### 1. Check baselines first

```bash
python3.11 training/committee_eval.py --policy heuristic
python3.11 training/committee_eval.py --policy random
python3.11 training/committee_grpo_train.py --print-baselines
```

### 2. Run a dry training build

This checks the dataset and trainer construction without launching a real training job.

```bash
python3.11 training/committee_grpo_train.py --print-baselines --dry-run --use-lora --colab-email anuagar@groww.in
```

This now also writes `baseline_report.json` into the chosen output directory so you have a stable before-training artifact.

### 3. Run a tiny Colab / GPU training job

Recommended place:

- Google Colab
- hackathon compute environment

Use this Colab account for the run:

- `anuagar@groww.in`

Example:

```bash
python3 training/committee_grpo_train.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/committee-grpo \
  --repeats-per-task 2 \
  --max-steps 4 \
  --use-lora \
  --print-baselines \
  --colab-email anuagar@groww.in
```

Do not use a CPU runtime for the real `trainer.train()` step. In live Colab testing on April 22, 2026, the Qwen3 run could build successfully on CPU but stalled in the first forward pass and had to be interrupted after about two minutes. Use CPU only for the dry-run and baseline cells.

Recommended smoke configuration after live T4 verification on April 22, 2026:

- `Qwen/Qwen3-0.6B`
- `--use-lora`
- `--repeats-per-task 2`
- `--max-steps 4`

Why `4` and not `8`:

- the `4`-step T4 run ended above its starting reward
- the `8`-step T4 run peaked higher in the middle but regressed by the final step
- for the hackathon demo, the shorter run gives the cleanest "training improved reward" story

## What The Training Script Does

- creates a prompt dataset across the 4 committee tasks
- wraps the committee environment in a TRL `environment_factory` class
- exposes PM actions as tools:
  - query research
  - query risk
  - allocate
  - move to cash
  - hold
- computes reward from multiple functions:
  - final task score
  - compliance quality
  - risk response
  - information usage
- exports:
  - `baseline_report.json`
  - `training_log_history.json`
  - `reward_series.json`
  - `reward_curve.png`
  - plot-ready metric series via `committee_artifacts.py`

## Why Multiple Reward Functions

This follows the official guidance:

- stronger verification
- lower reward-hacking risk
- easier debugging
- easier to explain in the pitch

The current scaffold also weights rewards by episode completion progress, so the PM cannot learn a degenerate policy that stops immediately and collects easy compliance points.

## What To Show Judges

Minimum compelling story:

1. heuristic baseline
2. random baseline
3. one tiny training run
4. before/after evaluation table
5. one reward curve

Verified T4 smoke result worth showing:

- heuristic baseline overall score: `0.4084`
- random baseline overall score: `0.2504`
- T4 `4`-step LoRA reward series: `[0.0193, 0.0160, 0.0006, 0.0275]`
- T4 `4`-step final reward delta: `+0.0082`

Artifact commands:

```bash
python3.11 training/committee_artifacts.py baselines --output-dir outputs/committee-grpo
python3.11 training/committee_artifacts.py series --log-file outputs/committee-grpo/training_log_history.json --output-dir outputs/committee-grpo
python3.11 training/committee_artifacts.py plot --log-file outputs/committee-grpo/training_log_history.json --output-dir outputs/committee-grpo
python3.11 training/committee_artifacts.py report --output-dir outputs/committee-grpo --model-name Qwen/Qwen3-0.6B --colab-email anuagar@groww.in --log-file outputs/committee-grpo/training_log_history.json
python3.11 training/committee_artifacts.py demo --output-dir outputs/committee-grpo --model-name Qwen/Qwen3-0.6B --colab-email anuagar@groww.in --log-file outputs/committee-grpo/training_log_history.json
```

## Current Limitation

This scaffold is intentionally minimal.

It is designed to:

- prove trainability
- create a clean demo path
- keep scope solo-friendly

It is not yet:

- a full-scale training pipeline
- tuned for maximum reward
- optimized with Unsloth by default

## Next Training Step

After this scaffold works in Colab, add:

- optional Unsloth acceleration
- rollout logging to a file
- one saved reward curve plot
- one saved before/after comparison artifact

After claiming HF credits, the next best step is:

1. launch one `t4-small` HF Job smoke run
2. confirm the reward curve and judging report in logs
3. launch one slightly longer run, for example:

```bash
python3 training/launch_hf_job.py launch --max-steps 8 --output-dir outputs/committee-grpo-hf-job-8
```

4. compare the two runs and keep the cleaner one for the demo story
