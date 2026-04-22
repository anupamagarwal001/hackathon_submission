# Round 2 Onsite Checklist

This checklist is optimized for the actual hackathon constraints:

- solo participation
- 3-minute pitch + 2-minute Q&A
- minimal training script expected
- Hugging Face mini-blog or sub-2-minute video expected
- latest OpenEnv release / working environment expected

## Before Travel

### Repo and code

- push the latest Round 2 repo state to GitHub
- make sure the repo root has:
  - environment code
  - training notebook
  - docs
  - README with exact entry points
- keep one known-good commit hash written down
- do not commit large output folders or local checkpoints

### Hugging Face

- confirm you can log in from the browser
- keep one working HF write token ready
- keep the existing Space available as a backup demo surface
- prepare the HF mini-blog draft locally so publishing onsite is fast

### Google Colab / accounts

- use `anuagar@groww.in`
- confirm Colab can access GPU runtime
- keep the notebook link handy
- keep the T4 smoke config handy:
  - model: `Qwen/Qwen3-0.6B`
  - `--use-lora`
  - `--repeats-per-task 2`
  - `--max-steps 4`

### Event logistics

- carry government ID
- carry company ID used at registration
- keep the confirmation email available on phone and laptop
- keep Discord installed and logged in

## What To Show If Time Is Short

Minimum viable demo:

1. explain the three committee roles
2. show the four tasks
3. show heuristic vs random baseline gap
4. show the reward curve from the T4 smoke run
5. close on why this is a real multi-agent workflow environment

## Onsite Build Order

### Step 1: environment sanity

- run the baseline evaluation first
- verify heuristic still beats random
- verify the environment still loads cleanly

### Step 2: training proof

- run the 4-step T4 LoRA smoke config first
- generate:
  - `baseline_report.json`
  - `training_log_history.json`
  - `reward_series.json`
  - `reward_curve.png`
  - `judging_report.md`
  - `onsite_demo_summary.md`

### Step 3: decide whether to scale

Only scale beyond the smoke run if:

- reward remains stable
- outputs are already saved
- you still have enough time for pitch prep

If training gets worse with longer runs, keep the shorter positive run and present that honestly.

### Step 4: final packaging

- publish HF mini-blog or record the short video
- keep one tab with:
  - repo
  - notebook
  - reward curve
  - onsite demo summary

## GitHub Setup Needed

- push the repo before travel
- optionally create a clean `round2` branch if you want isolation
- keep README updated to the on-site story

You do **not** need:

- a complex PR workflow
- large artifact commits
- model checkpoints on GitHub

## Hugging Face Setup Needed

- active account login
- working write token
- mini-blog draft ready
- Space available if you want live hosting

You do **not** need before travel:

- a fresh model release
- fully hosted trained weights
- a new Space unless the current one becomes unusable

## Fallback Rules

If compute is unstable:

- keep the environment demo and baseline table
- keep the previously verified T4 smoke metrics
- present the verified short training run instead of chasing a larger run

If the notebook breaks:

- use the CLI path from `training/committee_grpo_train.py`
- generate artifacts directly into `outputs/...`

If training regresses:

- show the best-step improvement and explain early overfitting
- do not pretend the longer run is better if the final metric is worse

## Final Onsite Success Condition

You leave the venue with:

- one working environment demo
- one short successful training run
- one reward curve
- one judging summary markdown
- one mini-blog or short video
- one practiced 3-minute pitch
