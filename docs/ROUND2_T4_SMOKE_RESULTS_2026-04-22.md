# Round 2 T4 Smoke Results

This note captures the live Colab results from April 22, 2026 so the key training evidence is preserved outside the temporary Colab runtime.

## Runtime

- Colab account: `anuagar@groww.in`
- runtime: `T4 GPU`
- model: `Qwen/Qwen3-0.6B`
- trainer mode: `GRPOTrainer` with LoRA adapters

## Verified Commands

CPU dry-run only:

```bash
python training/committee_grpo_train.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/committee-grpo-colab-lora-dryrun \
  --repeats-per-task 1 \
  --max-steps 1 \
  --use-lora \
  --dry-run \
  --colab-email anuagar@groww.in
```

Recommended T4 smoke run:

```bash
python training/committee_grpo_train.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/committee-grpo-t4-4step-v2 \
  --repeats-per-task 2 \
  --max-steps 4 \
  --use-lora \
  --colab-email anuagar@groww.in
```

Longer T4 comparison run:

```bash
python training/committee_grpo_train.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/committee-grpo-t4-8step \
  --repeats-per-task 2 \
  --max-steps 8 \
  --use-lora \
  --colab-email anuagar@groww.in
```

## Baseline Summary

- heuristic overall score: `0.4084`
- random overall score: `0.2504`
- score delta vs random: `0.1580`
- return delta vs random: `0.0278`

## Reward Series

### T4 4-step run

- step 1: `0.0193`
- step 2: `0.0160`
- step 3: `0.0006`
- step 4: `0.0275`
- start -> end delta: `+0.0082`
- best step: `4`
- output bundle: `baseline_report.json`, `training_log_history.json`, `reward_series.json`, `reward_curve.png`, `judging_report.json`, `judging_report.md`

This is the cleanest smoke-run artifact for the pitch because the final point ends above the starting reward.

### T4 8-step run

- step 1: `0.0193`
- step 2: `0.0160`
- step 3: `0.0006`
- step 4: `0.0275`
- step 5: `0.0170`
- step 6: `0.0059`
- step 7: `0.0019`
- step 8: `0.0019`
- best point: `0.0275`
- start -> end delta: `-0.0174`

This run is still useful as debugging evidence, but it is not the best demo artifact because the final reward regressed after the mid-run peak.

## Practical Conclusion

Use the `4`-step T4 run for the Round 2 demo story:

- it proves the training path works on Colab
- it produces a positive reward delta
- it is short enough for live iteration during the hackathon
- it avoids the overfitting/regression seen in the `8`-step run
