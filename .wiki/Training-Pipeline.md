# Training Pipeline

Use this page when running, skipping, or debugging training stages.

## Stage Order

```text
CPT -> Fact-SFT -> DPO -> GRPO -> merge -> heuristic quality eval
```

The pipeline produces PEFT adapters first, then merges the selected adapter into a full Hugging Face model.

Each adapter also stores a self-contained `adapter_provenance.json`. Merge reports derive the CPT, Fact-SFT, DPO, and GRPO stage chain from it, with recursive legacy-metadata fallback for older adapters. A failed quality gate retains training and merge artifacts but makes the full pipeline exit `8`, with `quality_gate_failed` and `release_ready: false` in the report.

## Full Pipeline

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

DPO and GRPO are enabled by default. Use a Git-ignored local YAML for the live external-judge settings; explicitly disable or skip a stage you do not need.

Important outputs:

| Output | When it appears |
|---|---|
| `outputs/lora_adapter/` | CPT adapter. |
| `outputs/fact_sft_adapter/` | Fact-SFT adapter. |
| `outputs/dpo_adapter/` | DPO adapter when `dpo.enabled=true`. |
| `outputs/grpo_adapter/` | GRPO adapter when `grpo.enabled=true`. |
| `outputs/merged_model/` | Merged model. |
| `outputs/cpt_dataset/coverage_report.md` | CPT coverage report. |
| `outputs/eval/eval_report.md` | Post-training quality evaluation report. |
| `outputs/reports/pipeline_report.md` | Pipeline summary report. |

See [Operations Runbook](Operations-Runbook.md) for report interpretation.

## Smoke Test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Use this before real GPU training. It validates config loading, corpus discovery, dataset preparation, PEFT adapter saving, merge, and report generation without downloading the default base model.

## Skip Stages

Stage skipping is explicit:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
```

Use skip flags only when the required upstream adapter already exists or the stage is intentionally disabled.

Common run-control flags:

| Flag | Purpose |
|---|---|
| `--skip_preflight` | Skip corpus safety preflight before CPT. |
| `--allow_unsafe_corpus` | Continue when preflight is blocked; see [Operations Runbook](Operations-Runbook.md). |
| `--skip_merge` | Do not merge after training. |
| `--skip_eval` | Do not run quality evaluation after merge. |
| `--device cpu|cuda|cuda:0|auto` | Override the configured training device. |

## Enable DPO by Configuration

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

DPO input rows require `prompt`, `chosen`, and `rejected`, with `chosen != rejected`. DPO defaults to continuing from `outputs/fact_sft_adapter`.

## Run GRPO Directly

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --max_steps 10
```

This command sets `grpo.enabled` to true and prepares the GRPO dataset before training by default. The external judge is enabled by default while all four built-in rewards are opt-in; judge mode accepts prompt-only rows.

GRPO resolves its starting point as explicit `grpo.base_adapter_dir` -> DPO -> Fact-SFT -> CPT. A candidate root must contain `adapter_config.json` plus either `adapter_model.safetensors` or `adapter_model.bin`; incomplete directories are skipped while fallback continues.

Common GRPO debugging flags:

| Flag | Purpose |
|---|---|
| `--prepare_only` | Prepare `outputs/grpo_dataset` without training. |
| `--train_only` | Train from an already prepared dataset. |
| `--num_generations` | Override completions sampled per prompt. |
| `--max_completion_length` | Override policy candidate length; it does not control the judge response budget. |
| `--base_adapter_dir` | Select the starting adapter for GRPO. |

GRPO samples multiple completions per prompt and applies the default external judge plus explicitly enabled built-in rewards. Reasoning-judge output is controlled by `grpo.reward_judge.max_tokens`; use `4096` tokens and a `120` second timeout as a starting point. See [GRPO And Reward Judge](GRPO-and-Reward-Judge.md).

External scoring is asynchronous within the current rollout batch: after candidate generation, requests run with `min(max_concurrency cap, completion count)`, and the trainer waits for the full batch before computing group advantage and performing synchronized backpropagation and optimizer update. `max_concurrency: "auto"` follows `num_generations`; an explicit integer can be shared across prompts in the same reward batch. It is a per-process/rank semaphore cap, and executor capacity may reduce physical concurrency further. The pipeline does not pre-generate the full dataset or update the model concurrently. Requests retry independently with backoff, valid `Retry-After` on 429/503 is honored, and any final failure fails the complete reward batch.

Continue the pipeline from existing DPO/Fact-SFT/CPT adapters:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

This selects a previous-stage adapter as a new GRPO starting point. `grpo.resume_from_checkpoint` instead restores a checkpoint from the same GRPO Trainer run.

## Real-Run Acceptance

- Use `bf16: auto`, `fp16: auto`, and `torch_dtype: auto` across all four stages; BF16 is preferred when supported.
- Keep `abort_on_nonfinite_grad_norm: true` so non-finite gradients fail immediately.
- Do not rely on finite loss or completed status; compare LoRA tensors across adjacent adapters to confirm real updates.
- Inspect judge reward mean/variance and `completions/clipped_ratio`. For high clipping, improve EOS/stop behavior before raising `max_completion_length`.
- Quality evaluation is a heuristic smoke gate, not production safety certification.

## Merge Behavior

When `merge.adapter_dir` is `null` and no `--adapter_dir` argument is provided, merge chooses among enabled stages:

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

If `grpo.enabled=false`, an existing `outputs/grpo_adapter` is not selected automatically. To merge a specific adapter, pass it explicitly:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.local.yaml --adapter_dir outputs/grpo_adapter
```

Or configure:

```yaml
merge:
  adapter_dir: "outputs/grpo_adapter"
```

## Static Verification

If training dependencies are unavailable:

```bash
python -m compileall pipeline scripts serve_inference.py
```

This verifies syntax only. It does not validate CUDA, dataset loading, model loading, or TRL behavior. See [Operations Runbook](Operations-Runbook.md) for environment diagnostics.
