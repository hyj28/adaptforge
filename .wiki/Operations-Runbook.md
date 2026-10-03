# Operations Runbook

Use this page when moving from “installed” to “operable”: stable training, failure diagnosis, and recovery.

## Pre-Training Environment Diagnostics

After installing dependencies, run:

```bash
python scripts/diagnostics/check_training_environment.py
```

Expected output includes:

- `python`
- `torch`
- `transformers`
- `peft`
- `datasets`
- `trl`
- `cuda_available`
- `torch_cuda`
- `gpu[...]`
- `nvidia-smi`

How to interpret it:

| Output | Meaning | Action |
|---|---|---|
| `torch: not installed` or missing core packages | The active environment does not have training dependencies | Reactivate the environment and install `requirements.txt`. |
| `torch import failed` | PyTorch cannot be imported; the script exits with code `2` | Install the PyTorch build that matches Python and CUDA. |
| `cuda_available: False` | PyTorch cannot see CUDA | Do not start real training; check driver, CUDA wheel, virtualenv, and `nvidia-smi`. CPU is only for smoke tests. |
| No `gpu[...]` lines | No visible GPU or CUDA is unavailable | Fix the environment before training. |
| `nvidia-smi: not found` | Driver utility is unavailable or not in PATH | Confirm NVIDIA driver installation and PATH. |

## Prepare the Base Model

The default config uses:

```yaml
base_model_repo_id: "Qwen/Qwen3.5-0.8B"
base_model_name_or_path: "models/base-model"
```

`base_model_repo_id` is the download source; `base_model_name_or_path` is what training, merge, and inference actually load.

Download the configured Hugging Face model snapshot:

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
```

Expected result: `models/base-model/` contains Hugging Face model files such as `config.json`, tokenizer files, and safetensors weights.

For private models, authenticate through Hugging Face before downloading. Do not put tokens in config files:

```bash
huggingface-cli login
```

If you already have a local snapshot, point the config at it:

```yaml
base_model_name_or_path: "D:/models/my-base-model"
```

You can also override the download target:

```bash
python scripts/model_artifacts/download_models.py --model_id Qwen/Qwen3.5-0.8B --local_dir models/base-model
```

## Live Training Config Preflight

Run real training with the Git-ignored `configs/domain_post_training.local.yaml`. DPO, GRPO, and the external judge are enabled by default; confirm that both datasets exist and configure judge `base_url`, `model`, and plaintext `api_key`. Environment variables are only an optional fallback when `api_key` is empty. The default `max_concurrency: "auto"` follows `num_generations`, and `retry_backoff_seconds: 1.0` is the base delay for independent exponential retries.

Keep these settings across all four stages:

```yaml
bf16: auto
fp16: auto
torch_dtype: auto
abort_on_nonfinite_grad_norm: true
```

Auto mode prefers BF16 when supported and otherwise falls back to FP16 on CUDA. Still confirm the effective precision and finite gradients in logs and metadata.

## Where to Look After a Failure

For full-pipeline failures, check:

```text
outputs/reports/failure_report.md
outputs/reports/pipeline_report.md
outputs/logs/preflight_report.md
outputs/logs/discovered_corpus.json
```

Common stage exit codes:

| Exit code | Source | Meaning |
|---|---|---|
| `2` | `check_training_environment.py` | PyTorch import failed. |
| `4` | CPT training | CPT training failed. |
| `5` | adapter merge | Adapter merge failed. |
| `6` | quality evaluation | Evaluation execution failed or did not complete. |
| `7` | `train_pipeline.py` | Full pipeline failed and wrote a failure report. |
| `8` | Full pipeline, Fact-SFT, or ONNX export | Full-pipeline artifacts completed but the quality gate failed; standalone Fact-SFT/ONNX scripts still use `8` for their own failure. |
| `9` | DPO | DPO preparation or training failed. |
| `10` | GRPO | GRPO preparation or training failed. |

## Corpus Safety Preflight

The default pipeline runs a safety preflight before CPT. It scans high-risk content, source paths, and possible secrets, then writes:

```text
outputs/logs/preflight_report.md
outputs/logs/preflight_report.json
```

If the report status is blocked, training stops by default. Handling order:

1. Open `outputs/logs/preflight_report.md`.
2. Remove or replace private documents, secrets, long code blocks, internal URLs, and license-restricted material.
3. Rerun the pipeline.
4. Use `--allow_unsafe_corpus` only in an offline, private, controlled environment where the corpus is approved for training.

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --allow_unsafe_corpus
```

## Resume and Retry

Prefer stage-level retries so completed stages do not run again.

The full pipeline can skip stages:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft
```

To run only GRPO from existing outputs:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

GRPO selects a complete PEFT adapter in explicit path, DPO, Fact-SFT, CPT order. The root must contain `adapter_config.json` and one adapter weight file; incomplete directories are recorded and skipped.

Stage scripts support preparation-only and training-only modes:

```bash
python -m pipeline.fact_sft --config configs/domain_post_training.local.yaml --prepare_only
python -m pipeline.fact_sft --config configs/domain_post_training.local.yaml --train_only
python -m pipeline.dpo --config configs/domain_post_training.local.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --train_only
```

To resume from a checkpoint, set the matching stage’s `resume_from_checkpoint`:

```yaml
fact_sft:
  resume_from_checkpoint: "outputs/fact_sft_adapter/checkpoint-100"

dpo:
  resume_from_checkpoint: "outputs/dpo_adapter/checkpoint-100"

grpo:
  resume_from_checkpoint: "outputs/grpo_adapter/checkpoint-100"
```

Skipping earlier stages starts a new GRPO stage from an existing adapter. `resume_from_checkpoint` restores Trainer state from the same stage run; do not treat them as the same operation.

To merge a specific adapter without relying on stage switches:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.local.yaml --adapter_dir outputs/grpo_adapter
```

## Reading the Main Reports

| Report | Use |
|---|---|
| `outputs/cpt_dataset/coverage_report.md` | CPT document discovery, chunking, and coverage. |
| `outputs/logs/preflight_report.md` | Pre-training corpus safety checks. |
| `outputs/fact_sft_dataset/fact_sft_dataset_report.md` | SFT example counts, skipped examples, and assistant-only loss. |
| `outputs/dpo_dataset/dpo_dataset_report.md` | DPO pair counts, skip reasons, and category distribution. |
| `outputs/grpo_dataset/grpo_dataset_report.md` | GRPO prompt counts, skip reasons, judge state, built-in rewards, and category distribution. |
| `outputs/reports/grpo_report.md` | GRPO starting adapter, precision, loss, judge configuration, and summaries of effective concurrency, completion count, and judge batch latency. |
| `outputs/merged_model/merge_report.json` | Adapter source, base model, dtype, and load test. |
| `outputs/eval/eval_report.md` | Post-training quality evaluation, especially `safety_boundary` and `base_regression`. |
| `outputs/reports/pipeline_report.md` | Full-pipeline summary. |

## Real-Run Acceptance Checklist

1. Inspect `grad_norm` across CPT, Fact-SFT, DPO, and GRPO. Treat any NaN/Inf as failure; finite loss and completed status are insufficient.
2. Compare LoRA tensors across adjacent adapters to verify real parameter changes.
3. Inspect judge reward mean and variance. Low rewards require rubric/data review; near-constant rewards cannot rank candidates effectively.
4. Inspect `completions/clipped_ratio`. Improve EOS/stop behavior before raising `grpo.max_completion_length`.
5. Compare `configured_max_concurrency`, `effective_concurrency`, `completion_count`, and `judge_batch_latency_seconds`; effective concurrency must not exceed `min(cap, completion count)`. The semaphore cap applies per process/rank, and executor capacity may reduce physical connections.
6. On 429/503, inspect `Retry-After`, provider limits, and `retry_backoff_seconds`. Any final request failure should fail the complete reward batch rather than continue with partial or neutral scores.
7. For empty judge `content`, use `reward_judge.max_tokens: 4096` and `timeout_seconds: 120`; do not tune judge output with policy completion length.
8. Treat `eval_report.md` as a heuristic smoke gate. Human-review or independently judge safety cases before production release.

## Storage and Cleanup

Do not delete before merge, evaluation, or export:

- The adapter directory you plan to merge.
- `outputs/merged_model/`.
- Dataset reports and training metadata.

Usually safe after you no longer need the ability to resume training:

- Old checkpoints.
- Old smoke artifacts: `outputs/smoke/`.
- Python caches: `__pycache__/`.
- Obsolete temporary export directories.
