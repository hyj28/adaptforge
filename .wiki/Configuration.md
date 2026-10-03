# Configuration

Use this page when adapting AdaptForge to another domain, changing stage outputs, or tuning training behavior.

The full parameter reference lives in `configs/README.md`. This page focuses on the settings users usually edit first.

## Main Config File

The default config is:

```text
configs/domain_post_training.yaml
```

Create a Git-ignored local config and keep the real judge key only in that file:

```powershell
Copy-Item configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Then run commands with:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

Keep `configs/*.local.yaml` ignored by Git. `api_key` is a plaintext credential; never commit it, upload it, print it in logs, or share the live training config. Environment variables remain an optional fallback, not the primary configuration path.

## Path Rules

- Training artifacts usually resolve relative to the repository root, such as `outputs/lora_adapter`.
- Data input paths are usually resolved relative to the config file first. In the default config, `../data/...` points back to repository-level `data/...`.
- `null` means the code should use its default behavior.

## Replace These First

| Setting | Why it matters |
|---|---|
| `base_model_repo_id` | Hugging Face Hub repo used by `download_models.py`. |
| `base_model_name_or_path` | Actual local path or model id loaded for training, merge, and inference. |
| `corpus.input_paths` | CPT source documents for the domain. |
| `fact_sft.input_paths` | Fact-SFT JSONL data. |
| `dpo.input_path` | DPO preference data, if DPO is enabled. |
| `grpo.input_path` | GRPO reward-prompt data, if GRPO is enabled. |
| `eval.question_file` | Post-training quality evaluation questions. |
| `fact_sft.system_prompt` | Domain role, knowledge boundary, and safety boundary. |

`base_model_repo_id` and `base_model_name_or_path` are not duplicates. The first tells `download_models.py` what to download; the second tells training, merge, and inference what to load. The default pairing downloads the Hub snapshot into `models/base-model` and then trains from that local directory.

## Stage Switches

```yaml
fact_sft:
  enabled: true

dpo:
  enabled: true

grpo:
  enabled: true
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://your-judge.example/v1"
    model: "your-judge-model"
    api_key: "replace-with-your-key"
    timeout_seconds: 120
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
    max_tokens: 4096
```

The full pipeline starts with CPT and then runs enabled later stages. The default template enables DPO and GRPO, so real training requires their datasets and a configured judge. The judge is enabled by default while all four built-in rewards are opt-in. If you explicitly disable the judge, `builtin_rewards` must contain at least one reward backed by the row data. You can also skip stages from the command line. See [Training Pipeline](Training-Pipeline.md).

`max_concurrency: "auto"` resolves to `num_generations`, and effective concurrency for each batch is `min(cap, completion count)`. An explicit integer lets prompts in the same reward batch share a larger cap; group advantage and synchronized weight updates run only after the complete score set returns. The cap applies per training process/rank, and executor capacity may make physical connection concurrency lower. Requests retry independently with backoff, valid `Retry-After` values on 429/503 are honored, and one final request failure fails the complete reward batch.

To avoid storing the key in YAML, leave `api_key` empty and configure the optional fallback:

```yaml
grpo:
  reward_judge:
    api_key: null
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
```

## Memory-Sensitive Defaults

When GPU memory is tight, start with smaller sequence length, smaller LoRA rank, 4-bit loading, and gradient checkpointing:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 768
  load_in_4bit: true
  gradient_checkpointing: true

peft:
  r: 8
  lora_alpha: 8
```

Let all four training stages select precision for the hardware and fail on non-finite gradients:

```yaml
training:
  bf16: auto
  fp16: auto
  torch_dtype: auto
  abort_on_nonfinite_grad_norm: true
```

Fact-SFT, DPO, and GRPO use the same stage-level fields: BF16 is preferred when supported, CUDA otherwise falls back to FP16, and CPU enables neither. Automatic selection is only a compatibility default; after training, still inspect `grad_norm` and verify that adapter tensors actually changed.

## Validation and Quality Evaluation

Training validation and post-training quality evaluation are different:

- Validation set: used during training for loss/eval signal.
- Quality evaluation: run after training to check factual answers, safe refusals, and regressions.

The built-in quality evaluation is a heuristic smoke gate, not production safety certification. Independently judge or manually review safety cases before release.

Evaluation shares the deployed chat template, `enable_thinking: false`, reasoning cleanup, and plain-text rules. The default `eval.quality_gate` requires all three categories to pass; a failed gate retains artifacts and makes the full pipeline exit `8`. `fail_pipeline: false` only downgrades failure to a warning, while `--skip_eval` records `not_evaluated`; neither marks artifacts release-ready.

The mock data is small, so validation is disabled by default:

```yaml
corpus:
  validation_mode: "none"
fact_sft:
  validation_ratio: 0
dpo:
  validation_ratio: 0
```

For real projects, prefer independent held-out CPT documents:

```yaml
corpus:
  validation_mode: "separate_sources"
  validation_sources:
    - "../data/cpt_validation/source_documents"
```

## Related Pages

- [Data Contracts](Data-Contracts.md)
- [Training Pipeline](Training-Pipeline.md)
- [GRPO And Reward Judge](GRPO-and-Reward-Judge.md)
- [Troubleshooting](Troubleshooting.md)
