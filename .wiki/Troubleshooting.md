# Troubleshooting

Use this page when a command fails, training stops, or expected artifacts are missing.

## Check Failure Reports First

For full-pipeline failures, open:

```text
outputs/reports/failure_report.md
outputs/reports/pipeline_report.md
```

If the failure happened before CPT, also check:

```text
outputs/logs/preflight_report.md
outputs/logs/discovered_corpus.json
```

See [Operations Runbook](Operations-Runbook.md) for exit codes and report locations.

## `ModuleNotFoundError: No module named ...`

Applies to: install, smoke test, training, inference.

Likely cause: the active Python environment does not contain the required package.

Fix:

```bash
python -m pip install -r requirements.txt
```

For ONNX-only failures:

```bash
python -m pip install -r requirements-onnx.txt
```

Verify:

```bash
python -m compileall pipeline scripts serve_inference.py
python scripts/diagnostics/check_training_environment.py
```

## CUDA Out of Memory

Applies to: CPT, Fact-SFT, DPO, GRPO.

Likely cause: sequence length, batch size, number of GRPO generations, or LoRA rank is too high for the GPU.

Fix:

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

For GRPO, also lower:

```yaml
grpo:
  num_generations: 2
  max_completion_length: 128
```

## `cuda_available: False`

Applies to: environment diagnostics, training, inference, ONNX export.

Likely cause: PyTorch CUDA wheel and driver mismatch, the wrong virtual environment, missing NVIDIA driver, or a CPU-only machine.

Check:

```bash
python scripts/diagnostics/check_training_environment.py
```

If `nvidia-smi` sees a GPU but `cuda_available: False`, reinstall the PyTorch build that matches the machine. Do not continue real training in this state; CPU is only practical for smoke tests.

## Corpus Preflight Blocked Training

Applies to: full pipeline failures before CPT.

Likely cause: corpus contains high-risk content, possible secrets, internal paths, long code blocks, or data that should not be distributed.

Fix:

1. Open `outputs/logs/preflight_report.md`.
2. Remove or replace the high-risk material.
3. Rerun the pipeline.

Use this only in a private, offline, approved training environment:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --allow_unsafe_corpus
```

## No Valid GRPO Reward Examples

Applies to: `scripts/training/train_grpo.py`.

Likely cause: the judge is disabled and rows have no signal matching an enabled built-in reward. With the judge enabled, prompt-only rows are valid when a prompt can be constructed.

Fix:

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

With the judge disabled, verify that every row has a field matching `builtin_rewards`:

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

See [Data Contracts](Data-Contracts.md) for accepted aliases.

## DPO Row Is Rejected

Applies to: DPO dataset preparation.

Likely cause: missing field, empty value, or `chosen == rejected`.

Fix: ensure every row has non-empty `prompt`, `chosen`, and `rejected`, and that the two answers differ. See [Data Contracts](Data-Contracts.md) for accepted aliases.

## Upstream Adapter Not Found

Applies to: Fact-SFT, DPO, GRPO, merge.

Likely cause: an upstream adapter has not been produced, or a directory exists but is not a complete PEFT adapter. A valid root contains `adapter_config.json` plus either `adapter_model.safetensors` or `adapter_model.bin`.

Fix options:

- Run the upstream stage first.
- Point `base_adapter_dir` to a complete adapter. GRPO falls back through explicit path -> DPO -> Fact-SFT -> CPT and reports missing files for invalid candidates.
- For a specific merge, use `merge.adapter_dir` or `merge_adapter.py --adapter_dir`.
- Set the relevant `require_*_adapter` option to `false` only for intentional experiments.

## Reward Judge API Key Is Missing

Applies to: GRPO with `grpo.reward_judge.enabled=true`.

Likely cause: `api_key` is empty in the Git-ignored local YAML and the optional environment fallback named by `api_key_env` is also unset.

Fix:

```yaml
grpo:
  reward_judge:
    api_key: "your-key"
```

`api_key` is plaintext and belongs only in the untracked `configs/domain_post_training.local.yaml`. If you choose the environment fallback:

```bash
export GRPO_REWARD_JUDGE_API_KEY="your-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "your-key"
```

## Reward Judge Response Does Not Match `grpo_judge_v2`

Applies to: GRPO external reward judge.

Likely cause: the judge returned prose, Markdown, the legacy scalar `score` format, missing or extra fields, out-of-range dimensions, or unknown violations.

Fix: update the judge prompt or service so `choices[0].message.content` in the OpenAI-compatible response contains JSON like:

```json
{"schema_version":"grpo_judge_v2","dimension_scores":{"task_fulfillment":0.8,"factual_grounding":null,"explicit_constraints":1.0,"safety_refusal":1.0,"relevance_clarity":0.8},"violations":[],"reason":"The response follows the supplied constraints."}
```

Strict JSON must appear inside assistant content in the chat completions envelope; returning it as the raw HTTP body is insufficient. Local code weights dimensions and applies violation caps; it does not read a final judge `score`. See [GRPO And Reward Judge](GRPO-and-Reward-Judge.md).

## Reward Judge `message.content` Is Empty

Applies to: external judges that produce internal reasoning before the final answer.

Likely cause: a small output budget was consumed by provider `reasoning_content`, leaving no tokens for final v2 JSON, or the timeout is too short.

Fix:

```yaml
grpo:
  reward_judge:
    max_tokens: 4096
    timeout_seconds: 120
```

This `max_tokens` is the judge response budget. Raising `grpo.max_completion_length` only lengthens policy candidates and cannot fix an empty judge response.

## Reward Judge Returns 429/503 or the Batch Times Out

Applies to: asynchronous external-judge scoring.

Inspect `configured_max_concurrency`, `effective_concurrency`, `completion_count`, and `judge_batch_latency_seconds` in the training logs. `max_concurrency` is a per-process/rank upper bound; the current completion count and `asyncio.to_thread` executor capacity can reduce physical concurrency. Lower an explicit cap when the provider rate-limits requests; the default `auto` follows `num_generations`.

Each completion independently uses exponential backoff with jitter and honors valid `Retry-After` values returned with 429/503. If any request exhausts its retries, the complete reward batch fails and successful partial scores are not used for training. Do not mask the failure with neutral rewards; inspect endpoint limits, timeouts, service capacity, and `retry_backoff_seconds`.

## `grad_norm` Is NaN or Inf

Applies to: CPT, Fact-SFT, DPO, GRPO.

Do not ignore this because loss is finite or the report says completed; non-finite gradients can mean the adapter did not update meaningfully. Keep these defaults:

```yaml
training:
  bf16: auto
  fp16: auto
  torch_dtype: auto
  abort_on_nonfinite_grad_norm: true
```

Fact-SFT, DPO, and GRPO use matching stage-level fields. BF16 is preferred when supported. If failure remains, inspect learning rate, quantization, loss scaling, anomalous rows, and dependency versions. After rerunning, compare LoRA tensors across adjacent adapters to confirm real changes.

## GRPO Reward Is Low, Nearly Constant, or Highly Clipped

Inspect judge reward mean/variance and `completions/clipped_ratio` in logs and reports. Near-constant rewards provide no useful ranking, while high clipping means the judge repeatedly sees incomplete candidates.

First verify rubric/data alignment and improve EOS/stop behavior. Raise `grpo.max_completion_length` only when complete answers genuinely require more room. Do not use judge `max_tokens` to tune policy candidate length.

## Flask Service Starts but `/v1/chat/completions` Fails

Likely causes:

- `--model_path` does not point to a merged model.
- Model dependencies are missing.
- The request uses a malformed `messages` payload.
- The service started with `cuda` on a machine without available CUDA.

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Then retry a minimal chat completions request from [Inference And Export](Inference-and-Export.md). On CPU-only machines, start the service with `--device cpu` or `--device auto`.

## Static Check Passes but Training Fails

`compileall` only checks Python syntax. It does not load datasets, import all ML dependencies, allocate GPU memory, or run TRL.

Next checks:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python scripts/diagnostics/check_training_environment.py
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```
