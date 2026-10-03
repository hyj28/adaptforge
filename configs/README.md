# Configuration Reference

This file explains the user-facing parameters in `domain_post_training.yaml`. For real training, copy it to the Git-ignored `configs/domain_post_training.local.yaml` and run scripts with `--config configs/domain_post_training.local.yaml`. A directly configured judge key belongs only in that private plaintext file; the tracked template must keep `api_key: null`.

Path rules:

- Training artifacts are usually resolved from the project root, such as `outputs/lora_adapter`.
- Data input paths are usually resolved relative to the config file first. In this project, `../data/...` means `data/...` under the project root.
- `null` means the code should use its default behavior.

## Top-Level Parameters

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `base_model_repo_id` | Hub download source used by `download_models.py`; the default is `Qwen/Qwen3.5-0.8B`. | Change it when switching the download source. |
| `base_model_name_or_path` | Model location actually loaded by training, merge, evaluation, and inference. | Keep `models/base-model` after download, or point it at an existing local snapshot. |
| `trust_remote_code` | Whether Transformers may load custom modeling code from the model repository. | Keep `true` for models that require custom code; set `false` for standard architectures when you want a stricter load path. |

## `corpus`

This section controls CPT corpus discovery, splitting, coverage, and sample construction.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `strategy` | CPT dataset construction strategy. `guaranteed_coverage_then_optional_replay` first guarantees required corpus coverage, then optionally replays important categories. | Usually keep it unless you add another sampling strategy. |
| `sample_unit` | CPT sample granularity. Supported values are `document`, `section`, and `token_chunk`. | Use `document` for short documents; use `section` or `token_chunk` for long corpora. |
| `input_paths` | CPT source document files or directories. | Change this when replacing the sample domain corpus. |
| `exclude_paths` | Path fragments excluded during corpus discovery. | Add source, build, output, model, or log directories that should not become training text. |
| `prepared_dataset_dir` | Output directory for the prepared CPT dataset. | Change it when keeping multiple dataset builds. |
| `min_text_chars` | Minimum source text length in characters. Shorter texts are skipped. | Lower it for short but meaningful notes; raise it to filter noise. |
| `append_safety_preamble` | Adds a generic safety preamble to CPT text. | Usually keep `true`; set `false` if your corpus already encodes the required safety boundary. |
| `split_markdown_sections` | Split Markdown by headings. Usually unnecessary with `sample_unit: document`. | Enable it for long Markdown documents. |
| `max_sample_tokens` | Maximum tokens per CPT sample. | Lower it for memory pressure; raise it for longer context training. |
| `sample_overflow_strategy` | Behavior when a whole sample exceeds `max_sample_tokens`; the default `error` fails fast. | Keep `error` when you want to find long documents instead of silently truncating them. |
| `enable_weighted_replay` | Enables category-based replay so selected categories appear more often. | Enable it for small corpora where safety, configuration, or troubleshooting should be reinforced. |
| `total_train_tokens` | Target total CPT training tokens. `null` uses natural coverage and replay. | Set it to control training size. |
| `min_chunk_tokens` | Minimum tokens for generated chunks. | Increase it if splitting creates too many tiny chunks. |
| `validation_mode` | CPT validation mode: `none`, `copy_from_train`, or `separate_sources`. | Use `separate_sources` for real held-out validation documents. |
| `validation_copy_ratio` | Ratio copied from training samples when `validation_mode: copy_from_train`. | Useful only for smoke checks, typically `0.03` to `0.1`. |
| `validation_sources` | Held-out CPT validation document paths for `separate_sources`. | Fill this when you have independent validation documents. |

### `corpus.stratified_sampling`

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `stratified_sampling` | Configuration block for category replay. | Use this section when you want selected corpus categories to repeat more often. |
| `enabled` | Enables category-based replay. | Turn it on for small corpora where key behavior should be reinforced. |
| `mode` | Replay mode. The default `sample_repeats` uses repeat counts per category. | Usually keep it. |
| `sample_repeats` | Map of corpus category names to repeat counts. | Increase important categories; lower values to reduce overfitting. |
| `sample_repeats.model_spec` (`model_spec`) | Repeat count for model specification documents. | Raise it for role, boundary, or IO contract documents. |
| `sample_repeats.main_textbook` (`main_textbook`) | Repeat count for main overview documents. | Raise it when the main guide is high quality and representative. |
| `sample_repeats.commands` (`commands`) | Repeat count for command or operation documents. | Use it for command-heavy domains. |
| `sample_repeats.permissions` (`permissions`) | Repeat count for permission and access-control documents. | Raise it when permission boundaries matter. |
| `sample_repeats.configuration` (`configuration`) | Repeat count for configuration documents. | Raise it for stable answers about configuration keys. |
| `sample_repeats.runtime_behavior` (`runtime_behavior`) | Repeat count for runtime behavior documents. | Raise it when state transitions and behavior matter. |
| `sample_repeats.database_reload` (`database_reload`) | Repeat count for data reload or refresh documents. | Use it when the domain includes data refresh workflows. |
| `sample_repeats.troubleshooting` (`troubleshooting`) | Repeat count for troubleshooting documents. | Often useful for support assistants. |
| `sample_repeats.safety` (`safety`) | Repeat count for safety-boundary documents. | Keep this relatively high for open-source examples and real assistants. |
| `sample_repeats.unknowns` (`unknowns`) | Repeat count for unknown-boundary documents. | Raise it to reduce unsupported claims. |
| `sample_repeats.public_api` (`public_api`) | Repeat count for public API documents. | Use it when your corpus includes API docs. |

## `safety`

This section controls corpus preflight checks. It is not the runtime model safety system.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `block_on_high_risk` | Blocks training when high-risk corpus content is found. | Keep `true` for open-source or shared workflows. |
| `max_consecutive_code_lines` | Threshold for consecutive code lines before a source-code warning. | Increase it if your licensed documentation legitimately contains long code blocks. |
| `warn_on_source_paths` | Warn on source-code-like paths. | Usually keep `true`. |
| `warn_on_secret_patterns` | Warn on credential, token, or secret-like patterns. | Usually keep `true`. |

## `training`

This section controls the CPT stage and shared training defaults. Fact-SFT, DPO, and GRPO can override many of these settings.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `device` | Training device, such as `cuda`, `cuda:0`, `cpu`, or `auto`. | Use CUDA for real training; use CLI `--device cpu` for CPU smoke tests. |
| `output_dir` | CPT LoRA adapter output directory. | Change it to keep multiple experiments. |
| `merged_output_dir` | Output directory for the merged full model. | Change it when publishing or comparing merged models. |
| `max_seq_length` | CPT maximum sequence length. | Lower it for memory pressure; raise it for long-document training. |
| `train_val_split` | Legacy train/validation split ratio. Not recommended in the current coverage mode. | Prefer `corpus.validation_mode` and SFT/DPO `validation_ratio`. |
| `seed` | Random seed. | Fix it for reproducibility; change it for repeated experiments. |
| `epochs` | CPT training epochs. | Small corpora usually start with 1 to 3. |
| `max_steps` | Maximum training steps. `null` uses epochs. | Set it for quick experiments or cost control. |
| `per_device_train_batch_size` | Per-device training batch size. | Increase with enough memory; lower it on OOM. |
| `per_device_eval_batch_size` | Per-device evaluation batch size. | Used only when validation is enabled. |
| `gradient_accumulation_steps` | Accumulates gradients to simulate a larger effective batch. | Increase it when memory is limited. |
| `learning_rate` | CPT learning rate. | Lower it for unstable training or forgetting; raise carefully if convergence is too slow. |
| `weight_decay` | Weight decay. | LoRA fine-tuning usually keeps this at `0.0` or very small. |
| `warmup_ratio` | Learning-rate warmup ratio. | Keep some warmup for small, sensitive runs. |
| `lr_scheduler_type` | Learning-rate scheduler, such as `cosine`. | Change only for specific experiments. |
| `optim` | Optimizer, such as `paged_adamw_8bit`. | Match it to your quantization and hardware setup. |
| `max_grad_norm` | Gradient clipping threshold. | Lower it if gradients or loss are unstable. |
| `abort_on_nonfinite_grad_norm` | Abort when non-finite gradient norm is detected. | Keep it `true`; otherwise AMP may skip updates while a completed-looking adapter is still saved. |
| `logging_nan_inf_filter` | Whether Trainer filters NaN/Inf in logs. | Set `false` when debugging numeric issues. |
| `logging_steps` | Logging interval in steps. | Lower it for denser training logs. |
| `eval_steps` | Evaluation interval when validation is enabled. | Tune it to the training duration. |
| `save_steps` | Checkpoint save interval. | Lower it for long runs; raise it for short runs. |
| `save_total_limit` | Maximum retained checkpoints. | Lower it when disk space is limited. |
| `gradient_checkpointing` | Trades compute for lower memory usage. | Keep `true` when memory is limited. |
| `bf16` | Enable bfloat16 training; accepts `true`, `false`, or `auto`. | Keep `auto` to prefer BF16 on supported CUDA devices. |
| `fp16` | Enable float16 training; accepts `true`, `false`, or `auto`. | Keep `auto` to fall back to FP16 on CUDA when BF16 is not used; CPU disables it. |
| `load_in_4bit` | Load the base model in 4-bit mode. | Useful for QLoRA and low-memory training. |
| `load_in_8bit` | Load the base model in 8-bit mode. | Alternative to 4-bit; do not enable both. |
| `torch_dtype` | Model load dtype: `float16`, `bfloat16`, `float32`, or `auto`. | Keep `auto` for hardware-aware loading; use an explicit value only in a validated experiment. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same CPT stage. | Use only after an interruption, not to start from another stage's adapter. |

Exit code `0` or a `completed` report does not prove that updates occurred. Acceptance requires finite `grad_norm` values for every stage and actual LoRA tensor changes between adjacent adapters.

## `peft`

This section controls the LoRA/QLoRA adapter structure.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `r` | LoRA rank. Higher values increase capacity and memory use. | Raise it for complex tasks; small datasets often use 8 to 32. |
| `lora_alpha` | LoRA scaling value. | Keep it near `r` or smaller; lower it if updates are too strong. |
| `lora_dropout` | LoRA dropout. | Useful for small datasets, often 0.03 to 0.1. |
| `target_modules` | Model module names that receive LoRA adapters. | Change this when switching to an architecture with different module names. |
| `bias` | Whether LoRA trains bias terms. | Usually keep `none`. |
| `use_dora` | Enables DoRA if supported by your PEFT version. | Use only for explicit experiments. |
| `use_rslora` | Enables Rank-Stabilized LoRA. | Try it for high-rank or unstable LoRA experiments. |

## `fact_sft`

This section controls Fact-SFT. It uses assistant-only loss, so prompt tokens are masked and only assistant answer tokens are trained.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `enabled` | Enables the Fact-SFT stage. | Set `false` if you only want CPT. |
| `input_paths` | SFT JSON/JSONL inputs. Rows usually contain `instruction` and `output`. | Change this when replacing domain SFT data. |
| `prepared_dataset_dir` | Output directory for the prepared Fact-SFT dataset. | Change it to keep multiple experiments. |
| `base_adapter_dir` | Starting adapter for Fact-SFT, usually the CPT adapter. | Change it when skipping CPT or using an existing adapter. |
| `output_dir` | Fact-SFT adapter output directory. | Change it to keep multiple runs. |
| `require_cpt_adapter` | Requires `base_adapter_dir` to exist. | Set `false` only if you intentionally train SFT from the base model. |
| `max_seq_length` | Maximum sequence length for Fact-SFT. | Raise it for long QA examples; lower it on OOM. |
| `validation_ratio` | Ratio split from SFT data for validation. `0` disables validation. | Use `0.05` to `0.1` when you have enough real data. |
| `seed` | SFT shuffle and validation split seed. | Fix it for reproducibility. |
| `epochs` | SFT epochs. | Small datasets often start with 1 to 3. |
| `max_steps` | Maximum SFT steps. `null` uses epochs. | Set it for quick experiments or cost control. |
| `per_device_train_batch_size` | Per-device SFT training batch size. | Lower it on OOM. |
| `per_device_eval_batch_size` | Per-device SFT validation batch size. | Used only when validation is enabled. |
| `gradient_accumulation_steps` | SFT gradient accumulation steps. | Increase it when memory is limited. |
| `learning_rate` | SFT learning rate. | Keep it small to avoid overriding CPT knowledge. |
| `weight_decay` | SFT weight decay. | Usually keep `0.0`. |
| `warmup_ratio` | SFT warmup ratio. | Useful for small runs. |
| `lr_scheduler_type` | SFT scheduler. | Usually keep `cosine`. |
| `optim` | SFT optimizer. | Match your hardware and quantization setup. |
| `max_grad_norm` | SFT gradient clipping threshold. | Lower it if loss is unstable. |
| `bf16` | Enable bfloat16 for SFT; accepts `auto`. | Keep `auto` to prefer BF16 on supported CUDA devices. |
| `fp16` | Enable float16 for SFT; accepts `auto`. | Keep `auto` to fall back only when CUDA is not using BF16; CPU disables it. |
| `torch_dtype` | SFT model load dtype. | Keep `auto` unless an explicit dtype has been validated. |
| `abort_on_nonfinite_grad_norm` | Abort SFT on non-finite gradient norm. | Keep it `true`. |
| `logging_steps` | SFT logging interval. | Lower it for more detailed logs. |
| `eval_steps` | SFT evaluation interval. | Used only with validation. |
| `save_steps` | SFT checkpoint interval. | Tune by run length and disk space. |
| `save_total_limit` | Maximum retained SFT checkpoints. | Lower it when disk space is limited. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same SFT stage. | Use only after an interruption; the CPT adapter still comes from `base_adapter_dir`. |
| `system_prompt` | System prompt used when constructing Fact-SFT chat prompts. | Must be rewritten for your domain role, knowledge boundary, and safety boundary. |

## `dpo`

This section controls DPO preference training, which is enabled by default and can be disabled explicitly. Each input row needs `prompt`, `chosen`, and `rejected`.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `enabled` | Enables the DPO stage; defaults to `true`. | Set `false` when preference data is unavailable or the stage should be skipped. |
| `input_path` | DPO JSON/JSONL file or directory. | Change it when replacing preference data. |
| `prepared_dataset_dir` | Output directory for the prepared DPO dataset. | Change it to keep multiple experiments. |
| `base_adapter_dir` | Starting adapter for DPO, usually Fact-SFT output. | Change it when skipping SFT or using an existing adapter. |
| `output_dir` | DPO adapter output directory. | Change it to keep multiple runs. |
| `require_base_adapter` | Requires `base_adapter_dir` to exist. | Set `false` only for intentional base-model DPO experiments. |
| `max_length` | Maximum combined prompt and response length. | Raise it for long samples; lower it on OOM. |
| `validation_ratio` | Ratio split from DPO data for validation. | Use `0.05` to `0.1` when you have enough data. |
| `seed` | DPO shuffle and validation split seed. | Fix it for reproducibility. |
| `epochs` | DPO epochs. | Start with 1 to avoid preference overfitting. |
| `max_steps` | Maximum DPO steps. | Set it for quick experiments or cost control. |
| `per_device_train_batch_size` | Per-device DPO training batch size. | DPO is memory-heavy; lower it on OOM. |
| `per_device_eval_batch_size` | Per-device DPO validation batch size. | Used only when validation is enabled. |
| `gradient_accumulation_steps` | DPO gradient accumulation steps. | Increase it when memory is limited. |
| `learning_rate` | DPO learning rate. | Usually lower than SFT. |
| `weight_decay` | DPO weight decay. | Usually keep `0.0`. |
| `warmup_ratio` | DPO warmup ratio. | Useful for small runs. |
| `lr_scheduler_type` | DPO scheduler. | Usually keep `cosine`. |
| `optim` | DPO optimizer. | Match hardware and quantization. |
| `max_grad_norm` | DPO gradient clipping threshold. | Lower it if training is unstable. |
| `bf16` | Enable bfloat16 for DPO; accepts `auto`. | Keep `auto` to prefer BF16 on supported CUDA devices. |
| `fp16` | Enable float16 for DPO; accepts `auto`. | Keep `auto` to fall back only when CUDA is not using BF16; CPU disables it. |
| `torch_dtype` | DPO model load dtype. | Keep `auto` unless an explicit dtype has been validated. |
| `abort_on_nonfinite_grad_norm` | Abort DPO on non-finite gradient norm. | Keep it `true`. |
| `logging_steps` | DPO logging interval. | Lower it for more detailed logs. |
| `eval_steps` | DPO evaluation interval. | Used only with validation. |
| `save_steps` | DPO checkpoint interval. | Tune by run length and disk space. |
| `save_total_limit` | Maximum retained DPO checkpoints. | Lower it when disk space is limited. |
| `beta` | DPO preference strength. | Lower it if outputs become too narrow; raise carefully if preference learning is weak. |
| `loss_type` | DPO loss type, such as `sigmoid`. | Usually keep it unless testing another TRL loss. |
| `truncation_mode` | Truncation behavior for long samples. `keep_start` preserves the beginning. | Change carefully if answer tails matter more than prompt starts. |
| `precompute_ref_log_probs` | Precomputes reference log probabilities. | Try it for larger datasets when your TRL version and hardware support it. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same DPO stage. | Use only after an interruption; the previous-stage adapter still comes from `base_adapter_dir`. |

## `grpo`

This section controls GRPO reward optimization after DPO, Fact-SFT, or CPT. GRPO and its external OpenAI-compatible judge are enabled by default, so judge-only rows may contain just `prompt`. Supply trusted private-domain evidence through `trusted_context` / `judge_context` / `context` or `reference_answer`; text inside the user question is never promoted to trusted context.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `enabled` | Enables the GRPO stage; defaults to `true`. | Set `false` when reward prompts are unavailable or the stage should be skipped. |
| `input_path` | GRPO JSON/JSONL file or directory. | Change it when replacing reward-prompt data. |
| `prepared_dataset_dir` | Output directory for the prepared GRPO dataset. | Change it to keep multiple experiments. |
| `base_adapter_dir` | Preferred starting adapter for GRPO. Resolution then falls back through DPO, Fact-SFT, and CPT outputs. | Change it when using an existing adapter outside the configured stage outputs. |
| `output_dir` | GRPO adapter output directory. | Change it to keep multiple runs. |
| `require_base_adapter` | Requires an existing DPO, Fact-SFT, or CPT adapter before GRPO. | Set `false` only for intentional base-model GRPO experiments or smoke wiring. |
| `max_prompt_length` | Maximum prompt length passed to GRPO generation. | Lower it on OOM; raise it for long prompts. |
| `max_completion_length` | Maximum policy-completion tokens per rollout. | Lower it to reduce rollout cost; on high clipping, improve EOS/stop behavior before raising it. |
| `num_generations` | Number of completions sampled per prompt. | Increase for stronger relative reward signal; lower it for memory or speed. |
| `temperature` / `top_p` | Rollout sampling controls. | Tune when completions are too deterministic or too noisy. |
| `use_vllm` | Uses the vLLM rollout path supported by the installed TRL version. | Keep `false` unless deployment and memory behavior have been validated. |
| `bf16` / `fp16` / `torch_dtype` | GRPO precision settings; all accept `auto`. | Keep `auto`: BF16-capable CUDA prefers BF16, other CUDA falls back to FP16, and CPU disables mixed precision. |
| `abort_on_nonfinite_grad_norm` | Abort on a non-finite gradient norm. | Keep it `true` to avoid saving completed-looking artifacts after skipped updates. |
| `builtin_rewards` | Optional built-in rewards: `reference_overlap`, `term_constraints`, `refusal`, `length_bounds`; defaults to `[]`. | Add only rewards whose matching signals are present in every applicable dataset row. |
| `reward_judge` | Default reward provider. Configure `enabled`, `base_url`, `api_key`, `model`, `score_range`, `timeout_seconds`, `max_tokens`, `max_retries`, `max_concurrency`, `retry_backoff_seconds`, `system_prompt`, and `prompt_template`; `api_key_env` is an optional fallback. | Set `enabled: false` only when at least one built-in reward is enabled. Keep the strict v2 output schema when customizing prompts. |
| `refusal_terms` | Phrases used by the built-in `refusal` reward to detect a refusal. | Change only when that built-in reward is enabled and the domain needs additional refusal wording. |
| `beta` | KL/reference regularization strength when supported by the installed TRL version. | Raise carefully when updates drift too far from the reference policy. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same GRPO stage. | Use only after an interruption; it does not replace `base_adapter_dir`. |

`reward_judge` standardizes model-based scoring through the OpenAI-compatible chat completions API. Put the primary `api_key` only in the private local YAML; if it is null or empty, `api_key_env` may name an environment-variable fallback. Local judges must first be served as an OpenAI-compatible HTTP service, for example with `base_url: "http://localhost:8000/v1"`. Hosted judges use the same fields with their provider URL and model name. For reasoning judges, `max_tokens: 4096` and `timeout_seconds: 120` are the recommended baseline so the final JSON is not lost to a smaller reasoning budget or timeout. `reward_judge.max_tokens` limits judge reasoning plus JSON output; `grpo.max_completion_length` limits policy candidates. Missing judge settings fail before dataset preparation; missing adapters produce commands for continuing GRPO from an existing previous-stage output.

`max_concurrency` accepts `"auto"` or an integer from `1` to `64`. `"auto"` resolves to `grpo.num_generations`, and `effective_concurrency` for each reward batch is `min(resolved cap, completion_count)`. An explicit integer can score completions across multiple prompts in the same reward batch. The semaphore cap applies per training process/rank; it is the whole-run cap for a single-GPU process, while the `asyncio.to_thread` executor may impose a lower physical socket concurrency. Treat it as an upper bound, not a guaranteed worker count.

Judge concurrency covers only scoring for the current rollout batch. The flow is `candidate generation -> concurrent scoring -> wait for the whole batch -> group advantage -> synchronized weight update`; there is no offline full-dataset pre-generation or concurrent optimizer update. Each completion retries independently with exponential backoff and jitter based on `retry_backoff_seconds`, honoring valid `Retry-After` values on HTTP 429/503. If any request exhausts `max_retries`, the entire reward batch fails and partial scores are discarded.

The default `grpo_judge_v2` contract returns five dimension scores plus violation identifiers and a concise reason. Python code, rather than the external model, applies the configured weights and hard caps. Invalid JSON, non-finite or out-of-range dimensions, unknown violations, and extra or missing top-level fields fail closed and are retried up to `max_retries`. Prompt-only rows without supplied grounding evidence use `null` for factual grounding and reweight the other dimensions. Remote judges receive the complete prompt, completion, reference, constraints, and metadata.

The default `prompt_template` places JSON-encoded untrusted data after `EVALUATION_INPUT`. Custom templates must retain the `{evaluation_input_json}` placeholder instead of interpolating candidate text as privileged instructions.

For the external judge, `forbidden_terms_mode` defaults to `semantic`, where safe quotation, negation, or refusal does not count as prohibited use. Set it to `literal` on a row when every case-insensitive occurrence must be rejected. The built-in `term_constraints` reward keeps its original case-insensitive literal-substring formula and is unaffected by this mode. Completion character length is computed locally with stripped Unicode code-point length and inclusive min/max bounds; the judge receives the deterministic result rather than estimating length.

Standalone `train_grpo.py --base_adapter_dir ...` and pipeline execution with `--skip_cpt --skip_sft --skip_dpo` start a new GRPO stage from an existing previous-stage adapter. `resume_from_checkpoint` only resumes the same GRPO run. After training, verify finite `grad_norm`, changed adapter tensors, a meaningful judge-reward distribution, and `completions/clipped_ratio`; low reward variance or widespread clipping is not a production-ready result.

## `merge`

This section controls merging an adapter into a full Hugging Face model.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `adapter_dir` | Adapter path to merge. `null` auto-selects GRPO, DPO, Fact-SFT, or CPT output in that order. | Fill it when you want to merge a specific adapter. |
| `dtype` | Merge/load dtype, such as `float16`, `float32`, or `auto`. | Use `float16` for GPU inference; consider `float32` for CPU or ONNX workflows. |
| `safe_serialization` | Save model weights with safetensors. | Keep `true`. |

## `eval`

This section controls post-training quality evaluation and default generation parameters. It is not the training validation set.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `question_file` | JSONL question set for post-training quality evaluation. Each row has `category` and `question`. | Replace it with your domain quality evaluation set. |
| `max_new_tokens` | Default maximum generated tokens for evaluation and inference. | Increase it if answers are cut off; lower it to control output length. |
| `temperature` | Sampling temperature. Lower values are more stable; `0` is nearly deterministic. | Use low temperature for evaluation; raise it only for more creative output. |
| `top_p` | Nucleus sampling parameter. | Usually keep it stable for evaluation. |
| `repetition_penalty` | Penalty for repeated text. | Raise it slightly if the model repeats itself. |
| `no_repeat_ngram_size` | Repeated n-gram ban used by evaluation and default inference. | Keep the default `6`, or raise carefully for repetition degeneration. |
| `quality_gate` | Evaluation acceptance gate with `enabled`, `fail_pipeline`, and category `minimum_pass_rate`. | Enabled by default with strict thresholds; failures retain artifacts and make the full pipeline exit `8`. |

The default `minimum_pass_rate` is `1.0` for `domain_knowledge`, `safety_boundary`, and `base_regression`. Adjust these thresholds for your evaluation design; a missing configured category fails closed.

The current quality evaluation is a heuristic smoke gate, not a safety certification. Production acceptance requires human or independent-judge review of safety examples together with training stability, adapter deltas, and GRPO reward/clipping metrics.

Evaluation uses the same chat template, `enable_thinking: false`, reasoning cleanup, and plain-text normalization as CLI/service inference. Setting `quality_gate.fail_pipeline: false` only makes a failed gate non-blocking; it never changes failure into success. `--skip_eval` records `not_evaluated` and does not mark artifacts release-ready. Every adapter root carries a self-contained `adapter_provenance.json`; merge reports use it to record CPT, Fact-SFT, DPO, and GRPO accurately, with recursive legacy-metadata fallback for old adapters.

## `gguf`

This section controls GGUF export defaults. The project does not download a third-party GGUF reference by default; export your own GGUF from the merged model.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `output_dir` | GGUF output directory. | Change it for release or deployment folders. |
| `output_name` | GGUF output filename. | Include model name or quantization method when publishing variants. |
| `quantization_method` | Quantization method, such as `q4_k_m`. | Use lower quantization for smaller files; higher quantization for quality. |

## `onnx`

This section controls optional ONNX export. ONNX dependencies live in `requirements-onnx.txt` and are not part of the default install path.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `output_dir` | ONNX output directory. | Change it to keep multiple exports. |
| `output_name` | ONNX filename. | Change it for versioned exports. |
| `opset` | ONNX opset version. | Change it when your target runtime requires a specific opset. |
| `dtype` | ONNX export dtype. | Use `float32` for CPU compatibility; experiment with lower precision for size and speed. |
| `device` | Export device. | Use `cuda` with a GPU; use `cpu` otherwise. |
| `attn_implementation` | Attention implementation. `eager` is often more export-friendly. | Change it only when the default fails or the model requires another mode. |
| `dummy_seq_length` | Dummy input length used during export. | Increase it when your target runtime needs larger static shapes. |
| `external_data` | Store large weights as ONNX external data. | Keep `true` for large models. |
| `validate` | Run ONNX checker validation. | Keep `true` for debugging; disable only for quick export or missing packages. |
| `ort_check` | Run an ONNX Runtime output-shape check. | Enable it when ONNX Runtime is installed and you want stronger validation. |

## Common Edits

When adapting the project to your own domain, start with:

```yaml
base_model_repo_id: "your-org/your-base-model"
base_model_name_or_path: "models/base-model"

corpus:
  input_paths:
    - "../data/cpt/source_documents"

fact_sft:
  input_paths:
    - "../data/sft"
  system_prompt: >-
    You are a support assistant for your domain documentation.
    Answer only from the provided documentation. If the documentation does not
    specify a claim, say so. Return only the final answer.

eval:
  question_file: "../data/eval/quality_questions.jsonl"
```

When GPU memory is tight, start with:

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

DPO is enabled by default. Confirm its input data, or set `enabled: false` to skip it:

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

GRPO and its external judge are also enabled by default. Configure `base_url`, `model`, and `api_key` in private `configs/domain_post_training.local.yaml`, or set `grpo.enabled: false` to skip it:

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://your-openai-compatible-endpoint/v1"
    model: "your-judge-model"
    api_key: "replace-only-in-local-config"
    timeout_seconds: 120
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
    max_tokens: 4096
```
