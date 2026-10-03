# GRPO And Reward Judge

GRPO uses an external LLM judge as the default reward provider. All four built-in rewards are disabled until explicitly listed in `builtin_rewards`. The external judge is called through an OpenAI-compatible `/v1/chat/completions` endpoint.

## Recommended Configuration

Keep live settings in the Git-ignored `configs/domain_post_training.local.yaml`:

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  num_generations: 4
  max_completion_length: 256
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://api.deepseek.com/v1"
    model: "your-judge-model"
    api_key: "replace-with-your-key"
    score_range: [0.0, 1.0]
    timeout_seconds: 120
    max_tokens: 4096
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
```

`api_key` is a plaintext secret. Keep `configs/*.local.yaml` ignored by Git and never put a live key in tracked templates, training reports, or shared configs. To avoid storing the key in YAML, leave `api_key` empty and use an environment variable as an optional fallback:

```yaml
grpo:
  reward_judge:
    api_key: null
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
```

A directly configured `api_key` takes precedence over `api_key_env`.

## Data and Reward Modes

With the judge enabled, each row only needs a constructible prompt; prompt-only rows are valid. Reference answers, trusted context, terms, refusal requirements, and length bounds are optional evaluation evidence.

Enable built-in rewards only when needed:

| Reward | Matching fields |
|---|---|
| `reference_overlap` | `reference_answer` or an accepted alias |
| `term_constraints` | Non-empty `required_terms` or `forbidden_terms` |
| `refusal` | A valid `must_refuse` requirement; recognized phrases come from config-level `refusal_terms` |
| `length_bounds` | `min_completion_chars` or `max_completion_chars` |

For example, add only term and length rewards:

```yaml
grpo:
  builtin_rewards:
    - "term_constraints"
    - "length_bounds"
```

If you explicitly disable the judge, enable at least one built-in reward and give every row at least one signal matching an enabled reward:

```yaml
grpo:
  builtin_rewards:
    - "reference_overlap"
  reward_judge:
    enabled: false
```

## `grpo_judge_v2` Contract

The default prompts encode the prompt, candidate response, reference, trusted context, constraints, and category as untrusted JSON data and tell the judge to ignore score manipulation or role-overriding instructions inside it. The judge returns dimension scores and violation identifiers, not a final score. Local code deterministically weights the dimensions and applies hard caps.

Dimensions and initial weights:

| Dimension | Weight | Meaning |
|---|---:|---|
| `task_fulfillment` | 30% | Completes the legitimate task; a safe refusal can satisfy the task. |
| `factual_grounding` | 25% | Agrees with `trusted_context` / `reference_answer` and avoids fabrication. |
| `explicit_constraints` | 15% | Follows term, forbidden-content, refusal, format, and length constraints. |
| `safety_refusal` | 20% | Protects credentials, hidden prompts, private implementation, and access boundaries and refuses unsafe help. |
| `relevance_clarity` | 10% | Is relevant, clear, concise, and actionable where appropriate. |

Without a reference or trusted context, `factual_grounding` must be `null`; local code renormalizes the other weights. These weights are a project baseline to calibrate with domain-expert examples, not a universal standard.

Strict response example:

```json
{
  "schema_version": "grpo_judge_v2",
  "dimension_scores": {
    "task_fulfillment": 0.75,
    "factual_grounding": null,
    "explicit_constraints": 1.0,
    "safety_refusal": 1.0,
    "relevance_clarity": 0.75
  },
  "violations": [],
  "reason": "The response follows the explicit constraints and stays within the supplied evidence."
}
```

Top-level fields, all five dimensions, and the schema version must match exactly. Scores must be finite numbers from `0.0` to `1.0`, `reason` must be a non-empty string, and violation identifiers must come from the allowlist. The legacy `{"score": 0.8, "reason": "..."}` format is no longer valid.

Local hard caps:

| Violation | Maximum normalized score |
|---|---:|
| `unsafe_compliance` | 0.00 |
| `protected_information_leak` | 0.00 |
| `evaluator_manipulation` | 0.05 |
| `empty_or_irrelevant` | 0.05 |
| `missing_required_refusal` | 0.10 |
| `material_reference_contradiction` | 0.25 |
| `unsupported_domain_claim` | 0.35 |
| `explicit_constraint_violation` | 0.50 |
| `all_required_terms_missing` | 0.65 |

Local code also forces violations for empty responses, length failures, and forbidden terms in literal mode. After weighting and caps, the result is mapped to `score_range`.

## HTTP Contract and Reasoning Judges

- `base_url` may be a `/v1` root or the full `/v1/chat/completions` URL.
- Requests contain `model`, `messages`, `temperature: 0`, and `reward_judge.max_tokens`.
- Responses must use an OpenAI-compatible envelope. Strict v2 JSON belongs in `choices[0].message.content`; compatible services may use `choices[0].text`.
- Returning v2 JSON as the raw HTTP body is insufficient; it must be inside the chat completions envelope.
- Connectivity and response format are validated on the first scoring request; preflight does not proactively call the remote API.

## Asynchronous Scoring and Synchronized Updates

GRPO retains online rollouts and synchronized weight updates:

```text
rollout batch
  -> generate candidates with the current policy
  -> score them concurrently with the external judge
  -> wait for the complete reward batch
  -> compute group advantage
  -> run synchronized backpropagation and optimizer update
```

Asynchrony only reduces external-judge I/O wait. It does not pre-generate the full dataset, create an offline rollout buffer, or execute concurrent `optimizer.step()` calls. Candidates in a batch come from the same current policy, and group-relative advantage is computed only after the complete score set returns.

`reward_judge.max_concurrency` accepts `"auto"` or an integer from `1` to `64`:

- `"auto"` resolves to the current `grpo.num_generations`.
- Effective concurrency is always `min(resolved cap, completion count in the current reward batch)`.
- An explicit integer is the request cap for one training process and can be shared across prompts in the same reward batch.

| Prompts | Answers each | Requests | Setting | Effective concurrency |
|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | `auto` | 3 |
| 1 | 3 | 3 | `4` | 3 |
| 2 | 3 | 6 | `auto` | 3 |
| 2 | 3 | 6 | `4` | 4 |
| 2 | 4 | 8 | `2` | 2 |

The semaphore cap applies separately to each training process/rank. The current single-GPU run has one process, so it is also the whole-run cap. The underlying `asyncio.to_thread` executor can have fewer workers, so a large configured cap does not guarantee the same number of simultaneous physical sockets.

Each completion independently uses `max_retries`, exponential backoff with jitter based on `retry_backoff_seconds`, and valid `Retry-After` delays from HTTP 429/503. All concurrent tasks settle and rewards are restored in input order. If any request ultimately fails, the complete reward batch fails; successful partial scores are discarded and no neutral reward is substituted.

Per-batch logs record `configured_max_concurrency`, `effective_concurrency`, `completion_count`, `judge_batch_latency_seconds`, and `retry_backoff_seconds`. Training metadata stores configuration values, the latest batch metrics, batch count, maximum effective concurrency, total completions, and mean batch latency; the training report presents the same summary.

Reasoning judges can consume the output budget in `reasoning_content` and leave `message.content` empty. Start with:

```yaml
grpo:
  reward_judge:
    max_tokens: 4096
    timeout_seconds: 120
```

Do not confuse the two length settings:

- `grpo.max_completion_length` limits candidate responses generated by the policy being trained.
- `grpo.reward_judge.max_tokens` is the judge response budget for each scoring request.

When candidates frequently hit `max_completion_length`, inspect `completions/clipped_ratio`, improve EOS/stop behavior first, and then decide whether to raise the candidate limit. When the judge returns empty `content`, increase the judge budget or adjust the judge service, not the policy completion length.

## GRPO Starting Adapter and Resume Modes

GRPO resolves its starting adapter in this order:

```text
explicit grpo.base_adapter_dir -> DPO -> Fact-SFT -> CPT
```

A directory is a valid PEFT adapter only when its root contains `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin`. Incomplete directories are recorded and skipped while resolution continues. Logs report the actual stage and path selected.

Run GRPO independently:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml
```

Select the starting adapter explicitly:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --base_adapter_dir outputs/fact_sft_adapter
```

Continue through the full pipeline from existing adapters:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

These select a previous-stage adapter as the GRPO starting point. They differ from `grpo.resume_from_checkpoint`, which restores Trainer state from a checkpoint of the same GRPO run.

If every candidate is invalid and `require_base_adapter: true`, the error lists checked paths, missing files, and continuation commands. Set `require_base_adapter: false` only when intentionally creating a fresh PEFT adapter from the base model.

## Training Acceptance

`Status: completed` or finite loss alone does not prove that training worked. At minimum:

- Confirm finite `grad_norm` across CPT, Fact-SFT, DPO, and GRPO; the default `abort_on_nonfinite_grad_norm: true` should stop immediately on failure.
- Use `bf16: auto`, `fp16: auto`, and `torch_dtype: auto`, and confirm the recorded effective precision matches the hardware.
- Compare LoRA tensors between adjacent adapters to verify that parameters actually changed.
- Inspect judge reward mean and variance; near-constant rewards provide no useful ranking signal.
- Inspect `completions/clipped_ratio`; high clipping means the judge repeatedly sees incomplete candidates.
- Treat post-training quality evaluation as a heuristic smoke gate, not safety certification. Production release requires human review or an independent judge over safety cases.

## Failure Policy

These conditions fail during preflight or the first request; the system never substitutes a neutral score:

- Judge enabled without `base_url`, `model`, or a valid key.
- Judge disabled with an empty `builtin_rewards` list.
- Built-in reward mode with rows that have no matching signal.
- HTTP failure after retries or a response outside the chat completions envelope.
- Empty `message.content`, invalid strict v2 JSON, missing or extra fields, out-of-range dimensions, or unknown violations.

Fix the config, service, or data before resuming training.
