# Data Contracts

Use this page when replacing the packaged mock data with your own domain data.

Before publishing a derivative repository, confirm that `data/` does not contain private documents, customer data, credentials, private system prompts, source code, or license-restricted corpora.

## Packaged Data Layout

```text
data/cpt/source_documents/*.md      # CPT source documents
data/sft/*.jsonl                    # Fact-SFT examples
data/dpo/preference_examples.jsonl  # DPO preference pairs
data/grpo/reward_examples.jsonl     # GRPO reward examples
data/eval/quality_questions.jsonl   # post-training quality evaluation questions
```

The sample domain is `AsterHelp`, a fictional internal support knowledge-base assistant. It is deliberately small and static.

## CPT Documents

Input:

```text
data/cpt/source_documents/*.md
```

Purpose:

- Teach domain concepts, policies, procedures, safety boundaries, and troubleshooting knowledge.
- Feed corpus discovery, safety preflight, and coverage-aware dataset construction.

Important config:

```yaml
corpus:
  input_paths:
    - "../data/cpt/source_documents"
```

## Fact-SFT Rows

Input:

```text
data/sft/*.jsonl
```

Recommended fields:

| Field | Meaning |
|---|---|
| `instruction` | User task or question. |
| `output` | Expected assistant answer. |
| `input` or `context` | Optional context appended to the user prompt. |
| `category` or `type` | Optional category for reports. |

The loader also accepts:

- `question` + `answer`
- `prompt` + `response`
- `messages`, where the first assistant message becomes the answer and earlier system/user messages become the prompt.

Fact-SFT uses assistant-only loss, so prompt tokens are masked and only answer tokens train the model.

## DPO Rows

Input:

```text
data/dpo/preference_examples.jsonl
```

Recommended fields:

| Field | Meaning |
|---|---|
| `prompt` | User prompt or task. |
| `chosen` | Preferred answer. |
| `rejected` | Less preferred answer. |
| `category` or `type` | Optional category for reports. |

Accepted aliases:

| Standard field | Accepted aliases |
|---|---|
| `prompt` | `instruction`, `question` |
| `chosen` | `preferred`, `accept` |
| `rejected` | `bad`, `reject` |

Rules:

- `prompt`, `chosen`, and `rejected` must be non-empty.
- `chosen` must differ from `rejected`.

## GRPO Rows

Input:

```text
data/grpo/reward_examples.jsonl
```

Required baseline:

- A prompt must be constructible.
- Prompt-only rows are valid when the judge is enabled.
- With the judge disabled, every row needs a signal matching an enabled built-in reward.

Prompt can come from:

- `messages`
- `prompt`
- `instruction`
- `question`

Optional `context` or `input` is appended to the prompt. `include_system_prompt: false` prevents the configured system prompt from being added to that example.

Reward signal fields:

| Field | Meaning |
|---|---|
| `reference_answer` | Reference text for overlap-style scoring and judge context. |
| `answer`, `solution`, `ground_truth`, `expected` | Accepted aliases for `reference_answer`. |
| `trusted_context`, `judge_context` | Trusted judge evidence; `context` / `input` also act as fallbacks. |
| `required_terms` | Terms the completion should include. |
| `must_include`, `keywords` | Accepted aliases for `required_terms`. |
| `forbidden_terms` | Terms the completion should avoid. |
| `must_not_include`, `banned_terms` | Accepted aliases for `forbidden_terms`. |
| `forbidden_terms_mode` | `semantic` (default) evaluates meaning; `literal` uses case-insensitive text occurrence. |
| `must_refuse` | Whether the correct behavior is refusal. |
| `requires_refusal` | Accepted alias for `must_refuse`. |
| `min_completion_chars` | Optional lower length bound. |
| `max_completion_chars` | Optional upper length bound. |
| `category`, `type`, `task` | Optional grouping label for reports and judge context. |

Built-in rewards map directly to fields: `reference_overlap` uses `reference_answer`; `term_constraints` uses required/forbidden terms; `refusal` uses a valid refusal requirement; and `length_bounds` uses a minimum or maximum character count. All four built-in rewards are disabled by default and activate only when explicitly listed in `builtin_rewards`.

These fields also provide context for the external `reward_judge`. The judge is enabled by default and uses the strict five-dimension `grpo_judge_v2` response. Without a reference answer or trusted context, `factual_grounding` must be `null`. Model-based scoring does not load local model paths or Hub IDs; configure an HTTP judge through [GRPO And Reward Judge](GRPO-and-Reward-Judge.md).

## Quality Evaluation Rows

Input:

```text
data/eval/quality_questions.jsonl
```

Fields:

| Field | Meaning |
|---|---|
| `category` | Must be `domain_knowledge`, `safety_boundary`, or `base_regression`. |
| `question` | Non-empty prompt used for post-training quality evaluation. |

Quality evaluation is not a training validation set. It runs after training or merge to inspect output behavior; the current rules are a heuristic smoke gate, not production safety certification.

## Publication Checklist

Before publishing a derivative repository:

1. Confirm all data is public, licensed, and safe to distribute.
2. Remove private paths, customer names, internal URLs, secrets, and proprietary prompts.
3. Check `configs/domain_post_training.yaml` for private defaults.
4. Keep `outputs/`, `models/`, `.venv/`, `__pycache__/`, and training artifacts out of commits.
