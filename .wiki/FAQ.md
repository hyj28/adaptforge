# FAQ

Use this page for conceptual questions. Use [Troubleshooting](Troubleshooting.md) for exact errors and symptoms.

## Is `.wiki/` Automatically Published as the GitHub Wiki?

No. GitHub Wiki content lives in a separate repository named like `OWNER/REPO.wiki.git`. The main repository's `.wiki/` directory is the source. See [Publishing](Publishing.md).

## What Is the Difference Between Validation and Quality Evaluation?

Validation provides loss/eval signals during training. Quality evaluation checks factual answers, safe refusals, and regressions after training. The built-in evaluation is a heuristic smoke gate, not production safety certification; independently judge or manually review safety cases before release.

## Are `base_model_repo_id` and `base_model_name_or_path` Duplicates?

No. `base_model_repo_id` is the source used by `download_models.py`; `base_model_name_or_path` is what training, merge, and inference actually load. The default config downloads the former into `models/base-model` and loads that local snapshot through the latter.

## Why Are DPO and GRPO Enabled by Default?

The default template represents the complete post-training chain. Real training requires DPO/GRPO data and a configured judge. Disable an unneeded stage in YAML or skip it with the pipeline's `--skip_*` flags.

## Can a GRPO Row Contain Only `prompt`?

Yes, when the external judge is enabled. With the judge disabled, every row needs at least one signal matching an enabled built-in reward.

## Does the External Judge Replace Built-in Rewards?

The judge is the default reward provider, and all four built-in rewards are disabled by default. Rewards listed in `builtin_rewards` run alongside the judge. If the judge is disabled, at least one built-in reward is required.

## Why Must a Local Judge Expose an OpenAI-Compatible API?

The training path uniformly calls `/v1/chat/completions` using `base_url`, `model`, and a key. Local models, DeepSeek, GLM, and other services use the same contract. The training client controls its per-process/rank request cap through `max_concurrency`, while the judge service remains responsible for deployment, server-side limits, capacity, and memory management.

## Where Should the Judge Key Live?

Put it directly in `reward_judge.api_key` inside the Git-ignored `configs/domain_post_training.local.yaml`. It is a plaintext secret and must not be committed or shared. `api_key_env` is only an optional fallback when `api_key` is empty.

## How Do `max_completion_length` and Judge `max_tokens` Differ?

`grpo.max_completion_length` limits candidate responses from the policy being trained. `grpo.reward_judge.max_tokens` limits each judge response. Candidate truncation points to the former and `completions/clipped_ratio`; empty judge `message.content` points to the latter. Start reasoning judges at `4096`.

## Is `max_concurrency` Useful Above `num_generations`?

It is an upper bound and never exceeds the number of completions in the current reward batch. One prompt with three answers and cap `4` still uses 3. If the same reward batch contains two prompts with three answers each, an explicit cap of `4` can score four requests across prompts, while `auto` still resolves to the per-prompt candidate count of 3. The semaphore cap applies per process/rank, and `asyncio.to_thread` executor capacity can make physical socket concurrency lower.

## Does an Asynchronous Judge Update the Model Concurrently?

No. The current policy generates one rollout batch, the judge scores it concurrently and the trainer waits for the complete batch, then computes group advantage and performs one synchronized weight update. The pipeline neither pre-generates the full dataset nor runs multiple optimizer updates concurrently. If any scoring request exhausts its independent retries, the complete batch fails rather than continuing with partial or neutral rewards.

## How Do I Know GRPO Really Updated the Model?

Do not rely only on finite loss or `Status: completed`. Confirm finite `grad_norm` across stages, compare LoRA tensors before and after GRPO, and inspect reward variance and `completions/clipped_ratio`.

## Can I Put Private Domain Documents in `data/`?

Do not publish private documents, credentials, customer tickets, internal prompts, source code, or license-restricted corpora. A remote judge also receives the full prompt, candidate response, and evaluation context; use a local judge when sensitive data cannot leave the environment.

## Is ONNX Required?

No. ONNX export is optional. The default local-deployment export path is GGUF.

## Why Did `pyyaml` Fail in a Local Check?

Without `pyyaml` in the active Python environment, YAML parsing fails with `ModuleNotFoundError: No module named 'yaml'`. Install dependencies in the same environment that runs the command.
