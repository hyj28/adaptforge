# AdaptForge

A domain-focused LLM post-training pipeline built with LoRA/QLoRA. AdaptForge turns source documents, factual examples, preference pairs, and reward prompts into adapters, a merged model, and evaluation reports.

```text
CPT -> Fact-SFT -> DPO -> GRPO -> merge -> quality evaluation -> inference/export
```

- **CPT:** corpus discovery, safety preflight, coverage-aware dataset preparation, and domain adaptation.
- **Fact-SFT:** supervised fine-tuning with assistant-only loss.
- **DPO:** preference alignment from complete `prompt` / `chosen` / `rejected` pairs.
- **GRPO:** online reward optimization with an OpenAI-compatible judge or explicitly enabled built-in rewards.
- **Deployment:** adapter merging, local inference, an OpenAI-compatible Flask API, GGUF export, and optional ONNX export.

The included `AsterHelp` domain is fictional. All bundled data is small, static mock data for checking the workflow; it does not demonstrate production model quality.

## Install

Run commands from the repository root. The default [requirements.txt](requirements.txt) targets **CUDA 12.6 GPU training**. For another CUDA version or a CPU-only environment, install a matching PyTorch build and the remaining dependencies as described in the [installation guide](.wiki/Installation.md).

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the environment before training:

```bash
python scripts/diagnostics/check_training_environment.py
```

## Verify the Workflow

With ML dependencies installed, run a CPU smoke test:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

This creates a tiny local model in `outputs/smoke/base_model` and checks dataset preparation, adapter saving, merging, and report generation without downloading the configured base model. It is a workflow check, not a full training run.

Unit tests require PyYAML and PyTorch; a CPU build is sufficient. They do not train a model or call an external judge:

```bash
python -m pip install PyYAML
python -m compileall -q pipeline scripts tests serve_inference.py
python -m unittest discover -s tests -v
```

## Configure and Train

Copy the [default configuration](configs/domain_post_training.yaml) to a Git-ignored local file:

```bash
cp configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

On PowerShell, use `Copy-Item` instead of `cp`.

Before real training:

1. Replace the mock data and `fact_sft.system_prompt` with your domain content and boundaries. See [data/README.md](data/README.md) for row schemas and reward signals.
2. Set `base_model_repo_id` to the Hugging Face download source and `base_model_name_or_path` to the local model directory. The template uses `Qwen/Qwen3.5-0.8B` and `models/base-model`.
3. Review each stage's precision, batch size, sequence length, and output settings in the [configuration reference](configs/README.md).
4. **DPO and GRPO are enabled by default.** Supply preference data and configure `grpo.reward_judge.base_url`, `model`, and credentials, or disable stages you do not need. Keep real API keys in the ignored local configuration or use `api_key_env`; the tracked template keeps `api_key: null`.
5. Enable training validation when you have enough data. CPT supports separate held-out sources; Fact-SFT and DPO support `validation_ratio`. The small mock dataset disables validation by default.

Download the model, then train:

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

If the model already exists locally, point `base_model_name_or_path` to it and skip downloading.

GRPO uses an external OpenAI-compatible chat completions service by default. Scoring runs concurrently within each rollout batch; the complete batch is scored before the policy update. A prompt-only row is valid, but private-domain factual grading needs separate `trusted_context` or `reference_answer` evidence. Failed judge requests fail the batch rather than substitute neutral rewards. See the [GRPO guide](.wiki/GRPO-and-Reward-Judge.md) for the strict judge response contract, retries, built-in rewards, and resume behavior.

Standalone GRPO and explicit stage skipping are supported:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --base_adapter_dir outputs/dpo_adapter
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

These commands start GRPO from an existing adapter. `grpo.resume_from_checkpoint` instead resumes an interrupted run of the same GRPO stage. See the [training guide](.wiki/Training-Pipeline.md) for all stage controls.

## Evaluate the Result

Training validation measures loss during training. Post-training quality evaluation checks domain answers, safety boundaries, and base-capability regressions using [data/eval/quality_questions.jsonl](data/eval/quality_questions.jsonl).

```bash
python scripts/inference/run_quality_evaluation.py --config configs/domain_post_training.local.yaml --targets merged
```

The default quality gate requires all configured categories to pass. Failure preserves training artifacts but returns exit code `8`, records `quality_gate_failed`, and sets `release_ready: false`. Explicit `--skip_eval` permits continuation without marking the result release-ready.

The evaluator is a heuristic smoke gate. Acceptance also requires finite `grad_norm` values, actual LoRA tensor changes between stages, useful reward variation, and independent review of safety examples. Keep `abort_on_nonfinite_grad_norm: true`; automatic precision settings select BF16 on supported CUDA devices, FP16 on other CUDA devices, and no mixed precision on CPU.

| Output | Default path |
| --- | --- |
| CPT adapter | `outputs/lora_adapter/` |
| Fact-SFT adapter | `outputs/fact_sft_adapter/` |
| DPO adapter, when enabled | `outputs/dpo_adapter/` |
| GRPO adapter, when enabled | `outputs/grpo_adapter/` |
| Merged model | `outputs/merged_model/` |
| Corpus coverage report | `outputs/cpt_dataset/coverage_report.md` |
| Quality evaluation report | `outputs/eval/eval_report.md` |
| Pipeline report | `outputs/reports/pipeline_report.md` |

## Inference and Export

Query a merged model:

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.local.yaml --model_path outputs/merged_model "What can this assistant answer from the documentation?"
```

Start the service:

```bash
python serve_inference.py --host 127.0.0.1 --port 8000 --model_path outputs/merged_model
```

The API exposes `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, and `POST /generate`. OpenAPI is available at `/openapi.json`, with interactive documentation at `/docs`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local-domain-model","messages":[{"role":"user","content":"What can this assistant answer from the documentation?"}],"temperature":0,"max_tokens":128}'
```

Export the merged Hugging Face model to GGUF using a local llama.cpp checkout:

```bash
python scripts/model_artifacts/export_gguf.py --config configs/domain_post_training.local.yaml --backend llama_cpp --llama_cpp_dir /path/to/llama.cpp
```

The default filename is `AdaptForge-Q4_K_M.gguf`. ONNX export is optional and requires [requirements-onnx.txt](requirements-onnx.txt):

```bash
python -m pip install -r requirements-onnx.txt
python scripts/model_artifacts/export_onnx.py --config configs/domain_post_training.local.yaml
```

See the [inference and export guide](.wiki/Inference-and-Export.md) for merging and deployment details.

## Repository Guide

| Path | Contents |
| --- | --- |
| `pipeline/` | Reusable data, training, evaluation, inference, and export logic |
| [scripts/README.md](scripts/README.md) | Command-line entrypoints grouped by workflow |
| [configs/README.md](configs/README.md) | Complete parameter reference |
| [data/README.md](data/README.md) | Mock data, including `data/dpo/preference_examples.jsonl` and `data/grpo/reward_examples.jsonl` |
| [.wiki/Home.md](.wiki/Home.md) | Detailed setup, training, and troubleshooting guides |
| `tests/` | Lightweight unit tests |

Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution checks and [SECURITY.md](SECURITY.md) for credential handling and remote-judge data sharing. Keep private configurations, model weights, and generated outputs out of commits.

## License

Code and bundled mock data use [Apache License 2.0](LICENSE). Check the license terms of any base model, dataset, or domain documents you add.

This version modifies the source snapshot's project identifiers, documentation, and script descriptions.
