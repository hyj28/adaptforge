# AdaptForge Wiki

AdaptForge is a reproducible LLM post-training pipeline for turning domain documents, factual SFT examples, preference data, and GRPO reward prompts into LoRA/QLoRA adapters, a merged model, quality evaluation reports, local inference services, and export artifacts.

These English pages provide task-oriented setup, training, and troubleshooting guides.

`.wiki/` in the main repository is the source directory for GitHub Wiki pages. To make changes visible to readers, copy these Markdown files to the separate `OWNER/REPO.wiki.git` repository and push them. See [Publishing](Publishing.md).

## Start Here

- First-time user: [Quick Start](Quick-Start.md)
- Installing dependencies: [Installation](Installation.md)
- Pre-training environment, model, and failure reports: [Operations Runbook](Operations-Runbook.md)
- Changing training behavior: [Configuration](Configuration.md)
- Replacing sample data: [Data Contracts](Data-Contracts.md)
- Running the full pipeline: [Training Pipeline](Training-Pipeline.md)
- Using GRPO and reward judging: [GRPO And Reward Judge](GRPO-and-Reward-Judge.md)
- Serving or exporting a model: [Inference And Export](Inference-and-Export.md)
- Something failed: [Troubleshooting](Troubleshooting.md)
- Contributing changes: [Contributing](Contributing.md)

## Common Tasks

| I want to... | Read this |
|---|---|
| Run the shortest local verification | [Quick Start](Quick-Start.md) |
| Check CUDA, PyTorch, TRL, and GPU state | [Operations Runbook](Operations-Runbook.md) |
| Download or point to a base model | [Operations Runbook](Operations-Runbook.md) |
| Install CUDA, PyTorch, or ONNX dependencies | [Installation](Installation.md) |
| Replace the mock domain with my own data | [Data Contracts](Data-Contracts.md) |
| Tune batch size, LoRA rank, validation, or output paths | [Configuration](Configuration.md) |
| Enable DPO or GRPO | [Training Pipeline](Training-Pipeline.md) |
| Use a local, DeepSeek, or GLM reward judge | [GRPO And Reward Judge](GRPO-and-Reward-Judge.md) |
| Inspect failure reports and resume work | [Operations Runbook](Operations-Runbook.md) |
| Export GGUF or ONNX | [Inference And Export](Inference-and-Export.md) |
| Publish `.wiki` to GitHub Wiki | [Publishing](Publishing.md) |

## Default Pipeline

```text
CPT -> Fact-SFT -> DPO -> GRPO -> merge -> quality eval -> inference/export
```

The repository ships static mock data for the fictional `AsterHelp` domain. Before training or publishing a derivative project, replace the corpus, SFT rows, preference rows, reward prompts, quality evaluation questions, and system prompts with data you are licensed to use.

## Primary Repository Docs

The Wiki is task-oriented. The main repository keeps compact project-level references:

- `README.md`: project overview and core workflow.
- `configs/README.md`: full configuration reference.
- `data/README.md`: packaged mock data summary and publication warning.
- `scripts/README.md`: script grouping.
- `CONTRIBUTING.md`: contribution checks.
- `SECURITY.md`: security reporting and data safety boundary.

## Runtime Assumptions

- Default dependencies target CUDA 12.6 GPU training.
- Before training, run `python scripts/diagnostics/check_training_environment.py`.
- ONNX export is optional and uses `requirements-onnx.txt`.
- GRPO reward-model scoring uses `grpo.reward_judge` to call an OpenAI-compatible chat completions API; scoring can run concurrently within the current rollout batch while weight updates remain synchronized after the batch completes.
- GitHub Wiki content is maintained in `.wiki/` and must be copied to the separate `OWNER/REPO.wiki.git` repository to go live.
