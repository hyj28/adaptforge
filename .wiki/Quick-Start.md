# Quick Start

Use this page when you want the shortest path from a fresh checkout to a verified AdaptForge setup.

The quick smoke path verifies wiring only. It does not mean a real model has been trained. Real training still requires replacing data, configuring the base model, preparing GPU resources, and checking output quality.

## Prerequisites

- A Python environment compatible with the chosen PyTorch build.
- A local checkout of the main repository.
- Enough disk space for `outputs/` if you run smoke tests or training.
- A CUDA GPU for real training. CPU is only practical for smoke wiring.

## 1. Create an Environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python --version
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python --version` prints nothing, Windows may be resolving the WindowsApps placeholder. Use `py` or `\.venv\Scripts\python.exe` for subsequent commands. Expected result: the environment contains PyTorch, Transformers, Datasets, PEFT, TRL, Flask, and the other default pipeline dependencies.

## 2. Copy the Default Config

```bash
cp configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Windows PowerShell:

```powershell
Copy-Item configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Edit `configs/domain_post_training.local.yaml`. This name pattern should be ignored by Git and can hold the plaintext `grpo.reward_judge.api_key`. Keep tracked `configs/domain_post_training.yaml` as the key-free template.

DPO and GRPO are enabled by default. Before real training, prepare both datasets and set the judge `base_url`, `model`, and `api_key` in the local YAML. For reasoning judges, start with `max_tokens: 4096` and `timeout_seconds: 120`. The default `max_concurrency: "auto"` follows `num_generations`; scoring is concurrent within the current rollout batch, and weight updates remain synchronized after the full batch completes.

## 3. Run the CPU Smoke Test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Expected result:

- A tiny smoke model is created under `outputs/smoke/base_model`.
- Dataset preparation, PEFT adapter saving, adapter merge, and report generation are exercised.
- The command does not download the default base model.

## 4. Static Check When ML Dependencies Are Unavailable

```bash
python -m compileall pipeline scripts serve_inference.py
```

Expected result: Python files compile without syntax errors. This does not prove that GPU training dependencies are installed.

## 5. Next Pages

Download the base model and run the complete chain:

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

A completed run is not proof of production quality. Check finite `grad_norm` across all stages, real LoRA tensor changes, judge reward variance, and GRPO clipping, and independently review safety cases.

- Replace the mock data: [Data Contracts](Data-Contracts.md)
- Check the training environment and base model: [Operations Runbook](Operations-Runbook.md)
- Change paths or training settings: [Configuration](Configuration.md)
- Run the full training workflow: [Training Pipeline](Training-Pipeline.md)
- Diagnose failures: [Troubleshooting](Troubleshooting.md)
