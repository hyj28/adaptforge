# Scripts

Run commands from the repository root. Script entrypoints are grouped by workflow:

| Directory | Purpose |
| --- | --- |
| `training/` | CPT, Fact-SFT, DPO, GRPO, and full-pipeline training |
| `model_artifacts/` | Model download, adapter merge, GGUF and ONNX export |
| `inference/` | Inference, quality evaluation, and DPO rejected-answer completion |
| `diagnostics/` | Python, dependency, CUDA, and GPU checks |
| `_shared/` | Internal project-root discovery and import bootstrap |

Use `--help` on an entrypoint for its command-line options. The default configuration is [domain_post_training.yaml](../configs/domain_post_training.yaml); see the [configuration reference](../configs/README.md) for path-resolution rules and settings.
