# Inference And Export

Use this page after training or when you want to serve, merge, or export a model.

## Merge Adapter into a Full Model

The pipeline can merge automatically. When `merge.adapter_dir` is `null` and no `--adapter_dir` argument is provided, merge selects among enabled stages:

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

To merge a specific existing adapter, pass it directly:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml --adapter_dir outputs/grpo_adapter
```

Expected output:

```text
outputs/merged_model/
```

## Run Single-Text Inference

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.yaml --model_path outputs/merged_model --device auto "What can this assistant answer from the documentation?"
```

Expected result: the script loads `outputs/merged_model` and prints the model completion.

Common flags:

| Flag | Purpose |
|---|---|
| `--device cuda|cuda:0|cpu|auto` | Select inference device; default is `cuda`. |
| `--dtype auto|float16|bfloat16|float32` | Override load dtype. |
| `--system_prompt` | Override the default system prompt. |
| `--raw_prompt` | Use input text directly without chat prompt wrapping. |
| `--allow_markdown` | Do not strip markdown. |
| `--return_reasoning` | Return model-emitted `<think>` content for debugging. |

## Start the Flask Service

```bash
python serve_inference.py --host 0.0.0.0 --port 8000 --model_path outputs/merged_model --device auto
```

Service endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check with model path, served model name, device, and OpenAI-compatible status. |
| `GET /openapi.json` | OpenAPI schema. |
| `GET /docs` | Browser docs page. |
| `GET /v1/models` | OpenAI-compatible model list. |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions. |
| `POST /generate` | Simple generation endpoint. |
| `POST /` | Alias for `/generate`. |

Health check:

```bash
curl http://localhost:8000/health
```

OpenAI-compatible request example:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-domain-model",
    "messages": [{"role": "user", "content": "What can this assistant answer from the documentation?"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Common service flags:

| Flag or env var | Purpose |
|---|---|
| `--device` / `FLASK_INFRA_DEVICE` | Defaults to `cuda`; set `auto` or `cpu` when needed. |
| `--dtype` / `FLASK_INFRA_DTYPE` | Defaults to `float16`. |
| `--served_model_name` | Model name exposed by `/v1/models` and chat completions. |
| `--raw_prompt` | Do not wrap prompts server-side. |
| `--allow_markdown` | Do not strip markdown. |
| `--return_reasoning` | Return a reasoning field for debugging. |

Compatibility boundary: the service provides basic OpenAI-compatible chat completions. Do not assume streaming, tools/function calling, or multimodal input support.

## Export GGUF

GGUF is the recommended default local-deployment export path. Prepare a local `llama.cpp` checkout or use an available Unsloth backend before running GGUF export.

```bash
python scripts/model_artifacts/export_gguf.py --config configs/domain_post_training.yaml --backend llama_cpp --llama_cpp_dir /path/to/llama.cpp
```

Output is controlled by:

```yaml
gguf:
  output_dir: "models/gguf"
  output_name: "AdaptForge-Q4_K_M.gguf"
  quantization_method: "q4_k_m"
```

## Export ONNX

Install optional dependencies first:

```bash
python -m pip install -r requirements-onnx.txt
```

The default config sets `onnx.device` to `cuda`. On CPU-only machines, pass `--device cpu` or change the config:

```bash
python scripts/model_artifacts/export_onnx.py --config configs/domain_post_training.yaml --device cpu
```

Common flags:

| Flag | Purpose |
|---|---|
| `--model_path` | Merged model directory; defaults to `training.merged_output_dir`. |
| `--output_dir` | ONNX output directory; defaults to `onnx.output_dir`. |
| `--opset` | ONNX opset; default config is `17`. |
| `--external_data` / `--no_external_data` | Toggle ONNX external data. |
| `--validate` / `--no_validate` | Toggle `onnx.checker` validation. |
| `--ort_check` | Run an ONNX Runtime shape check. |

ONNX export settings live under `onnx` in `configs/domain_post_training.yaml`.

## Related Pages

- [Installation](Installation.md)
- [Configuration](Configuration.md)
- [Operations Runbook](Operations-Runbook.md)
- [Troubleshooting](Troubleshooting.md)
