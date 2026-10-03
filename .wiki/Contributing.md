# Contributing

Use this page when preparing code, data, or documentation changes for AdaptForge.

## Contribution Boundary

Keep the repository domain-neutral. Do not add:

- proprietary customer data
- real credentials or tokens
- private URLs
- internal system prompts
- product-specific corpora
- license-restricted training data

Before open-sourcing, check `data/`, `configs/`, `outputs/`, `models/`, and Wiki examples to avoid leaking private information.

## Recommended Checks

Run:

```bash
python -m compileall pipeline scripts
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

If the smoke test cannot run because the local machine lacks ML dependencies, include:

- the exact failure reason
- the static checks you did run
- any environment constraints, such as no CUDA or missing package manager

## Documentation Changes

When changing behavior:

1. Update `README.md` only for high-level project behavior or fastest-start guidance.
2. Update `configs/README.md` when config semantics change.
3. Update `.wiki/` pages when user procedures, troubleshooting, or publishing guidance changes.
4. Update `Data-Contracts.md` when any JSON/JSONL row contract changes.
5. Update `Troubleshooting.md` when a fixed bug can still affect older versions or common setups.

## Test Expectations for GRPO Changes

At minimum, run:

```bash
py -m unittest tests.test_grpo_core
py -m compileall pipeline scripts tests
```

GRPO reward judge changes should cover:

- config validation
- disabled judge behavior
- enabled judge builder behavior
- JSON score parsing
- score clamping
- secret-free metadata

## Security Reports

For security issues, follow `SECURITY.md`: use the repository's private security reporting channel, and include reproduction steps, expected impact, and any safe mitigation already tested.
