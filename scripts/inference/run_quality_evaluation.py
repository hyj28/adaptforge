"""Post-training quality evaluation entrypoint for base, adapter, or merged models.
Use it after training or merge to check factuality, refusal boundaries, and base-capability regressions; this is not the training validation set.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)

from pipeline.cuda_bootstrap import ensure_pip_cuda_libraries_preferred

ensure_pip_cuda_libraries_preferred()

from pipeline.evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
