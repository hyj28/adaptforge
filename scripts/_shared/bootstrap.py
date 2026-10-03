"""Shared bootstrap helper for locating the project root from any scripts subdirectory and updating sys.path.
Use it when adding or moving command-line scripts before importing the pipeline package.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root(script_file: str | Path) -> Path:
    """Find the repository root from a nested script path and add it to sys.path."""
    script_path = Path(script_file).resolve()
    for parent in script_path.parents:
        if (parent / "pipeline").is_dir() and (parent / "configs" / "domain_post_training.yaml").is_file():
            root = parent
            break
    else:
        raise RuntimeError(f"Could not find AdaptForge project root from {script_path}")

    for path in (script_path.parent, root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    return root
