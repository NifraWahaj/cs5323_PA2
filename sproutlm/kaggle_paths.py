"""Find artifacts from earlier notebooks under /kaggle/input, wherever Kaggle mounted them."""

from __future__ import annotations

from pathlib import Path

INPUT = Path("/kaggle/input")


def find_run(run_name: str, which: str = "best", root: Path = INPUT) -> Path:
    """Return .../<run_name>/<which> that contains a sprout_note.json, e.g. find_run('sft')."""
    hits = sorted(p.parent for p in root.rglob("sprout_note.json")
                  if p.parent.name == which and p.parent.parent.name == run_name)
    if not hits:
        raise FileNotFoundError(f"no {run_name}/{which} checkpoint under {root}; is the notebook output attached?")
    return hits[0]


def find_file(name: str, root: Path = INPUT) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def find_dir(name: str, root: Path = INPUT) -> Path:
    hits = sorted(p for p in root.rglob(name) if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"directory {name} not found under {root}")
    return hits[0]
