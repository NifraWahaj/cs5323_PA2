"""Plumbing shared by both loops: config, seeding, batching order, schedules, logging, saving."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml


# ---------------------------------------------------------------- config
def read_config(path: str, overrides: list[str]) -> dict:
    """YAML file plus `key=value` overrides from the command line (values parsed as YAML)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, _, raw = item.partition("=")
        cfg[key] = yaml.safe_load(raw)
    # YAML 1.1 reads "1e-6" (no dot) as a string; coerce anything float-shaped.
    for key, value in cfg.items():
        if isinstance(value, str):
            try:
                cfg[key] = float(value)
            except ValueError:
                pass
    return cfg


def cli_config() -> dict:
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        raise SystemExit("usage: python -m sproutlm.<loop> --config CONFIG.yaml [key=value ...]")
    return read_config(sys.argv[2], sys.argv[3:])


# ---------------------------------------------------------------- reproducibility
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- distributed (SFT only)
def dist_info() -> tuple[int, int, int]:
    """(rank, local_rank, world_size); (0, 0, 1) when not launched by torchrun."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


# ---------------------------------------------------------------- batching order
def bucketed_order(lengths: list[int], batch_size: int, seed: int, pool: int = 64) -> list[list[int]]:
    """Shuffled batches whose members have similar length.

    Shuffle all indices, cut them into pools of `pool * batch_size`, sort each pool by
    length, slice into batches, then shuffle the batches. Padding waste drops a lot while
    the batch order stays random.
    """
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    span = pool * batch_size
    batches = []
    for start in range(0, len(idx), span):
        chunk = sorted(idx[start : start + span], key=lengths.__getitem__)
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- schedules
def lr_shape(kind: str, total_steps: int, warmup_frac: float):
    """Multiplier on the peak LR for optimizer step `s` (0-based)."""
    warm = max(1, int(round(total_steps * warmup_frac))) if warmup_frac > 0 else 0

    def fn(s: int) -> float:
        if warm and s < warm:
            return (s + 1) / warm
        if kind == "constant":
            return 1.0
        progress = min(1.0, (s - warm) / max(1, total_steps - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))   # cosine to zero

    return fn


def split_decay_groups(model, weight_decay: float):
    """No weight decay on biases and norm weights (1-D tensors)."""
    decay, plain = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else plain).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}]


# ---------------------------------------------------------------- logging
class Ledger:
    """Append-only metrics.jsonl, flushed on every write so a dropped Kaggle session keeps it."""

    def __init__(self, run_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self.path = run_dir / "metrics.jsonl"
        self.t0 = time.time()

    def write(self, kind: str, step: int, **values) -> None:
        if not self.enabled:
            return
        record = {"kind": kind, "step": step, "minutes": round((time.time() - self.t0) / 60, 2)}
        record.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in values.items()})
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


class TimeBudget:
    """Stop training cleanly before Kaggle's session limit kills the kernel."""

    def __init__(self, minutes: float | None):
        self.deadline = None if not minutes else time.time() + 60 * minutes

    def expired(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(model, tokenizer, out_dir: Path, note: dict) -> None:
    """Save weights as fp16 safetensors (half the disk of fp32) plus a small provenance file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    core = getattr(model, "module", model)
    half = {k: v.detach().to(torch.float16).cpu() for k, v in core.state_dict().items()}
    core.save_pretrained(out_dir, state_dict=half, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "sprout_note.json").write_text(json.dumps(note, indent=2))


def dump_config(cfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
