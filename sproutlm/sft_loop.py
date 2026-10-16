"""Supervised fine-tuning loop (plain PyTorch; no HF Trainer).

Single GPU:   python -m sproutlm.sft_loop --config configs/sft.yaml
Both T4s:     torchrun --nproc_per_node 2 -m sproutlm.sft_loop --config configs/sft.yaml

Gradient accumulation is derived from `effective_batch`, so the global batch stays 32
whether one or two GPUs are used.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import runtime, sources
from .packing import _pad_id, collate_segments, segment_row
from .scoring import sft_loss, token_logps


def prepare_rows(cfg: dict, run_dir: Path, rank: int, world: int):
    """Rank 0 samples and freezes the subsets; everyone reads the frozen files."""
    train_path, val_path = run_dir / "data" / "train.jsonl", run_dir / "data" / "val.jsonl"
    if rank == 0 and not train_path.exists():
        train, val = sources.load_splits(cfg["dataset"], cfg["train_examples"], cfg["val_examples"], cfg["seed"])
        sources.freeze(train, train_path)
        sources.freeze(val, val_path)
    if world > 1:
        dist.barrier()
    return sources.thaw(train_path), sources.thaw(val_path)


@torch.no_grad()
def validation_loss(model, tokenizer, segs, batch_size: int, device) -> float:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pad = _pad_id(tokenizer)
    order = sorted(range(len(segs)), key=lambda i: segs[i].length)
    nll, count = 0.0, 0
    for s in range(0, len(order), batch_size):
        batch = collate_segments([segs[i] for i in order[s : s + batch_size]], pad)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            per_token, keep = token_logps(model, batch)
        nll -= per_token.sum().item()
        count += int(keep.sum().item())
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return nll / max(count, 1)


def main() -> None:
    cfg = runtime.cli_config()
    rank, local_rank, world = runtime.dist_info()
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    lead = rank == 0
    runtime.seed_everything(cfg["seed"] + rank)

    run_dir = Path(cfg["output_dir"])
    if lead:
        runtime.dump_config(cfg, run_dir)
    ledger = runtime.Ledger(run_dir, enabled=lead)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, val_rows = prepare_rows(cfg, run_dir, rank, world)
    L = cfg["max_length"]
    train_segs = [segment_row(tokenizer, r["prompt"], r["response"], L) for r in train_rows]
    val_segs = [segment_row(tokenizer, r["prompt"], r["response"], L) for r in val_rows]
    train_segs = [s for s in train_segs if s.response_ids]   # prompt filled the window: nothing to learn

    per_dev = cfg["per_device_batch"]
    accum = max(1, cfg["effective_batch"] // (per_dev * world))
    batches = runtime.bucketed_order([s.length for s in train_segs], per_dev, cfg["seed"])
    batches = batches[rank::world][: len(batches) // world]          # equal share per rank
    total_steps = len(batches) // accum

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    core = getattr(model, "module", model)

    optim = torch.optim.AdamW(
        runtime.split_decay_groups(core, cfg["weight_decay"]), lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, runtime.lr_shape(cfg["schedule"], total_steps, cfg["warmup_frac"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    budget = runtime.TimeBudget(cfg.get("time_budget_minutes"))

    if lead:
        print(f"[sprout-sft] world={world} per_device={per_dev} accum={accum} "
              f"effective={per_dev * accum * world} steps={total_steps}", flush=True)
        ledger.write("val", 0, val_loss=validation_loss(core, tokenizer, val_segs, cfg["eval_batch"], device))

    pad = _pad_id(tokenizer)
    best, step, window_loss, window_tokens = float("inf"), 0, 0.0, 0
    model.train()
    for micro, idx in enumerate(batches[: total_steps * accum]):
        boundary = (micro + 1) % accum == 0
        batch = collate_segments([train_segs[i] for i in idx], pad)
        sync = contextlib.nullcontext() if (boundary or world == 1) else model.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss, n_tok = sft_loss(model, batch)
            scaler.scale(loss / accum).backward()
        window_loss += loss.item() / accum
        window_tokens += n_tok
        if not boundary:
            continue

        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), cfg["grad_clip"]).item()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        sched.step()
        step += 1

        if step % cfg["log_every"] == 0:
            ledger.write("train", step, loss=window_loss, grad_norm=grad_norm,
                         lr=sched.get_last_lr()[0], tokens=window_tokens, scale=scaler.get_scale())
        window_loss, window_tokens = 0.0, 0

        stop = budget.expired()
        if step % cfg["eval_every"] == 0 or step == total_steps or stop:
            if lead:
                v = validation_loss(core, tokenizer, val_segs, cfg["eval_batch"], device)
                ledger.write("val", step, val_loss=v)
                if v < best:
                    best = v
                    runtime.save_checkpoint(core, tokenizer, run_dir / "best", {"step": step, "val_loss": v, "base": cfg["model"]})
            if world > 1:
                dist.barrier()
        if stop:
            if lead:
                print("[sprout-sft] time budget reached, stopping early", flush=True)
            break

    if lead:
        runtime.save_checkpoint(core, tokenizer, run_dir / "final", {"step": step, "best_val_loss": best, "base": cfg["model"]})
        ledger.write("done", step, best_val_loss=best)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
