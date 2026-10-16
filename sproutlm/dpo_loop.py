"""From-scratch DPO loop (plain PyTorch; no TRL).

    python -m sproutlm.dpo_loop --config configs/dpo_hh.yaml
    python -m sproutlm.dpo_loop --config configs/dpo_uf.yaml

Placement on Kaggle T4x2: trainable fp32 policy on cuda:0, frozen fp16 reference on
cuda:1. The two forwards run on different GPUs, and only policy activations are kept.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import runtime, sources
from .packing import _pad_id, collate_segments, segment_row
from .preference import dpo_objective
from .scoring import paired_logps


def tokenise_pairs(tokenizer, rows, max_length, max_prompt_length):
    out = []
    for r in rows:
        w = segment_row(tokenizer, r["prompt"], r["chosen"], max_length, max_prompt_length)
        l = segment_row(tokenizer, r["prompt"], r["rejected"], max_length, max_prompt_length)
        if w.response_ids and l.response_ids:
            out.append((w, l))
    return out


def collate_pairs(pairs, pad):
    goods, bads = [p[0] for p in pairs], [p[1] for p in pairs]
    width = max(s.length for s in goods + bads)
    return {"chosen": collate_segments(goods, pad, width), "rejected": collate_segments(bads, pad, width)}


def load_models(cfg):
    two_gpus = torch.cuda.device_count() >= 2
    pol_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_dev = torch.device("cuda:1") if two_gpus else pol_dev
    src = cfg["sft_checkpoint"]

    policy = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa")
    policy.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.to(pol_dev).train()

    ref_dtype = torch.float16 if ref_dev.type == "cuda" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(src, dtype=ref_dtype, attn_implementation="sdpa")
    reference.to(ref_dev).eval().requires_grad_(False)
    return policy, reference, pol_dev


def reference_scores(reference, pair):
    with torch.no_grad():
        return paired_logps(reference, pair)


def policy_scores(policy, pair, use_amp):
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        return paired_logps(policy, pair)


@torch.no_grad()
def evaluate(policy, reference, val_pairs, pad, beta, use_amp, batch_size):
    policy.eval()
    torch.cuda.empty_cache()
    losses, correct, margins, rw, rl = [], [], [], [], []
    for s in range(0, len(val_pairs), batch_size):
        pair = collate_pairs(val_pairs[s : s + batch_size], pad)
        ref_w, ref_l = reference_scores(reference, pair)
        pol_w, pol_l = policy_scores(policy, pair, use_amp)
        out = dpo_objective(pol_w, pol_l, ref_w, ref_l, beta)
        n = pol_w.size(0)
        losses.append(out.loss.item() * n)
        correct += out.correct.tolist()
        margins += out.margin.tolist()
        rw += out.reward_chosen.tolist()
        rl += out.reward_rejected.tolist()
    policy.train()
    n = max(len(correct), 1)
    return {
        "val_loss": sum(losses) / n,
        "reward_acc": sum(correct) / n,
        "reward_margin": sum(margins) / n,
        "reward_chosen": sum(rw) / n,
        "reward_rejected": sum(rl) / n,
    }


def main() -> None:
    cfg = runtime.cli_config()
    runtime.seed_everything(cfg["seed"])
    run_dir = Path(cfg["output_dir"])
    runtime.dump_config(cfg, run_dir)
    ledger = runtime.Ledger(run_dir)

    train_path, val_path = run_dir / "data" / "train.jsonl", run_dir / "data" / "val.jsonl"
    if not train_path.exists():
        train, val = sources.load_splits(cfg["dataset"], cfg["train_pairs"], cfg["val_pairs"], cfg["seed"])
        sources.freeze(train, train_path)
        sources.freeze(val, val_path)
    train_rows, val_rows = sources.thaw(train_path), sources.thaw(val_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["sft_checkpoint"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad = _pad_id(tokenizer)
    L, P = cfg["max_length"], cfg["max_prompt_length"]
    train_pairs = tokenise_pairs(tokenizer, train_rows, L, P)
    val_pairs = tokenise_pairs(tokenizer, val_rows, L, P)
    print(f"[sprout-dpo] usable pairs: train {len(train_pairs)}/{len(train_rows)}  val {len(val_pairs)}/{len(val_rows)}", flush=True)

    policy, reference, pol_dev = load_models(cfg)
    use_amp = pol_dev.type == "cuda"
    beta, per_dev = cfg["beta"], cfg["per_device_batch"]
    accum = max(1, cfg["effective_batch"] // per_dev)
    lengths = [max(w.length, l.length) for w, l in train_pairs]
    batches = runtime.bucketed_order(lengths, per_dev, cfg["seed"])
    total_steps = len(batches) // accum

    params = [p for p in policy.parameters() if p.requires_grad]
    if cfg["optimizer"] == "rmsprop":
        optim = torch.optim.RMSprop(params, lr=cfg["lr"])
    else:
        optim = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, runtime.lr_shape(cfg.get("schedule", "constant"), total_steps, cfg.get("warmup_frac", 0.0))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    budget = runtime.TimeBudget(cfg.get("time_budget_minutes"))

    def checkpoint_eval(step):
        m = evaluate(policy, reference, val_pairs, pad, beta, use_amp, cfg["eval_batch"])
        ledger.write("val", step, **m)
        return m

    print(f"[sprout-dpo] accum={accum} effective={per_dev * accum} steps={total_steps} beta={beta}", flush=True)
    checkpoint_eval(0)   # expect loss=log2, acc=0 (ties), margin=0
    best_key, step = (-1.0, float("-inf")), 0
    win = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

    for micro, idx in enumerate(batches[: total_steps * accum]):
        pair = collate_pairs([train_pairs[i] for i in idx], pad)
        ref_w, ref_l = reference_scores(reference, pair)
        pol_w, pol_l = policy_scores(policy, pair, use_amp)
        out = dpo_objective(pol_w, pol_l, ref_w, ref_l, beta)
        scaler.scale(out.loss / accum).backward()

        n = pol_w.size(0)
        win["loss"] += out.loss.item() * n
        win["acc"] += out.correct.sum().item()
        win["margin"] += out.margin.sum().item()
        win["n"] += n
        if (micro + 1) % accum:
            continue

        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]).item()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        sched.step()
        step += 1

        if step % cfg["log_every"] == 0:
            k = max(win["n"], 1)
            ledger.write("train", step, loss=win["loss"] / k, reward_acc=win["acc"] / k,
                         reward_margin=win["margin"] / k, grad_norm=grad_norm, lr=sched.get_last_lr()[0])
        win = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

        stop = budget.expired()
        if step % cfg["eval_every"] == 0 or step == total_steps or stop:
            m = checkpoint_eval(step)
            key = (m["reward_acc"], m["reward_margin"])      # accuracy first, margin breaks ties
            if key > best_key:
                best_key = key
                runtime.save_checkpoint(policy, tokenizer, run_dir / "best",
                                        {"step": step, **m, "reference": cfg["sft_checkpoint"], "beta": beta})
        if stop:
            print("[sprout-dpo] time budget reached, stopping early", flush=True)
            break

    runtime.save_checkpoint(policy, tokenizer, run_dir / "final",
                            {"step": step, "reference": cfg["sft_checkpoint"], "beta": beta})
    ledger.write("done", step, best_reward_acc=best_key[0], best_reward_margin=best_key[1])


if __name__ == "__main__":
    main()
