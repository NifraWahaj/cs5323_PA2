"""Batched greedy generation over a fixed evaluation set.

    python -m sproutlm.generate --model PATH --eval_set alpaca --generator sft --out outputs/gen/alpaca__sft.json
    python -m sproutlm.generate ... --limit 3          # sanity check on 3 prompts, printed to stdout

Qwen checkpoints (base / SFT / DPO) use the Sprout template from template.py.
SmolLM2 (persona track) uses its own chat template; pass --adapter for the LoRA weights.
The output file is written once all prompts finish; an existing file is skipped (resume).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .template import clean_response, render_prompt, stop_token_ids

REPO = Path(__file__).resolve().parent.parent
EVAL_FILES = {
    "alpaca": REPO / "eval_sets" / "alpaca_eval_300_seed42.json",
    "hh": REPO / "eval_sets" / "hh_rlhf_eval_300_seed42.json",
    "persona": REPO / "eval_sets" / "persona_style_prompts.json",
}
DATASET_TAG = {"alpaca": "alpaca_eval", "hh": "hh_rlhf_style_eval", "persona": "persona_style_eval"}

# One decoding policy for every model in a comparison (documented in the report).
DECODE = dict(do_sample=False, max_new_tokens=384, repetition_penalty=1.1, no_repeat_ngram_size=0)


def ensure_alpaca_subset() -> Path:
    path = EVAL_FILES["alpaca"]
    if not path.exists():
        from .sources import alpaca_eval_subset

        path.write_text(json.dumps(alpaca_eval_subset(300, 42), indent=1, ensure_ascii=False))
    return path


def load_prompts(eval_set: str) -> list[dict]:
    if eval_set == "alpaca":
        ensure_alpaca_subset()
    return json.loads(EVAL_FILES[eval_set].read_text())


def load_generator(model_path: str, adapter: str | None, device: str):
    tok = AutoTokenizer.from_pretrained(adapter or model_path)
    tok.padding_side = "left"                       # decoder-only batching needs left padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if "cuda" in device else torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.to(device).eval()


def format_prompt(tok, instruction: str, style: str) -> str:
    if style == "sprout":
        return render_prompt(instruction)
    return tok.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_all(tok, model, prompts: list[str], batch_size: int, device: str) -> list[str]:
    stops = stop_token_ids(tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # similar lengths per batch
    outs = [""] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, **DECODE, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, row in zip(idx, new):
            outs[i] = clean_response(tok.decode(row, skip_special_tokens=False).replace(tok.pad_token or "", ""))
        print(f"  {min(s + batch_size, len(order))}/{len(order)}", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval_set", choices=list(EVAL_FILES), required=True)
    ap.add_argument("--generator", required=True, help="name stored in the output, e.g. sft, dpo_uf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--style", choices=["sprout", "chat_template"], default="sprout")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.out and args.limit is None and Path(args.out).exists():
        print(f"[sprout-gen] {args.out} exists, skipping")
        return
    rows = load_prompts(args.eval_set)[: args.limit]
    tok, model = load_generator(args.model, args.adapter, args.device)
    t0 = time.time()
    outputs = generate_all(tok, model, [format_prompt(tok, r["instruction"], args.style) for r in rows], args.batch_size, args.device)

    records = [
        {
            "id": r["id"],
            "instruction": r["instruction"],
            "output": o,
            "generator": args.generator,
            "dataset": DATASET_TAG[args.eval_set],
            **({"category": r["category"]} if "category" in r else {}),
        }
        for r, o in zip(rows, outputs)
    ]
    if args.limit is not None:
        for rec in records:
            print("=" * 80, f"\n[{rec['id']}] {rec['instruction']}\n---\n{rec['output']}")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    meta = {"model": args.model, "adapter": args.adapter, "style": args.style, "decode": DECODE, "n": len(records),
            "minutes": round((time.time() - t0) / 60, 2)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[sprout-gen] wrote {out} ({len(records)} rows, {meta['minutes']} min)")


if __name__ == "__main__":
    main()
