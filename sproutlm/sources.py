"""Dataset loaders. Every loader returns plain dict rows with the prompt already rendered.

SFT rows:        {"id", "prompt", "response"}
Preference rows: {"id", "prompt", "chosen", "rejected"}

All sampling is deterministic: we shuffle with a seeded `random.Random` and keep the
first N rows that survive filtering, so the exact subset is reproducible and is also
written to disk (see `freeze`) for the report.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .template import clean_response, render_prompt

HH_TURN = "\n\nHuman: "
HH_REPLY = "\n\nAssistant: "


def _take(rows, n: int, seed: int, keep) -> list[dict]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out = []
    for idx in order:
        item = keep(rows[idx], idx)
        if item is not None:
            out.append(item)
            if len(out) == n:
                break
    return out


def _usable(*texts: str, min_chars: int = 2, max_chars: int = 6000) -> bool:
    return all(t and min_chars <= len(t) <= max_chars for t in texts)


# ---------------------------------------------------------------- SFT: UltraChat
def ultrachat_sft(split: str, n: int, seed: int) -> list[dict]:
    """First user->assistant exchange of each UltraChat conversation."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    def keep(row, idx):
        msgs = row["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            return None
        q, a = msgs[0]["content"].strip(), clean_response(msgs[1]["content"])
        if not _usable(q, a):
            return None
        return {"id": f"uc-{split}-{idx}", "prompt": render_prompt(q), "response": a}

    return _take(ds, n, seed, keep)


# ---------------------------------------------------------------- DPO: Anthropic HH
def _split_hh(transcript: str):
    """'\\n\\nHuman: q\\n\\nAssistant: a' -> (q, a); None if multi-turn or malformed."""
    if transcript.count(HH_TURN) != 1 or transcript.count(HH_REPLY) != 1:
        return None
    head, answer = transcript.split(HH_REPLY)
    question = head.split(HH_TURN, 1)[1]
    return question.strip(), answer.strip()


def hh_pairs(split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split=split)

    def keep(row, idx):
        good, bad = _split_hh(row["chosen"]), _split_hh(row["rejected"])
        if good is None or bad is None or good[0] != bad[0]:
            return None
        q, yw, yl = good[0], good[1], bad[1]
        if not _usable(q, yw, yl) or yw == yl:
            return None
        return {"id": f"hh-{split}-{idx}", "prompt": render_prompt(q), "chosen": yw, "rejected": yl}

    return _take(ds, n, seed, keep)


# ---------------------------------------------------------------- DPO: UltraFeedback
def ultrafeedback_pairs(split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)

    def keep(row, idx):
        c, r = row["chosen"], row["rejected"]
        if len(c) != 2 or len(r) != 2:            # single user turn + single reply only
            return None
        q = row["prompt"].strip()
        yw, yl = clean_response(c[1]["content"]), clean_response(r[1]["content"])
        if not _usable(q, yw, yl) or yw == yl or row.get("score_chosen") == row.get("score_rejected"):
            return None
        return {"id": f"uf-{split}-{idx}", "prompt": render_prompt(q), "chosen": yw, "rejected": yl}

    return _take(ds, n, seed, keep)


# ---------------------------------------------------------------- eval: AlpacaEval subset
def alpaca_eval_subset(n: int = 300, seed: int = 42) -> list[dict]:
    """Fixed AlpacaEval subset: random.Random(seed).sample over the 805 prompts, kept in sampled order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    rows = json.loads(Path(path).read_text())
    picks = random.Random(seed).sample(range(len(rows)), n)
    return [
        {"id": f"alpaca-{seed}-{i:03d}", "source_index": j, "instruction": rows[j]["instruction"], "dataset": "alpaca_eval"}
        for i, j in enumerate(picks)
    ]


LOADERS = {
    "ultrachat": (ultrachat_sft, "train_sft", "test_sft"),
    "hh_rlhf": (hh_pairs, "train", "test"),
    "ultrafeedback": (ultrafeedback_pairs, "train_prefs", "test_prefs"),
}


def load_splits(name: str, n_train: int, n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    fn, train_split, val_split = LOADERS[name]
    return fn(train_split, n_train, seed), fn(val_split, n_val, seed + 1)


def freeze(rows: list[dict], path: str | Path) -> None:
    """Write the exact rows used, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def thaw(path: str | Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
