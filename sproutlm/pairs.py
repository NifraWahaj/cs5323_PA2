"""Build blind pairwise files: candidate vs baseline, A/B order randomised per prompt.

    python -m sproutlm.pairs --candidate gen/alpaca__dpo_uf.json --baseline gen/alpaca__sft.json \
        --rubric general --out eval/dpo_uf_vs_sft_pairs.jsonl

Judge-facing fields: pair_id, instruction, response_a, response_b, rubric.
Hidden fields (never sent to the judge): `hidden.a_is`, `hidden.b_is`, category.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def coin(pair_id: str, seed: int) -> bool:
    """Deterministic per-pair coin flip (stable even if the file is rebuilt in another order)."""
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()[0] % 2 == 1


def build(candidate: list[dict], baseline: list[dict], rubric: str, seed: int) -> list[dict]:
    base_by_id = {r["id"]: r for r in baseline}
    pairs = []
    for cand in candidate:
        base = base_by_id.get(cand["id"])
        if base is None:
            continue
        assert base["instruction"] == cand["instruction"], f"prompt mismatch on {cand['id']}"
        swap = coin(cand["id"], seed)
        a, b = (base, cand) if swap else (cand, base)
        pairs.append({
            "pair_id": cand["id"],
            "instruction": cand["instruction"],
            "response_a": a["output"],
            "response_b": b["output"],
            "rubric": rubric,
            "hidden": {"a_is": a["generator"], "b_is": b["generator"],
                       "candidate": cand["generator"], "baseline": base["generator"],
                       "category": cand.get("category")},
        })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--rubric", choices=["general", "safety", "persona_style", "persona_quality"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=532)
    args = ap.parse_args()

    pairs = build(json.loads(Path(args.candidate).read_text()), json.loads(Path(args.baseline).read_text()),
                  args.rubric, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    a_first = sum(p["hidden"]["a_is"] == p["hidden"]["candidate"] for p in pairs)
    print(f"[sprout-pairs] {out}: {len(pairs)} pairs, candidate shown as A in {a_first}")


if __name__ == "__main__":
    main()
