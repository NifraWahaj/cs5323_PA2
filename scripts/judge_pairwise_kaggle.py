"""Judge saved pairwise JSONL files with a Kaggle model-proxy LLM (kaggle_benchmarks).

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash \
        --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

Must run inside a Kaggle notebook with benchmark/model access (kaggle_benchmarks importable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sproutlm import judge  # noqa: E402


def kaggle_ask(judge_model: str):
    """Resolve the judge through kbench.llms, trying the full slug then the bare model name."""
    import kaggle_benchmarks as kbench

    llm = None
    for key in (judge_model, judge_model.split("/")[-1]):
        try:
            llm = kbench.llms[key]
            break
        except Exception:
            continue
    if llm is None:
        raise SystemExit(f"judge model {judge_model!r} not available; check kbench.llms / `kaggle b t models`")

    def ask(prompt: str) -> str:
        return str(llm.prompt(prompt, temperature=0))

    return ask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--judge_model", default="google/gemini-3.6-flash")
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--sleep_seconds", type=float, default=5.0)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--reuse_judgments", action="store_true")
    args = ap.parse_args()
    judge.run(args.pairs_path, args.output_path, kaggle_ask(args.judge_model), args.judge_model,
              args.batch_size, args.sleep_seconds, args.max_retries, args.reuse_judgments)


if __name__ == "__main__":
    main()
