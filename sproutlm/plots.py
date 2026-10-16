"""Training curves for the report.

    python -m sproutlm.plots --run outputs/sft --run outputs/dpo_hh --run outputs/dpo_uf --out outputs/plots

Raw points are drawn faintly; the solid line is an EMA (alpha stated in the legend).
The selected `best` checkpoint step is marked with a dashed vertical line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALPHA = 0.15


def ema(values, alpha=ALPHA):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def load(run: Path):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    best = None
    note = run / "best" / "sprout_note.json"
    if note.exists():
        best = json.loads(note.read_text()).get("step")
    return [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "val"], best


def series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_run(run: Path, out: Path):
    train, val, best = load(run)
    is_dpo = any("reward_acc" in r for r in val)
    panels = [("loss", "val_loss", "loss")] + ([("reward_acc", "reward_acc", "reward accuracy"),
                                               ("reward_margin", "reward_margin", "reward margin")] if is_dpo else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6), squeeze=False)
    for ax, (tk, vk, label) in zip(axes[0], panels):
        x, y = series(train, tk)
        if x:
            ax.plot(x, y, color="tab:blue", alpha=0.25, lw=0.8)
            ax.plot(x, ema(y), color="tab:blue", lw=1.6, label=f"train (EMA α={ALPHA})")
        x, y = series(val, vk)
        if x:
            ax.plot(x, y, "o-", color="tab:orange", ms=3.5, lw=1.4, label="validation")
        if best is not None:
            ax.axvline(best, ls="--", color="grey", lw=1, label=f"best ckpt (step {best})")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(run.name)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{run.name}.png", dpi=160)
    plt.close(fig)
    print(f"[sprout-plots] {out / (run.name + '.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="outputs/plots")
    args = ap.parse_args()
    for r in args.run:
        plot_run(Path(r), Path(args.out))


if __name__ == "__main__":
    main()
