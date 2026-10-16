"""Task 5: LoRA-DPO persona tuning with TRL's DPOTrainer (the one place TRL is allowed).

    python -m sproutlm.persona_lora --config configs/persona_lora.yaml

DPOConfig fields move between TRL releases, so we pass only the keys the installed
version actually defines and print anything dropped, rather than pinning one exact TRL.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from . import runtime


def conversational(rows):
    """{prompt, chosen, rejected} strings -> TRL conversational records (role-message lists)."""
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ]


def build_args(cfg: dict) -> DPOConfig:
    wanted = dict(
        output_dir=str(Path(cfg["output_dir"]) / "trainer"),
        beta=cfg["beta"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["schedule"],
        warmup_ratio=cfg["warmup_frac"],
        per_device_train_batch_size=cfg["per_device_batch"],
        per_device_eval_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        max_grad_norm=1.0,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=False,
        use_cpu=bool(cfg.get("use_cpu", False)),   # local smoke tests on a Mac (avoids MPS)
    )
    known = {f.name for f in dataclasses.fields(DPOConfig)}
    dropped = sorted(set(wanted) - known)
    if dropped:
        print(f"[sprout-lora] this TRL has no {dropped}; using its defaults for those", flush=True)
    return DPOConfig(**{k: v for k, v in wanted.items() if k in known})


def main() -> None:
    cfg = runtime.cli_config()
    runtime.seed_everything(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    runtime.dump_config(cfg, out_dir)

    rows = json.loads(Path(cfg["data_path"]).read_text())
    random.Random(cfg["seed"]).shuffle(rows)
    val_rows, train_rows = rows[: cfg["val_rows"]], rows[cfg["val_rows"] :]
    print(f"[sprout-lora] train {len(train_rows)}  val {len(val_rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32)

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # with a peft_config, TRL uses the adapter-disabled base as reference
        args=build_args(cfg),
        train_dataset=Dataset.from_list(conversational(train_rows)),
        eval_dataset=Dataset.from_list(conversational(val_rows)),
        processing_class=tok,
        peft_config=lora,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[sprout-lora] trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    trainer.train()
    final_eval = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for rec in trainer.state.log_history:
            fh.write(json.dumps(rec) + "\n")
    (out_dir / "final_eval.json").write_text(json.dumps(final_eval, indent=2))
    print(f"[sprout-lora] adapter saved to {adapter_dir}\n{json.dumps(final_eval, indent=2)}")


if __name__ == "__main__":
    main()
