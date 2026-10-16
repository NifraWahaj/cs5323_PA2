"""sproutlm: a small, hand-rolled SFT + DPO pipeline for Qwen2.5-0.5B.

Layout
------
template.py     how an instruction becomes a prompt string (one place, used everywhere)
packing.py      prompt/response text -> padded LM batches with response-only labels
scoring.py      summed response log-probs and the SFT loss
preference.py   the DPO objective and its implicit-reward diagnostics
sources.py      dataset loaders that emit plain {prompt, response} / {prompt, chosen, rejected} rows
runtime.py      seeding, devices, schedules, metric ledger, checkpoint writing
sft_loop.py     `python -m sproutlm.sft_loop --config configs/sft.yaml`
dpo_loop.py     `python -m sproutlm.dpo_loop --config configs/dpo_hh.yaml`
"""
