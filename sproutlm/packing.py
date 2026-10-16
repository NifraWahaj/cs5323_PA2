"""Turn (prompt, response) text into padded causal-LM batches.

Every row is built as two token segments, a prompt segment and a response segment,
so the response/prompt boundary is known exactly instead of being re-derived later by
searching for pad or EOS ids (Qwen's pad and EOS can share an id, which breaks that).

Rules (from the assignment contract):
  * response segment = tokens(response) + [EOS]           (EOS added *before* truncation)
  * row = prompt segment + response segment, cut to max_length from the right
    (so a long response can lose its EOS, and an over-long prompt leaves no response)
  * labels = input ids on response positions, -100 on prompt and padding
  * right padding with the tokenizer's pad id (falls back to EOS), attention 0 on pads
  * row order is never changed
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE = -100


@dataclass
class Segmented:
    """One tokenised row before padding."""

    prompt_ids: list[int]
    response_ids: list[int]

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = tokenizer.eos_token_id
    return 0 if pad is None else int(pad)


def segment_row(tokenizer, prompt: str, response: str, max_length: int, max_prompt_length: int | None = None) -> Segmented:
    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        response_ids.append(int(eos))

    # Optional training-time prompt cap. We keep the *tail* of the prompt, because the
    # assistant header and the end of the question are what the response depends on.
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    # Contract truncation: cut the joined sequence from the right.
    room_for_response = max(0, max_length - len(prompt_ids))
    prompt_ids = prompt_ids[:max_length]
    response_ids = response_ids[:room_for_response]
    return Segmented(prompt_ids, response_ids)


def collate_segments(rows: list[Segmented], pad_id: int, width: int | None = None) -> dict[str, torch.Tensor]:
    """Right-pad a list of segmented rows into input_ids / attention_mask / labels."""
    width = width or max(1, max(r.length for r in rows))
    n = len(rows)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    labels = torch.full((n, width), IGNORE, dtype=torch.long)
    for i, row in enumerate(rows):
        p, r = len(row.prompt_ids), len(row.response_ids)
        if p + r == 0:
            continue
        input_ids[i, : p + r] = torch.tensor(row.prompt_ids + row.response_ids, dtype=torch.long)
        attention_mask[i, : p + r] = 1
        if r:
            labels[i, p : p + r] = torch.tensor(row.response_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_lm_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None, width=None):
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts but {len(responses)} responses")
    rows = [segment_row(tokenizer, p, r, max_length, max_prompt_length) for p, r in zip(prompts, responses)]
    return collate_segments(rows, _pad_id(tokenizer), width)


def build_pair_batch(tokenizer, prompts, chosen, rejected, max_length, max_prompt_length=None):
    """Chosen and rejected batches padded to the *same* width.

    A shared width lets the trainer stack both halves into one forward pass
    (see scoring.paired_logps) instead of running the model twice.
    """
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen and rejected must have equal length")
    good = [segment_row(tokenizer, p, c, max_length, max_prompt_length) for p, c in zip(prompts, chosen)]
    bad = [segment_row(tokenizer, p, r, max_length, max_prompt_length) for p, r in zip(prompts, rejected)]
    width = max(1, max(row.length for row in good + bad))
    pad = _pad_id(tokenizer)
    return {"chosen": collate_segments(good, pad, width), "rejected": collate_segments(bad, pad, width)}
