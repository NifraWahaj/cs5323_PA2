"""Response-only log-probabilities and the SFT loss.

Token log-probs come from `cross_entropy(..., reduction="none")` on the shifted
logits rather than a full `log_softmax` + `gather`: same numbers, but it never keeps a
second (batch, seq, vocab) tensor around, which matters with Qwen's 151k vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .packing import IGNORE


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def token_logps(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-token log-probs, response mask), both shaped (batch, seq-1).

    Position t scores token t+1 given tokens <= t. Masked positions hold 0.
    """
    device = _model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["labels"].to(device)[:, 1:]

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    keep = targets.ne(IGNORE)
    nll = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE,
        reduction="none",
    ).view_as(targets)
    return (-nll) * keep, keep


def sequence_logps(model, batch) -> torch.Tensor:
    """One summed response log-probability per row (0.0 for rows with no response left)."""
    per_token, _ = token_logps(model, batch)
    return per_token.sum(dim=-1)


def sft_loss(model, batch) -> tuple[torch.Tensor, int]:
    """Mean NLL over every unmasked response token in the batch (token-level reduction).

    Returns (loss, number of response tokens). Token-level weighting is used so long
    answers are not down-weighted relative to one-line answers.
    """
    per_token, keep = token_logps(model, batch)
    count = int(keep.sum().item())
    return -per_token.sum() / max(count, 1), count


def paired_logps(model, pair: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Score chosen and rejected halves in a single stacked forward pass."""
    stacked = {k: torch.cat([pair["chosen"][k], pair["rejected"][k]], dim=0) for k in pair["chosen"]}
    scores = sequence_logps(model, stacked)
    n = pair["chosen"]["input_ids"].size(0)
    return scores[:n], scores[n:]
