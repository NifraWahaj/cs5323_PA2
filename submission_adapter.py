"""Stable grading interface for the SFT and DPO implementation.

Each function is a thin wrapper over the `sproutlm` package. Prompts arrive here
already formatted (the chat template is applied in sproutlm.sources), so nothing in
this file adds or strips template text.
"""

from __future__ import annotations

from sproutlm import packing, preference, scoring


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return packing.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = packing.build_lm_batch(tokenizer, prompts, responses, max_length)
    return scoring.sequence_logps(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss (mean over all unmasked response tokens)."""

    batch = packing.build_lm_batch(tokenizer, prompts, responses, max_length)
    loss, _ = scoring.sft_loss(model, batch)
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return packing.build_pair_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return preference.dpo_objective(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    ).loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    pair = packing.build_pair_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return preference.dpo_step(policy_model, reference_model, pair, beta).loss
