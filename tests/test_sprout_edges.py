"""Edge cases the public smoke tests do not cover (truncation, ordering, exact math)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer

TOK = ToyTokenizer()


def _ids(text):
    return TOK.encode(text, add_special_tokens=False)


def test_eos_is_dropped_when_response_hits_max_length():
    prompt, response = "Q: ", "abcdef"           # 3 prompt + 6 response + EOS = 10
    batch = adapter.build_sft_batch(TOK, [prompt], [response], max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    assert TOK.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0, 3:].tolist() == _ids("abcde")


def test_eos_survives_at_exact_boundary():
    batch = adapter.build_sft_batch(TOK, ["Q: "], ["ab"], max_length=6)   # 3 + 2 + 1 = 6
    assert batch["labels"][0, -1].item() == TOK.eos_token_id
    assert batch["attention_mask"][0].tolist() == [1] * 6


def test_prompt_longer_than_max_length_leaves_no_response_and_zero_logp():
    model = ToyCausalLM(TOK.vocab_size, bias_scale=1.0)
    batch = adapter.build_sft_batch(TOK, ["abcdefghij"], ["xyz"], max_length=5)
    assert batch["labels"].eq(-100).all()
    logp = adapter.compute_response_logprobs(model, TOK, ["abcdefghij"], ["xyz"], max_length=5)
    assert logp.item() == 0.0


def test_row_order_and_padding_with_mixed_lengths():
    prompts = ["a: ", "longer prompt: ", "b: "]
    responses = ["x", "y", "a much longer answer"]
    batch = adapter.build_sft_batch(TOK, prompts, responses, max_length=64)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        n = len(_ids(p)) + len(_ids(r)) + 1
        assert batch["input_ids"][i, :n].tolist() == _ids(p) + _ids(r) + [TOK.eos_token_id]
        assert batch["attention_mask"][i].sum().item() == n
        assert batch["input_ids"][i, n:].eq(TOK.pad_token_id).all()


def test_logprobs_match_hand_computation():
    model = ToyCausalLM(TOK.vocab_size, bias_scale=0.9)
    logp_table = F.log_softmax(model.logit_bias.detach(), dim=-1)
    response = "hey"
    expected = sum(logp_table[t].item() for t in _ids(response) + [TOK.eos_token_id])
    got = adapter.compute_response_logprobs(model, TOK, ["P: "], [response], max_length=32)
    assert math.isclose(got.item(), expected, rel_tol=1e-5)


def test_logprobs_independent_of_batch_companions():
    model = ToyCausalLM(TOK.vocab_size, bias_scale=0.5)
    alone = adapter.compute_response_logprobs(model, TOK, ["P: "], ["ok"], max_length=32)
    mixed = adapter.compute_response_logprobs(
        model, TOK, ["a much longer prompt: ", "P: "], ["long response here", "ok"], max_length=32
    )
    torch.testing.assert_close(alone[0], mixed[1])


def test_dpo_loss_from_logps_matches_formula():
    pw, pl = torch.tensor([-3.0, -7.0]), torch.tensor([-5.0, -6.0])
    rw, rl = torch.tensor([-4.0, -6.5]), torch.tensor([-4.5, -6.0])
    beta = 0.1
    z = beta * ((pw - pl) - (rw - rl))
    expected = (-torch.log(torch.sigmoid(z))).mean()
    torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pw, pl, rw, rl, beta), expected)


def test_dpo_loss_is_stable_for_huge_gaps():
    big = torch.tensor([-1e4])
    loss = adapter.compute_dpo_loss_from_logps(big, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 1.0)
    assert torch.isfinite(loss) and loss.item() > 1e3


def test_dpo_pair_batches_share_width_and_differ_only_in_response():
    batch = adapter.build_dpo_batch(TOK, ["Pick: "], ["a"], ["a longer one"], max_length=32)
    c, r = batch["chosen"], batch["rejected"]
    assert c["input_ids"].shape == r["input_ids"].shape
    n = len(_ids("Pick: "))
    assert torch.equal(c["input_ids"][0, :n], r["input_ids"][0, :n])


def test_one_dpo_step_raises_chosen_over_rejected():
    policy = ToyCausalLM(TOK.vocab_size)
    reference = ToyCausalLM(TOK.vocab_size)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    args = (TOK, ["Pick: "], ["a"], ["b"])
    before = adapter.compute_response_logprobs(policy, TOK, ["Pick: ", "Pick: "], ["a", "b"], 16)
    loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=16)
    loss.backward()
    opt.step()
    after = adapter.compute_response_logprobs(policy, TOK, ["Pick: ", "Pick: "], ["a", "b"], 16)
    assert (after[0] - after[1]) > (before[0] - before[1])
    assert reference.logit_bias.grad is None
