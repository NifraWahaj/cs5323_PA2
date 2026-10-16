"""The DPO objective, written in terms of the two preference gaps.

    gap_policy = log pi(y_w|x)   - log pi(y_l|x)
    gap_ref    = log ref(y_w|x)  - log ref(y_l|x)
    loss       = -log sigmoid( beta * (gap_policy - gap_ref) )

`-logsigmoid(z)` is `softplus(-z)`, which stays finite for large |z|.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .scoring import paired_logps


@dataclass
class PreferenceReadout:
    loss: torch.Tensor            # scalar, batch mean (differentiable)
    reward_chosen: torch.Tensor   # (batch,) beta * (log pi - log ref) on y_w, detached
    reward_rejected: torch.Tensor

    @property
    def margin(self) -> torch.Tensor:
        return self.reward_chosen - self.reward_rejected

    @property
    def correct(self) -> torch.Tensor:
        # strict ">" on purpose: identical policy/reference gives 0 accuracy, not 0.5
        return (self.reward_chosen > self.reward_rejected).float()


def dpo_objective(pol_w, pol_l, ref_w, ref_l, beta: float) -> PreferenceReadout:
    ref_w = ref_w.to(pol_w.device, pol_w.dtype).detach()
    ref_l = ref_l.to(pol_l.device, pol_l.dtype).detach()
    gap_policy = pol_w - pol_l
    gap_ref = ref_w - ref_l
    loss = F.softplus(-beta * (gap_policy - gap_ref)).mean()
    return PreferenceReadout(
        loss=loss,
        reward_chosen=(beta * (pol_w - ref_w)).detach(),
        reward_rejected=(beta * (pol_l - ref_l)).detach(),
    )


def reference_logps(reference_model, pair) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return paired_logps(reference_model, pair)


def dpo_step(policy_model, reference_model, pair, beta: float) -> PreferenceReadout:
    ref_w, ref_l = reference_logps(reference_model, pair)
    pol_w, pol_l = paired_logps(policy_model, pair)
    return dpo_objective(pol_w, pol_l, ref_w, ref_l, beta)
