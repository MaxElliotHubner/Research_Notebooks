# linear monitor, pooling rules

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class LinearProbe(nn.Module):
    """single affine logit fo one residual-stream activation."""

    def __init__(self, activation_dim: int) -> None:
        super().__init__()

        self.linear = nn.Linear(activation_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        logits = self.linear(activations)

        return logits.squeeze(-1)


def fit_linear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[LinearProbe, list[float]]:
    """fit full-batch probe with Lebesgue 2 penalty."""

    activation_dim = activations.shape[-1]

    probe = LinearProbe(activation_dim).to(
        device=activations.device,
        dtype=activations.dtype,
    )

    labels_float = labels.to(
        device=activations.device,
        dtype=activations.dtype,
    )

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=learning_rate,
    )

    history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad()

        logits = probe(activations)

        classification_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels_float,
        )

        l2_penalty = probe.linear.weight.square().sum()

        loss = (
            classification_loss
            + l2_coefficient * l2_penalty
        )

        loss.backward()
        optimizer.step()

        history.append(float(loss.detach().cpu()))

    return probe, history


def mean_difference_direction(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    positive_mean = activations[labels == 1].mean(dim=0)
    negative_mean = activations[labels == 0].mean(dim=0)

    direction = positive_mean - negative_mean
    direction = direction / direction.norm()

    midpoint = 0.5 * (positive_mean + negative_mean)
    bias = -torch.dot(direction, midpoint)

    return direction, bias

def segment_logits(
    activations: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | float,
) -> torch.Tensor:

    return activations @ weight + bias


def pool_segment_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    method: Literal["max", "final", "topk_mean"],
    *,
    top_k: int = 3,
) -> torch.Tensor:
    """C2_04: Pool valid segment logits into one context-level score."""

    if method == "max":
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        return masked_logits.max(dim=-1).values

    if method == "final":
        n_valid = mask.sum(dim=-1)
        final_indices = n_valid - 1

        return logits.gather(
            dim=-1,
            index=final_indices.unsqueeze(-1),
        ).squeeze(-1)

    if method == "topk_mean":
        pooled_scores = []

        for row_logits, row_mask in zip(logits, mask):
            valid_logits = row_logits[row_mask]

            k = min(top_k, valid_logits.numel())

            top_values = torch.topk(
                valid_logits,
                k=k,
            ).values

            pooled_scores.append(top_values.mean())

        return torch.stack(pooled_scores)

    raise ValueError(f"Unknown pooling method: {method}")


def threshold_at_target_tpr(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_tpr: float,
) -> float:
    """Highest observed positive threshold retaining target TPR."""

    positive_scores = scores[labels == 1]

    sorted_scores = torch.sort(
        positive_scores,
        descending=True,
    ).values

    n_positive = sorted_scores.numel()

    n_required = int(
        torch.ceil(
            torch.tensor(target_tpr * n_positive)
        ).item()
    )

    threshold = sorted_scores[n_required - 1]

    return float(threshold)
