# Tokenization, Padding, Boundary Activations

# Data preparation motivated by 
# "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
# https://transformer-circuits.pub/2023/monosemantic-features
# and Anthropic's attatched notebooks

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .data import ContextExample


@dataclass(frozen=True)
class TokenizedContext:
    """One context represented as token IDs and monitored boundary positions."""

    context: ContextExample
    input_ids: torch.Tensor
    boundary_positions: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.input_ids.numel())


@dataclass(frozen=True)
class ActivationRecord:
    """Only the activations and metadata required by the scientific analysis."""

    example_id: str
    activations: torch.Tensor
    token_count: int
    label: int
    condition: str
    near_miss_count: int
    segment_kinds: tuple[str, ...]
    families: tuple[str, ...]
    pair_id: str | None = None
    placement: str | None = None


HiddenStateRunner = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def encode_segmented_context(
    tokenizer: Any,
    context: ContextExample,
    separator: str,
    add_bos: bool = True,
) -> TokenizedContext:
    """Tokenize segments separately and record each final segment token."""

    input_ids: list[int] = []
    boundary_positions: list[int] = []

    if add_bos:
        input_ids.append(tokenizer.bos_token_id)

    separator_ids = tokenizer.encode(
        separator,
        add_special_tokens=False,
    )

    for i, segment in enumerate(context.segments):

        # Put the separator between segments, but not before the first one.
        if i > 0:
            input_ids.extend(separator_ids)

        segment_ids = tokenizer.encode(
            segment.text,
            add_special_tokens=False,
        )

        input_ids.extend(segment_ids)

        # The final token of this segment is the position we monitor.
        boundary_positions.append(len(input_ids) - 1)

    return TokenizedContext(
        context=context,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        boundary_positions=torch.tensor(
            boundary_positions,
            dtype=torch.long,
        ),
    )


def pad_tokenized_contexts(
    examples: Sequence[TokenizedContext],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padding"""

    batch_size = len(examples)

    max_tokens = max(example.input_ids.numel() for example in examples)
    max_boundaries = max(
        example.boundary_positions.numel()
        for example in examples
    )

    input_ids = torch.full(
        (batch_size, max_tokens),
        pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, max_tokens),
        dtype=torch.bool,
    )

    boundary_positions = torch.zeros(
        (batch_size, max_boundaries),
        dtype=torch.long,
    )

    boundary_mask = torch.zeros(
        (batch_size, max_boundaries),
        dtype=torch.bool,
    )

    for i, example in enumerate(examples):
        n_tokens = example.input_ids.numel()
        n_boundaries = example.boundary_positions.numel()

        input_ids[i, :n_tokens] = example.input_ids
        attention_mask[i, :n_tokens] = True

        boundary_positions[i, :n_boundaries] = (
            example.boundary_positions
        )
        boundary_mask[i, :n_boundaries] = True

    return (
        input_ids,
        attention_mask,
        boundary_positions,
        boundary_mask,
    )


def gather_monitored_positions(
    hidden_states: torch.Tensor,
    boundary_positions: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> torch.Tensor:
    """Hidden states at every valid segment boundary."""

    batch_size = hidden_states.shape[0]

    batch_indices = torch.arange(
        batch_size,
        device=hidden_states.device,
    ).unsqueeze(1)

    gathered = hidden_states[
        batch_indices,
        boundary_positions,
    ]

    gathered = gathered.masked_fill(
        ~boundary_mask.unsqueeze(-1),
        0.0,
    )

    return gathered


def extract_boundary_activations(
    tokenized_contexts: Sequence[TokenizedContext],
    hidden_state_runner: HiddenStateRunner,
    batch_size: int,
    pad_token_id: int,
) -> list[ActivationRecord]:
    """C1_04: Run batches and retain one activation per segment boundary."""

    records: list[ActivationRecord] = []

    for start in range(0, len(tokenized_contexts), batch_size):

        batch = tokenized_contexts[
            start : start + batch_size
        ]

        (
            input_ids,
            attention_mask,
            boundary_positions,
            boundary_mask,
        ) = pad_tokenized_contexts(
            batch,
            pad_token_id,
        )

        hidden_states = hidden_state_runner(
            input_ids,
            attention_mask,
        )

        boundary_positions = boundary_positions.to(
            hidden_states.device
        )

        boundary_mask = boundary_mask.to(
            hidden_states.device
        )

        monitored = gather_monitored_positions(
            hidden_states,
            boundary_positions,
            boundary_mask,
        )

        for batch_index, example in enumerate(batch):

            context = example.context

            n_segments = len(context.segments)

            activations = (
                monitored[
                    batch_index,
                    :n_segments,
                ]
                .detach()
                .cpu()
                .clone()
            )

            records.append(
                ActivationRecord(
                    example_id=context.example_id,
                    activations=activations,
                    token_count=example.token_count,
                    label=context.label,
                    condition=context.condition,
                    near_miss_count=context.near_miss_count,
                    segment_kinds=tuple(
                        segment.kind
                        for segment in context.segments
                    ),
                    families=tuple(
                        segment.family
                        for segment in context.segments
                    ),
                    pair_id=context.pair_id,
                    placement=context.placement,
                )
            )

    return records