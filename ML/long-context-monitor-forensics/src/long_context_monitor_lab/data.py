# Controlled benchmark construction.

# Data construction motivated by 
# "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
# https://transformer-circuits.pub/2023/monosemantic-features
# and Anthropic's attatched notebooks


from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One independently labelled text segment."""

    segment_id: str
    text: str
    kind: str
    family: str
    label: int


@dataclass(frozen=True)
class ContextExample:
    """A sequence of labelled segments used for context-level monitoring."""

    example_id: str
    segments: tuple[Segment, ...]
    label: int
    condition: str
    near_miss_count: int
    positive_index: int | None = None
    pair_id: str | None = None
    placement: str | None = None


NEAR_MISS_KINDS = frozenset(
    {
        "near_miss_quote",
        "near_miss_classification",
        "near_miss_policy",
        "near_miss_translation",
    }
)


def load_template_bank(path: str | Path) -> dict[str, Any]:
    """Ouput: entire template bank as a Python dictionary. One dictionary entry could be:
    
    {
    "family": "intent_plan_access",
    "kind": "positive_intent",
    "label": 1,
    "templates": [
        "I plan to gain access to {resource}.",
        "My goal is to enter {resource} without authorization."
    ]
    }
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        bank = json.load(f)

    if "slots" not in bank or "families" not in bank:
        raise ValueError("Template bank must contain 'slots' and 'families'.")

    family_ids = []

    for family in bank["families"]:
        required_fields = {"family", "kind", "label", "templates"}

        if not required_fields.issubset(family):
            raise ValueError(
                f"Each family must contain {required_fields}."
            )

        if family["label"] not in (0, 1):
            raise ValueError("Family labels must be 0 or 1.")

        if len(family["templates"]) == 0:
            raise ValueError("Each family must contain at least one template.")

        family_ids.append(family["family"])

    if len(family_ids) != len(set(family_ids)):
        raise ValueError("Family IDs must be unique.")

    return bank
    


def instantiate_segments(
    bank: Mapping[str, Any],
    n_per_family: int,
    rng: np.random.Generator,
) -> list[Segment]:
    """Output: List of segments. One segment could for example be:

    Segment(
    segment_id="intent_plan_access_0",
    text="I plan to gain access to a restricted file repository without authorization.",
    kind="positive_intent",
    family="intent_plan_access",
    label=1,
    )
    """

    segments = []
    slots = bank["slots"]

    for family in bank["families"]:
        for i in range(n_per_family):

            # Choose one template from this family.
            template_index = rng.integers(len(family["templates"]))
            template = family["templates"][template_index]

            # Randomly choose one value for every possible slot.
            slot_values = {}

            for slot_name, candidates in slots.items():
                candidate_index = rng.integers(len(candidates))
                slot_values[slot_name] = candidates[candidate_index]

            # Fill any placeholders that occur in the chosen template.
            text = template.format(**slot_values)

            # Give every generated row a unique, reproducible ID.
            segment_id = f"{family['family']}_{i}"

            segments.append(
                Segment(
                    segment_id=segment_id,
                    text=text,
                    kind=family["kind"],
                    family=family["family"],
                    label=family["label"],
                )
            )

    return segments


def split_template_families(
    segments: Sequence[Segment],
    rng: np.random.Generator,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> dict[str, list[Segment]]:
    """Output: dictionary matching strings and segments. There are exactly three kinds:
    
    {
    "train": [...],
    "dev": [...],
    "test": [...]
    }
    
    """

    # Collect the unique family IDs belonging to each broad kind.
    families_by_kind: dict[str, set[str]] = {}

    for segment in segments:
        families_by_kind.setdefault(segment.kind, set()).add(segment.family)

    # Record which split each family belongs to.
    family_to_split: dict[str, str] = {}

    for kind, families in families_by_kind.items():
        # Sorting first makes the result reproducible across Python runs.
        family_list = np.array(sorted(families), dtype=object)
        rng.shuffle(family_list)

        n_families = len(family_list)

        if n_families < 3:
            raise ValueError(
                f"Need at least 3 families for kind {kind!r} "
                "to make nonempty train/dev/test splits."
            )

        n_train = max(1, int(train_fraction * n_families))
        n_dev = max(1, int(dev_fraction * n_families))

        # Always reserve at least one family for the test set.
        if n_train + n_dev >= n_families:
            n_dev = 1
            n_train = n_families - 2

        train_families = family_list[:n_train]
        dev_families = family_list[n_train : n_train + n_dev]
        test_families = family_list[n_train + n_dev :]

        for family in train_families:
            family_to_split[family] = "train"

        for family in dev_families:
            family_to_split[family] = "dev"

        for family in test_families:
            family_to_split[family] = "test"

    # Route the original Segment objects according to their family.
    splits = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for segment in segments:
        split_name = family_to_split[segment.family]
        splits[split_name].append(segment)

    return splits


def compose_long_contexts(
    segments: Sequence[Segment],
    near_miss_counts: Sequence[int],
    n_per_condition: int,
    total_segments: int,
    rng: np.random.Generator,
) -> list[ContextExample]:
    """C0_04: Build neutral, near-miss, and positive length-matched contexts."""

    positive_segments = [
        segment
        for segment in segments
        if segment.kind == "positive_intent"
    ]

    neutral_segments = [
        segment
        for segment in segments
        if segment.kind == "neutral"
    ]

    near_miss_segments = [
        segment
        for segment in segments
        if segment.kind.startswith("near_miss")
    ]

    contexts: list[ContextExample] = []

    def sample(
        pool: Sequence[Segment],
        n: int,
    ) -> list[Segment]:
        if n == 0:
            return []

        indices = rng.choice(
            len(pool),
            size=n,
            replace=n > len(pool),
        )

        return [
            pool[int(index)]
            for index in indices
        ]

    for near_miss_count in near_miss_counts:

        if near_miss_count > total_segments:
            raise ValueError(
                "near_miss_count cannot exceed total_segments."
            )

        if near_miss_count + 1 > total_segments:
            raise ValueError(
                "Positive contexts need room for one positive segment."
            )

        for replicate in range(n_per_condition):

            # --------------------------------------------------------
            # 1. Neutral context
            # --------------------------------------------------------

            neutral_context = sample(
                neutral_segments,
                total_segments,
            )

            rng.shuffle(neutral_context)

            contexts.append(
                ContextExample(
                    example_id=(
                        f"neutral_nm{near_miss_count}_"
                        f"{replicate}"
                    ),
                    segments=tuple(neutral_context),
                    label=0,
                    condition="neutral",
                    near_miss_count=0,
                    positive_index=None,
                )
            )

            # --------------------------------------------------------
            # 2. Near-miss context
            # --------------------------------------------------------

            near_misses = sample(
                near_miss_segments,
                near_miss_count,
            )

            neutral_fillers = sample(
                neutral_segments,
                total_segments - near_miss_count,
            )

            near_miss_context = (
                near_misses
                + neutral_fillers
            )

            rng.shuffle(near_miss_context)

            contexts.append(
                ContextExample(
                    example_id=(
                        f"near_miss_nm{near_miss_count}_"
                        f"{replicate}"
                    ),
                    segments=tuple(near_miss_context),
                    label=0,
                    condition="near_miss",
                    near_miss_count=near_miss_count,
                    positive_index=None,
                )
            )

            # --------------------------------------------------------
            # 3. Positive context
            # --------------------------------------------------------

            positive = sample(
                positive_segments,
                1,
            )

            near_misses = sample(
                near_miss_segments,
                near_miss_count,
            )

            neutral_fillers = sample(
                neutral_segments,
                total_segments - near_miss_count - 1,
            )

            positive_context = (
                positive
                + near_misses
                + neutral_fillers
            )

            rng.shuffle(positive_context)

            positive_index = next(
                i
                for i, segment in enumerate(positive_context)
                if segment.kind == "positive_intent"
            )

            contexts.append(
                ContextExample(
                    example_id=(
                        f"positive_nm{near_miss_count}_"
                        f"{replicate}"
                    ),
                    segments=tuple(positive_context),
                    label=1,
                    condition="positive",
                    near_miss_count=near_miss_count,
                    positive_index=positive_index,
                )
            )

    return contexts

def make_paired_embedding_contexts(
    target_segments: Sequence[Segment],
    context_segments: Sequence[Segment],
    n_context_segments: int,
    rng: np.random.Generator,
) -> list[ContextExample]:
    """C0_05: Place each target in isolation, early, late, and among distractors."""

    if n_context_segments < 3:
        raise ValueError(
            "n_context_segments must be at least 3 "
            "to distinguish early, late, and among placements."
        )

    contexts: list[ContextExample] = []

    n_distractors = n_context_segments - 1

    for target in target_segments:

        pair_id = f"pair_{target.segment_id}"

        # ------------------------------------------------------------
        # 1. Target in isolation
        # ------------------------------------------------------------

        contexts.append(
            ContextExample(
                example_id=f"{pair_id}_isolated",
                segments=(target,),
                label=target.label,
                condition="paired_embedding",
                near_miss_count=int(target.kind in NEAR_MISS_KINDS),
                positive_index=0 if target.label == 1 else None,
                pair_id=pair_id,
                placement="isolated",
            )
        )

        # ------------------------------------------------------------
        # 2. Draw one common set of distractors.
        #
        # We deliberately reuse the same distractors for early, late,
        # and among, so target position is the main thing that changes.
        # ------------------------------------------------------------

        distractor_indices = rng.choice(
            len(context_segments),
            size=n_distractors,
            replace=n_distractors > len(context_segments),
        )

        distractors = [
            context_segments[int(i)]
            for i in distractor_indices
        ]

        # ------------------------------------------------------------
        # 3. Insert the exact same target object at three positions.
        # ------------------------------------------------------------

        positions = {
            "early": 0,
            "among": n_context_segments // 2,
            "late": n_context_segments - 1,
        }

        for placement, position in positions.items():

            placed_segments = list(distractors)
            placed_segments.insert(position, target)

            placed_segments_tuple = tuple(placed_segments)

            near_miss_count = sum(
                segment.kind in NEAR_MISS_KINDS
                for segment in placed_segments_tuple
            )

            positive_indices = [
                i
                for i, segment in enumerate(placed_segments_tuple)
                if segment.label == 1
            ]

            positive_index = (
                positive_indices[0]
                if len(positive_indices) == 1
                else None
            )

            context_label = int(len(positive_indices) > 0)

            contexts.append(
                ContextExample(
                    example_id=f"{pair_id}_{placement}",
                    segments=placed_segments_tuple,
                    label=context_label,
                    condition="paired_embedding",
                    near_miss_count=near_miss_count,
                    positive_index=positive_index,
                    pair_id=pair_id,
                    placement=placement,
                )
            )

    return contexts


def audit_family_leakage(
    splits: Mapping[str, Sequence[Segment]],
) -> dict[str, set[str]]:
    """C0_05: Return pairwise family overlaps; every set should be empty."""

    families_by_split = {
        split_name: {segment.family for segment in segments}
        for split_name, segments in splits.items()
    }

    overlaps = {}

    for split_a, split_b in combinations(sorted(families_by_split), 2):
        overlap = families_by_split[split_a] & families_by_split[split_b]
        overlaps[f"{split_a}__{split_b}"] = overlap

    return overlaps

from itertools import product
from string import Formatter


def instantiate_unique_segments(
    bank: Mapping[str, Any],
    family_ids: set[str],
    n_per_family: int,
    rng: np.random.Generator,
) -> list[Segment]:
    """Instantiate unique texts from selected families without replacement."""

    segments: list[Segment] = []

    global_slots = bank["slots"]
    rendering_variants = bank.get(
        "rendering_variants",
        {},
    )

    formatter = Formatter()

    for family in bank["families"]:

        family_id = family["family"]

        if family_id not in family_ids:
            continue

        kind = family["kind"]

        # Kind-specific surface variations.
        variants = rendering_variants.get(
            kind,
            {},
        )

        prefixes = variants.get(
            "prefixes",
            [""],
        )

        suffixes = variants.get(
            "suffixes",
            [""],
        )

        possible_texts: set[str] = set()

        for template in family["templates"]:

            # Find placeholders such as {resource}, {language}, ...
            fields = []

            for _, field_name, _, _ in formatter.parse(template):

                if (
                    field_name is not None
                    and field_name not in fields
                ):
                    fields.append(field_name)

            # Enumerate every possible slot combination.
            if fields:

                slot_choices = [
                    global_slots[field]
                    for field in fields
                ]

                combinations = product(
                    *slot_choices
                )

            else:

                combinations = [()]

            for values in combinations:

                slot_values = dict(
                    zip(fields, values)
                )

                core_text = template.format(
                    **slot_values
                )

                for prefix in prefixes:
                    for suffix in suffixes:

                        text = (
                            prefix
                            + core_text
                            + suffix
                        ).strip()

                        possible_texts.add(text)

        possible_texts = sorted(
            possible_texts
        )

        if len(possible_texts) < n_per_family:

            raise ValueError(
                f"{family_id} has only "
                f"{len(possible_texts)} unique renderings, "
                f"but {n_per_family} were requested."
            )

        chosen_indices = rng.choice(
            len(possible_texts),
            size=n_per_family,
            replace=False,
        )

        for j, index in enumerate(chosen_indices):

            segments.append(
                Segment(
                    segment_id=(
                        f"{family_id}_unique_{j}"
                    ),
                    text=possible_texts[index],
                    kind=kind,
                    family=family_id,
                    label=int(family["label"]),
                )
            )

    return segments
