"""Small explicit configuration for the research lab."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class LabConfig:
    """Configuration values that should remain visible in the main notebook."""

    mode: str = "pilot"
    seed: int = 1729
    model_name: str = "gemma-2-2b"
    hf_model_id: str = "google/gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_12/width_16k/canonical"
    hook_name: str = "blocks.12.hook_resid_post"
    layer: int = 12
    separator: str = "\n\n[END OF SEGMENT]\n\n"
    device: str = "cuda"
    dtype: str = "bfloat16"
    activation_batch_size: int = 2
    probe_steps: int = 500
    probe_learning_rate: float = 0.03
    probe_l2: float = 1e-4
    target_tpr: float = 0.95
    near_miss_counts: tuple[int, ...] = tuple(range(0, 33, 2))
    total_segments: int = 33
    pilot_examples_per_family: int = 5
    core_examples_per_family: int = 4
    pilot_contexts_per_condition: int = 20
    core_contexts_per_condition: int = 12
    correction_top_k: int = 8
    bootstrap_repetitions: int = 1_000
    max_tokens: int = 4_096
    data_path: Path = Path("data/template_bank.json")
    cache_dir: Path = Path("cache")
    figures_dir: Path = Path("figures")
    results_dir: Path = Path("results")
    tail_examples_per_benign_family: int = 42

    @property
    def examples_per_family(self) -> int:
        return (
            self.pilot_examples_per_family
            if self.mode == "pilot"
            else self.core_examples_per_family
        )

    @property
    def contexts_per_condition(self) -> int:
        return (
            self.pilot_contexts_per_condition
            if self.mode == "pilot"
            else self.core_contexts_per_condition
        )


def make_config(mode: str = "pilot", **overrides: object) -> LabConfig:
    """Return the visible pilot or core configuration with optional overrides."""

    if mode not in {"pilot", "core"}:
        raise ValueError("mode must be 'pilot' or 'core'")
    config = LabConfig(mode=mode)
    return replace(config, **overrides)
