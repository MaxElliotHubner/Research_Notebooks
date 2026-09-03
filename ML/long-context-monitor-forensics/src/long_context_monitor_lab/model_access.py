from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .config import LabConfig


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    sae: Any
    hook_name: str
    sae_metadata: dict[str, Any]
    sparsity: Any


def _sae_hook_name(sae: Any) -> str | None:
    """Read the hook name across small SAE Lens configuration changes."""

    cfg = getattr(sae, "cfg", None)
    direct = getattr(cfg, "hook_name", None)
    if direct is not None:
        return str(direct)
    metadata = getattr(cfg, "metadata", None)
    nested = getattr(metadata, "hook_name", None)
    return None if nested is None else str(nested)


def load_model_and_sae(
    config: LabConfig,
    *,
    device: str | None = None,
) -> ModelBundle:
    """Load the verified Gemma 2 base model and its matched Gemma Scope SAE.

    TransformerLens 3 recommends TransformerBridge for broad model coverage,
    but the legacy HookedTransformer API remains available. It is used here
    deliberately because this SAE release is registered against the exact
    TransformerLens hook ``blocks.12.hook_resid_post`` and its established
    activation convention.
    """

    try:
        from sae_lens import SAE
        from transformer_lens import HookedTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise ImportError(
            "Install requirements.txt before loading the live model and SAE."
        ) from exc

    chosen_device = config.device if device is None else device
    dtype = getattr(torch, config.dtype)

    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=chosen_device,
        dtype=dtype,
    )

    loaded = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.sae_id,
        device=chosen_device,
    )
    if isinstance(loaded, tuple):
        sae = loaded[0]
        cfg_dict = loaded[1] if len(loaded) > 1 else {}
        sparsity = loaded[2] if len(loaded) > 2 else None
    else:
        sae = loaded
        cfg_dict = {}
        sparsity = None

    actual_hook = _sae_hook_name(sae) or config.hook_name
    if actual_hook != config.hook_name:
        raise ValueError(
            f"SAE hook {actual_hook!r} does not match configured hook "
            f"{config.hook_name!r}."
        )

    d_model = int(model.cfg.d_model)
    decoder = getattr(sae, "W_dec")
    if decoder.ndim != 2 or int(decoder.shape[-1]) != d_model:
        raise ValueError(
            "SAE decoder and model residual dimensions do not match: "
            f"W_dec={tuple(decoder.shape)}, d_model={d_model}."
        )

    return ModelBundle(
        model=model,
        tokenizer=model.tokenizer,
        sae=sae,
        hook_name=actual_hook,
        sae_metadata=dict(cfg_dict) if isinstance(cfg_dict, dict) else {},
        sparsity=sparsity,
    )


def make_hidden_state_runner(
    model: Any,
    hook_name: str,
    device: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a small callable compatible with ``extract_boundary_activations``."""

    def run(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = input_ids.to(device)
        mask = attention_mask.to(device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                input_ids,
                attention_mask=attention_mask,
                names_filter=[hook_name],
                return_type=None,
                stop_at_layer=13,
            )
        return cache[hook_name]

    return run


def run_sae(
    sae: Any,
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode/decode activations and expose decoder rows and decoder bias."""

    with torch.inference_mode():
        features = sae.encode(activations)
        reconstruction = sae.decode(features)
    decoder = sae.W_dec
    decoder_bias = sae.b_dec
    return features, reconstruction, decoder, decoder_bias
