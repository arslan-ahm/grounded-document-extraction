"""Parameter and multiply-accumulate accounting.

MACs are *inputs* to an efficiency argument, never the claim itself -- see
:mod:`gdx.utils.latency` and the MACs-per-ms column in ``docs/RESULTS.md``, which
exists because a 10x MAC reduction did not buy 10x wall-clock anywhere in this
project.

The counter is a forward-hook accumulator over the module types this repository
actually uses (``Linear``, ``Embedding``, ``GRUCell``, ``LayerNorm``) plus an
explicit analytic term for attention, which has no parameters proportional to
sequence length and is therefore invisible to a parameter-based estimate. That
attention term is the dominant cost at long sequences, so omitting it -- the
usual shortcut -- would understate the encoder by a large factor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count.

    Args:
        module: Any module.
        trainable_only: Count only parameters with ``requires_grad``.
    """
    return int(
        sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
    )


def _linear_macs(mod: nn.Linear, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> int:
    del out
    x = inp[0]
    return int(x.numel() // x.shape[-1]) * mod.in_features * mod.out_features


def _embedding_macs(mod: nn.Embedding, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> int:
    del mod, inp, out
    return 0  # a table lookup is memory traffic, not arithmetic


def _layernorm_macs(mod: nn.LayerNorm, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> int:
    del mod, out
    return int(inp[0].numel()) * 2


def _grucell_macs(mod: nn.GRUCell, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> int:
    del out
    batch = int(inp[0].shape[0])
    return batch * 3 * mod.hidden_size * (mod.input_size + mod.hidden_size)


_RULES: dict[type, Callable[..., int]] = {
    nn.Linear: _linear_macs,
    nn.Embedding: _embedding_macs,
    nn.LayerNorm: _layernorm_macs,
    nn.GRUCell: _grucell_macs,
}


def attention_macs(seq_len: int, d_model: int, n_layers: int, n_heads: int = 1) -> int:
    """Analytic MAC count for the sequence-length-quadratic part of attention.

    ``QK^T`` and ``(attn @ V)`` each cost ``seq_len^2 * d_model`` multiply-adds
    per layer, summed over heads (the per-head dimension is ``d_model /
    n_heads``, so the head count cancels).

    Args:
        seq_len: Tokens per document.
        d_model: Model width.
        n_layers: Encoder layers.
        n_heads: Present for clarity; does not change the total.

    Returns:
        MACs, excluding the projections, which the ``Linear`` hooks cover.
    """
    del n_heads
    return int(2 * n_layers * seq_len * seq_len * d_model)


def count_macs(
    module: nn.Module,
    example_inputs: Sequence[Any] | dict[str, Any],
    extra: int = 0,
) -> int:
    """Count MACs for one forward pass over ``example_inputs``.

    Args:
        module: The module to profile. Put in ``eval`` mode and restored after.
        example_inputs: Positional args (sequence) or keyword args (dict).
        extra: Analytic terms the hooks cannot see, e.g.
            :func:`attention_macs`.

    Returns:
        Total MACs including ``extra``.
    """
    total = 0
    handles = []

    def make_hook(rule: Callable[..., int]):
        def hook(mod, inp, out):  # noqa: ANN001
            nonlocal total
            total += int(rule(mod, inp, out))

        return hook

    for sub in module.modules():
        rule = _RULES.get(type(sub))
        if rule is not None:
            handles.append(sub.register_forward_hook(make_hook(rule)))

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                module(**example_inputs)
            else:
                module(*example_inputs)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return int(total + extra)
