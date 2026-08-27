"""Measured efficiency: parameters, MACs, wall-clock, and token cost.

The claim is wall-clock; parameters and MACs are inputs to it. Both are reported
together with **MACs per millisecond**, because a MAC reduction does not convert
to time one-for-one and quoting only the MAC ratio is how an efficiency claim
becomes misleading.

The axis this project actually wins on is **decode cost**, and it is structural
rather than incidental. A span head emits two indices: one forward pass, a fixed
number of sequential steps regardless of how long the value is. A character
decoder emits ``L`` characters: ``L`` sequential GRU steps per field, each with an
attention pass over the document. So the gap grows with value length, and
:func:`decode_scaling` fits the exponent from measurements instead of asserting
it.

Protocol, from the project standard: warm-up >= 8, repeats >= 25, median and IQR.
Under-warmed measurement made a small model look 5x slower than reality in a
sibling project, so the warm-up count is a parameter with a floor, not an
afterthought.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import torch

from gdx.arms import ARM_BY_NAME, gen_candidates, span_candidates
from gdx.baselines.heuristic import VARIANT_BY_NAME
from gdx.baselines.heuristic import extract_candidates as heuristic_candidates
from gdx.config import Config
from gdx.data.dataset import DocumentDataset
from gdx.llm.stub import build_prompt
from gdx.models.model import build_model
from gdx.pipelines.core import Prepared
from gdx.utils.complexity import attention_macs, count_macs, count_params
from gdx.utils.latency import benchmark_callable, fit_power_exponent
from gdx.utils.logging import get_logger
from gdx.verify import verify_document

LOG = get_logger(__name__)

N_WARMUP = 10
N_REPEATS = 30


def _batch_of(data: DocumentDataset, size: int) -> Any:
    """A fixed batch of ``size`` documents, longest-first for a stable seq_len."""
    order = sorted(range(len(data)), key=lambda i: -len(data.docs[i]))
    return data.collate(order[:size])


def head_cost(cfg: Config, prepared: Prepared) -> pd.DataFrame:
    """Params, MACs, and end-to-end latency for both heads.

    Latency covers the whole inference path a deployed system would run:
    encode, decode, and -- for the verified arms -- the bounded verification loop.
    Timing the encoder alone would flatter both arms equally and hide the cost of
    the mechanism this repository adds.
    """
    test = prepared.splits.test
    rows: list[dict[str, Any]] = []
    for head, arm_name in (("span", "span_verify"), ("generative", "generative")):
        model = build_model(replace(cfg.model, head=head), prepared.vocab.size, seed=0)
        model.eval()
        arm = ARM_BY_NAME[arm_name]
        verify_cfg = arm.verify_config(cfg.verify)
        for bs in (1, 8):
            batch = _batch_of(test, bs)
            seq = batch.seq_len

            def run(m=model, b=batch, vc=verify_cfg, h=head) -> None:
                preds = m.predict(b, top_k=5)
                for doc, pred in zip(b.docs, preds, strict=True):
                    cands = span_candidates(pred) if h == "span" else gen_candidates(pred)
                    verify_document(doc, cands, vc)

            timing = benchmark_callable(run, N_WARMUP, N_REPEATS)
            macs = count_macs(
                model.encoder,
                {"batch": batch},
                extra=attention_macs(seq, cfg.model.d_model, cfg.model.n_layers),
            )
            if head == "span":
                head_macs = count_macs(
                    model.span_head, (model.encoder(batch), batch.mask)
                )
            else:
                states = model.encoder(batch)
                head_macs = count_macs(model.gen_head, (states, batch.mask, batch.chars))
            total_macs = macs + head_macs
            rows.append(
                {
                    "arm": arm_name,
                    "head": head,
                    "batch_size": bs,
                    "seq_len": seq,
                    "params": count_params(model),
                    "encoder_params": count_params(model.encoder),
                    "head_params": count_params(model) - count_params(model.encoder),
                    "mmacs": total_macs / 1e6,
                    "encoder_mmacs": macs / 1e6,
                    "head_mmacs": head_macs / 1e6,
                    "latency_ms": timing.median_ms,
                    "iqr_ms": timing.iqr_ms,
                    "latency_per_doc_ms": timing.median_ms / bs,
                    "macs_per_ms": (total_macs / 1e6) / max(1e-9, timing.median_ms),
                    "n_warmup": timing.n_warmup,
                    "n_repeats": timing.n_repeats,
                }
            )
            LOG.info(
                "%s bs=%d seq=%d latency %.2f ms (IQR %.2f) %.2f MMAC",
                arm_name, bs, seq, timing.median_ms, timing.iqr_ms, total_macs / 1e6,
            )
    frame = pd.DataFrame(rows)
    base = frame[(frame["head"] == "generative")].set_index("batch_size")
    frame["latency_reduction_vs_generative"] = frame.apply(
        lambda r: float(base.loc[r["batch_size"], "latency_ms"]) / r["latency_ms"], axis=1
    )
    frame["params_reduction_vs_generative"] = frame.apply(
        lambda r: float(base.loc[r["batch_size"], "params"]) / r["params"], axis=1
    )
    frame["macs_reduction_vs_generative"] = frame.apply(
        lambda r: float(base.loc[r["batch_size"], "mmacs"]) / r["mmacs"], axis=1
    )
    return frame


def decode_scaling(cfg: Config, prepared: Prepared,
                   lengths: tuple[int, ...] = (4, 8, 12, 16, 20, 24)) -> pd.DataFrame:
    """Decode latency against maximum value length, for both heads.

    Fits ``t = a * L**b`` in log space. The expectation, stated before measuring:
    ``b ~ 1`` for the generative head (one sequential step per character) and
    ``b ~ 0`` for the span head (value length does not enter its decode at all).
    Whether the measurement agrees is reported either way.
    """
    test = prepared.splits.test
    rows: list[dict[str, Any]] = []
    for head in ("span", "generative"):
        times: list[float] = []
        for length in lengths:
            model_cfg = replace(cfg.model, head=head, dec_max_len=length)
            model = build_model(model_cfg, prepared.vocab.size, seed=0)
            model.eval()
            data = DocumentDataset(test.docs[:8], prepared.vocab, length)
            local = data.collate(list(range(len(data.docs))))

            def run(m=model, b=local) -> None:
                m.predict(b, top_k=5)

            timing = benchmark_callable(run, N_WARMUP, N_REPEATS)
            times.append(timing.median_ms)
            rows.append(
                {
                    "head": head,
                    "dec_max_len": length,
                    "latency_ms": timing.median_ms,
                    "iqr_ms": timing.iqr_ms,
                    "batch_size": 8,
                    "seq_len": local.seq_len,
                }
            )
        exponent = fit_power_exponent([float(x) for x in lengths], times)
        for row in rows[-len(lengths) :]:
            row["fitted_exponent"] = exponent
        LOG.info("%s decode cost exponent in value length: %.3f", head, exponent)
    return pd.DataFrame(rows)


def baseline_cost(cfg: Config, prepared: Prepared, n_docs: int = 40) -> pd.DataFrame:
    """Per-document cost of the non-neural arms: rule latency and LLM tokens.

    The LLM row is a **token count, not a latency**: the stub answers instantly
    and timing it would be meaningless. Tokens are counted by whitespace, which
    understates a real BPE tokeniser on numeric text, so the figure is a lower
    bound on what a provider would bill.
    """
    docs = prepared.splits.test.docs[:n_docs]
    variant = VARIANT_BY_NAME.get(
        cfg.baseline.heuristic_variant, VARIANT_BY_NAME["row_first"]
    )
    arm = ARM_BY_NAME["heuristic"]
    verify_cfg = arm.verify_config(cfg.verify)

    def run_heuristic() -> None:
        for doc in docs:
            verify_document(doc, heuristic_candidates(doc, variant), verify_cfg)

    timing = benchmark_callable(run_heuristic, max(2, N_WARMUP // 4), max(5, N_REPEATS // 4))
    from gdx.data.schema import FIELDS
    from gdx.llm.client import count_tokens

    prompt_tokens = 0
    for doc in docs:
        for name in FIELDS:
            prompt_tokens += count_tokens(build_prompt(doc.texts, name))
    return pd.DataFrame(
        [
            {
                "arm": "heuristic",
                "metric": "latency_per_doc_ms",
                "value": timing.median_ms / len(docs),
                "iqr_ms": timing.iqr_ms,
                "n_docs": len(docs),
                "note": f"variant {variant.name}, warmup {timing.n_warmup}",
            },
            {
                "arm": "llm_stub",
                "metric": "prompt_tokens_per_doc",
                "value": prompt_tokens / len(docs),
                "iqr_ms": np.nan,
                "n_docs": len(docs),
                "note": f"{len(FIELDS)} calls/doc, whitespace tokens, lower bound",
            },
            {
                "arm": "span_verify",
                "metric": "prompt_tokens_per_doc",
                "value": 0.0,
                "iqr_ms": np.nan,
                "n_docs": len(docs),
                "note": "no provider call; local model",
            },
        ]
    )


def torch_thread_note() -> dict[str, Any]:
    """Environment facts every timing table needs stated next to it."""
    return {
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "n_warmup": N_WARMUP,
        "n_repeats": N_REPEATS,
    }
