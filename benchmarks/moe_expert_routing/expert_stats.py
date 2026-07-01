# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and aggregate vLLM routed-experts output.

vLLM emits per-request expert routing when the server is launched with
``--enable-return-routed-experts``. Every OpenAI ``/v1/completions`` (or chat)
choice then carries a ``routed_experts`` field: a base64-encoded ``.npy`` byte
stream that decodes to an array of shape
``(num_tokens - 1, num_layers, top_k)`` holding the *logical* expert ids
selected for every forwarded token at every MoE layer. The ids are logical
(captured before EPLB physical remap), so they are stable regardless of expert
rearrangement.

These helpers reduce that payload to per-layer expert-usage histograms and a
handful of imbalance / divergence metrics used by the KV-routing
expert-imbalance study.
"""

from __future__ import annotations

import io

import numpy as np
import pybase64 as base64


def decode_routed_experts(b64: str | None) -> np.ndarray | None:
    """Decode a base64 ``.npy`` routed-experts payload from a completion choice.

    Args:
        b64: The ``routed_experts`` string from a completion choice, or ``None``
            when the server emitted no routing for the request.

    Returns:
        Array of shape ``(num_tokens - 1, num_layers, top_k)`` (uint8/uint16),
        or ``None`` when ``b64`` is ``None``.
    """
    if b64 is None:
        return None
    return np.load(io.BytesIO(base64.b64decode(b64)))


def layer_expert_histogram(routing: np.ndarray, num_experts: int) -> np.ndarray:
    """Count expert selections per (layer, expert) for a single request.

    Args:
        routing: Array of shape ``(T, L, K)`` of logical expert ids.
        num_experts: Total number of routed experts ``E`` in the model.

    Returns:
        Int64 array of shape ``(L, E)`` where ``[layer, expert]`` is the number
        of token-slots that routed to ``expert`` at ``layer``. Each of the ``T``
        tokens contributes ``K`` selections, so the row sum per layer is ``T*K``
        (minus any out-of-range ids, which are dropped defensively).
    """
    if routing.ndim != 3:
        raise ValueError(f"expected (T, L, K) routing, got shape {routing.shape}")
    num_tokens, num_layers, _ = routing.shape
    if num_tokens == 0:
        return np.zeros((num_layers, num_experts), dtype=np.int64)

    ids = routing.astype(np.int64)
    layer_idx = np.broadcast_to(np.arange(num_layers)[None, :, None], ids.shape)
    valid = (ids >= 0) & (ids < num_experts)
    # Fold (layer, expert) into a single linear bin so one bincount covers all
    # layers: bin = layer * E + expert.
    linear = (layer_idx * num_experts + ids)[valid]
    counts = np.bincount(linear, minlength=num_layers * num_experts)
    return counts.reshape(num_layers, num_experts)


def aggregate_histograms(hists: list[np.ndarray]) -> np.ndarray:
    """Sum a list of ``(L, E)`` per-request histograms into a benchmark total."""
    if not hists:
        raise ValueError("no histograms to aggregate")
    total = np.zeros_like(hists[0], dtype=np.int64)
    for hist in hists:
        total += hist
    return total


def usage_distribution(hist_le: np.ndarray) -> np.ndarray:
    """Normalize an ``(L, E)`` count histogram to per-layer probability rows."""
    totals = hist_le.sum(axis=1, keepdims=True)
    safe = np.where(totals == 0, 1, totals)
    return hist_le / safe


def per_layer_imbalance_ratio(hist_le: np.ndarray) -> np.ndarray:
    """Per-layer expert-load imbalance ratio ``max / mean`` (PROBE ``IR``).

    ``1.0`` means perfectly uniform expert load within the layer; larger means
    a few experts absorb disproportionate load.
    """
    load = hist_le.astype(np.float64)
    mean = load.mean(axis=1)
    safe = np.where(mean == 0, 1, mean)
    return load.max(axis=1) / safe


def per_layer_gini(hist_le: np.ndarray) -> np.ndarray:
    """Per-layer Gini coefficient of the expert-load vector (0=uniform..1=peaked)."""
    load = np.sort(hist_le.astype(np.float64), axis=1)
    num_layers, num_experts = load.shape
    index = np.arange(1, num_experts + 1)[None, :]
    total = load.sum(axis=1)
    safe = np.where(total == 0, 1, total)
    gini = (2 * (index * load).sum(axis=1)) / (num_experts * safe) - (
        num_experts + 1
    ) / num_experts
    return np.where(total == 0, 0.0, gini)


def per_layer_entropy_bits(hist_le: np.ndarray) -> np.ndarray:
    """Per-layer Shannon entropy of expert usage, in bits (0..log2(E))."""
    prob = usage_distribution(hist_le)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(prob > 0, prob * np.log2(prob), 0.0)
    return -terms.sum(axis=1)


def top_experts_per_layer(hist_le: np.ndarray, top_n: int) -> np.ndarray:
    """Return the ``top_n`` hottest expert ids per layer, shape ``(L, top_n)``."""
    top_n = min(top_n, hist_le.shape[1])
    # argsort descending, take the first top_n columns per row.
    return np.argsort(-hist_le, axis=1)[:, :top_n]


def js_divergence_per_layer(p_le: np.ndarray, q_le: np.ndarray) -> np.ndarray:
    """Per-layer Jensen-Shannon divergence (bits) between two ``(L, E)`` dists.

    Inputs may be raw counts or probabilities; each layer row is renormalized.
    JS in base 2 lies in ``[0, 1]``: ``0`` means the two domains use experts
    identically at that layer, ``1`` means disjoint expert sets.
    """
    p = usage_distribution(p_le)
    q = usage_distribution(q_le)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(a > 0, a * np.log2(a / b), 0.0)
        return terms.sum(axis=1)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def experts_to_worker_load(load_e: np.ndarray, num_workers: int) -> np.ndarray:
    """Fold a per-expert load vector into per-worker load under block EP.

    Assumes the default contiguous expert-parallel placement where worker ``r``
    owns experts ``[r*E/R, (r+1)*E/R)``. This is an illustrative mapping for the
    *logical* placement (no EPLB rearrangement); true per-worker load under a
    live router/EPLB requires the multi-worker deployment.

    Args:
        load_e: Per-expert load, shape ``(E,)``.
        num_workers: Number of expert-parallel workers ``R`` (``E % R == 0``).

    Returns:
        Per-worker load, shape ``(R,)``.
    """
    num_experts = load_e.shape[0]
    if num_experts % num_workers != 0:
        raise ValueError(
            f"num_experts ({num_experts}) not divisible by num_workers ({num_workers})"
        )
    return load_e.reshape(num_workers, num_experts // num_workers).sum(axis=1)
