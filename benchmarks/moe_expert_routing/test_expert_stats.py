# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local (no-GPU) checks for the routed-experts decode/aggregation helpers.

Fabricates a server-shaped base64 ``.npy`` payload and verifies the decode and
every metric against hand-computed expectations. Run directly:

    .venv/bin/python benchmarks/moe_expert_routing/test_expert_stats.py

or under pytest.
"""

from __future__ import annotations

import io

import numpy as np
import pybase64 as base64

from benchmarks.moe_expert_routing.expert_stats import (
    aggregate_histograms,
    decode_routed_experts,
    experts_to_worker_load,
    js_divergence_per_layer,
    layer_expert_histogram,
    per_layer_entropy_bits,
    per_layer_imbalance_ratio,
    top_experts_per_layer,
    usage_distribution,
)


def _encode(arr: np.ndarray) -> str:
    """Mirror the server side: np.save -> base64, exactly as vLLM does."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_roundtrip() -> None:
    arr = np.array(
        [[[0, 1], [2, 3]], [[1, 1], [3, 3]]], dtype=np.uint8
    )  # (T=2,L=2,K=2)
    decoded = decode_routed_experts(_encode(arr))
    assert decoded is not None
    assert decoded.shape == (2, 2, 2)
    assert np.array_equal(decoded, arr)
    assert decode_routed_experts(None) is None


def test_layer_expert_histogram() -> None:
    # T=2 tokens, L=2 layers, K=2 top-k, E=4 experts.
    # layer 0 picks: token0 -> {0,1}, token1 -> {1,1}  => expert1 x3, expert0 x1
    # layer 1 picks: token0 -> {2,3}, token1 -> {3,3}  => expert3 x3, expert2 x1
    routing = np.array([[[0, 1], [2, 3]], [[1, 1], [3, 3]]], dtype=np.uint8)
    hist = layer_expert_histogram(routing, num_experts=4)
    assert hist.shape == (2, 4)
    assert np.array_equal(hist[0], [1, 3, 0, 0])
    assert np.array_equal(hist[1], [0, 0, 1, 3])
    # Row sum equals T*K.
    assert hist.sum() == routing.size


def test_empty_and_out_of_range() -> None:
    empty = np.zeros((0, 3, 2), dtype=np.uint8)
    assert layer_expert_histogram(empty, num_experts=8).shape == (3, 8)
    # An out-of-range id (99) must be dropped, not corrupt neighboring bins.
    routing = np.array([[[0, 99]]], dtype=np.int64)  # T=1,L=1,K=2
    hist = layer_expert_histogram(routing, num_experts=8)
    assert hist.sum() == 1 and hist[0, 0] == 1


def test_aggregate_and_distribution() -> None:
    h1 = np.array([[2, 0], [0, 2]], dtype=np.int64)
    h2 = np.array([[0, 2], [2, 0]], dtype=np.int64)
    agg = aggregate_histograms([h1, h2])
    assert np.array_equal(agg, [[2, 2], [2, 2]])
    dist = usage_distribution(agg)
    assert np.allclose(dist, 0.5)


def test_imbalance_and_entropy_bounds() -> None:
    uniform = np.ones((1, 8), dtype=np.int64)
    peaked = np.array([[8, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    assert np.isclose(per_layer_imbalance_ratio(uniform)[0], 1.0)
    assert np.isclose(per_layer_imbalance_ratio(peaked)[0], 8.0)  # max/mean = 8/1
    # Uniform over 8 experts -> log2(8) = 3 bits; fully peaked -> 0 bits.
    assert np.isclose(per_layer_entropy_bits(uniform)[0], 3.0)
    assert np.isclose(per_layer_entropy_bits(peaked)[0], 0.0)


def test_js_divergence_extremes() -> None:
    p = np.array([[4, 4, 0, 0]], dtype=np.int64)
    same = js_divergence_per_layer(p, p)
    assert np.isclose(same[0], 0.0)  # identical usage -> 0
    q = np.array([[0, 0, 4, 4]], dtype=np.int64)
    disjoint = js_divergence_per_layer(p, q)
    assert np.isclose(disjoint[0], 1.0)  # disjoint expert sets -> 1 bit


def test_top_experts_and_worker_fold() -> None:
    hist = np.array([[1, 5, 2, 9]], dtype=np.int64)
    top = top_experts_per_layer(hist, top_n=2)
    assert np.array_equal(top[0], [3, 1])  # expert 3 (9) then expert 1 (5)
    # 8 experts over 2 workers: worker0 owns [0..3], worker1 owns [4..7].
    load = np.array([1, 1, 1, 1, 10, 10, 10, 10], dtype=np.int64)
    worker = experts_to_worker_load(load, num_workers=2)
    assert np.array_equal(worker, [4, 40])


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nAll {len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
