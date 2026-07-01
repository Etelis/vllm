# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-domain prompt builders with strong intra-domain prefix sharing.

Each domain prepends a single, byte-identical few-shot preamble to every
question, so vLLM's prefix cache (and any KV-cache-aware router) sees a long
shared prefix within the domain while the two domains stay maximally distinct:

  * ``gsm8k`` -- grade-school math, 8-shot chain-of-thought.
  * ``mbpp``  -- Python coding, 3-shot ``[BEGIN]``/``[DONE]`` template.

Both are plain text-in / text-out (raw ``/v1/completions``), no agentic harness.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from dataclasses import dataclass, field

import requests

GSM8K_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MBPP_JSONL_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


@dataclass
class Domain:
    """A benchmark domain: shared preamble + per-question prompts."""

    name: str
    prompts: list[str]
    preamble: str
    stop: list[str]
    max_tokens: int
    num_shots: int
    meta: dict = field(default_factory=dict)


def _download_and_cache(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(filename):
        return filename
    print(f"Downloading {url} -> {filename}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(filename, "wb") as fout:
        for chunk in resp.iter_content(chunk_size=8192):
            fout.write(chunk)
    return filename


def _read_jsonl(filename: str) -> Generator[dict, None, None]:
    with open(filename) as fin:
        for line in fin:
            if line.strip() and not line.startswith("#"):
                yield json.loads(line)


def build_gsm8k_domain(
    num_questions: int,
    num_shots: int = 8,
    max_tokens: int = 256,
    cache_dir: str = "/tmp/moe_expert_routing",
) -> Domain:
    """GSM8K math domain: identical N-shot CoT preamble prepended to each test Q."""
    train = list(_read_jsonl(_download_and_cache(GSM8K_TRAIN_URL, cache_dir)))
    test = list(_read_jsonl(_download_and_cache(GSM8K_TEST_URL, cache_dir)))
    num_questions = min(num_questions, len(test))

    preamble = ""
    for i in range(num_shots):
        preamble += (
            f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n"
        )

    prompts = [
        preamble + f"Question: {test[i]['question']}\nAnswer:"
        for i in range(num_questions)
    ]
    return Domain(
        name="gsm8k",
        prompts=prompts,
        preamble=preamble,
        stop=["Question", "Assistant:", "<|separator|>"],
        max_tokens=max_tokens,
        num_shots=num_shots,
        meta={"preamble_chars": len(preamble)},
    )


def _mbpp_example(text: str, tests: list[str], code: str | None) -> str:
    block = (
        "You are an expert Python programmer, and here is your task: "
        f"{text} Your code should pass these tests:\n\n"
        + "\n".join(tests)
        + "\n[BEGIN]\n"
    )
    if code is not None:
        block += f"{code}\n[DONE]\n\n"
    return block


def _load_mbpp(cache_dir: str) -> dict[int, dict]:
    """Load MBPP indexed by task_id, preferring HF datasets, else raw JSONL."""
    try:
        from datasets import load_dataset

        rows: dict[int, dict] = {}
        for split in ("train", "validation", "test", "prompt"):
            try:
                ds = load_dataset("mbpp", "full", split=split)
            except Exception:
                continue
            for row in ds:
                rows[int(row["task_id"])] = row
        if rows:
            return rows
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"HF datasets MBPP load failed ({exc}); falling back to raw JSONL")

    path = _download_and_cache(MBPP_JSONL_URL, cache_dir)
    return {int(r["task_id"]): r for r in _read_jsonl(path)}


def build_mbpp_domain(
    num_questions: int,
    num_shots: int = 3,
    max_tokens: int = 512,
    cache_dir: str = "/tmp/moe_expert_routing",
) -> Domain:
    """MBPP code domain: fixed 3-shot [BEGIN]/[DONE] preamble prepended to each task."""
    rows = _load_mbpp(cache_dir)
    # Standard MBPP split: task_ids 1-10 reserved for few-shot, 11-510 for test.
    shot_ids = [2, 3, 4][:num_shots]
    preamble = ""
    for tid in shot_ids:
        row = rows[tid]
        preamble += _mbpp_example(row["text"], row["test_list"], row["code"])

    test_ids = [tid for tid in range(11, 511) if tid in rows][:num_questions]
    prompts = [
        preamble + _mbpp_example(rows[tid]["text"], rows[tid]["test_list"], None)
        for tid in test_ids
    ]
    return Domain(
        name="mbpp",
        prompts=prompts,
        preamble=preamble,
        stop=["[DONE]", "\n[BEGIN]"],
        max_tokens=max_tokens,
        num_shots=num_shots,
        meta={"preamble_chars": len(preamble), "shot_task_ids": shot_ids},
    )


DOMAIN_BUILDERS = {
    "gsm8k": build_gsm8k_domain,
    "mbpp": build_mbpp_domain,
}
