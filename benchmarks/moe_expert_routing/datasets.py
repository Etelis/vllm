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


SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def _squad_example(context: str, question: str, answer: str | None) -> str:
    block = f"Passage: {context}\nQuestion: {question}\nAnswer:"
    if answer is not None:
        block += f" {answer}\n\n"
    return block


def build_squad_domain(
    num_questions: int,
    num_shots: int = 4,
    max_tokens: int = 64,
    cache_dir: str = "/tmp/moe_expert_routing",
) -> Domain:
    """SQuAD reading-comprehension domain: fixed few-shot preamble per prompt."""
    path = _download_and_cache(SQUAD_DEV_URL, cache_dir)
    with open(path) as fin:
        data = json.load(fin)["data"]
    triples: list[tuple[str, str, str]] = []
    for article in data:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                if qa["answers"]:
                    triples.append(
                        (para["context"], qa["question"], qa["answers"][0]["text"])
                    )
    preamble = (
        "Answer each question using only the given passage. "
        "Reply with the shortest exact answer span.\n\n"
    )
    for context, question, answer in triples[:num_shots]:
        preamble += _squad_example(context, question, answer)
    pool = triples[num_shots : num_shots + num_questions]
    prompts = [preamble + _squad_example(c, q, None) for c, q, _ in pool]
    return Domain(
        name="squad",
        prompts=prompts,
        preamble=preamble,
        stop=["\n"],
        max_tokens=max_tokens,
        num_shots=num_shots,
        meta={"preamble_chars": len(preamble), "num_pool": len(pool)},
    )


WRITING_SUBJECTS = [
    "a lighthouse keeper",
    "a retired astronaut",
    "a street violinist",
    "an apprentice mapmaker",
    "a night-shift baker",
    "a deep-sea diver",
    "a clockmaker's daughter",
    "a wandering translator",
    "a beekeeper",
    "a tram conductor",
    "an archivist of lost letters",
    "a glassblower",
    "a mountain guide",
    "a radio operator",
    "a museum night guard",
    "a typewriter repairman",
    "a mushroom forager",
    "a bridge painter",
    "a subway busker",
    "a weather-station observer",
]
WRITING_QUIRKS = [
    "collects other people's shopping lists",
    "speaks to machines politely",
    "remembers every sunset in color names",
    "never walks the same route twice",
    "keeps a diary written backwards",
    "hums extinct birdsongs",
    "measures time in cups of tea",
    "names every stray cat after a philosopher",
    "folds origami from old maps",
    "counts stairs in prime numbers",
    "writes postcards to their future self",
    "repairs things nobody asked about",
    "grows herbs on a fire escape",
    "photographs only shadows",
    "trades stories instead of money",
]
WRITING_SETTINGS = [
    "a city built on canals",
    "the last train of the year",
    "an island with two lighthouses",
    "a library that never closes",
    "a rooftop greenhouse in winter",
    "a border town between time zones",
    "a harbor locked in fog",
    "a village above the clouds",
    "an abandoned funicular station",
    "a floating market at dawn",
]


def build_writing_domain(
    num_questions: int,
    num_shots: int = 0,
    max_tokens: int = 256,
    cache_dir: str = "/tmp/moe_expert_routing",
) -> Domain:
    """Creative-writing domain: shared style-guide preamble, templated prompts."""
    preamble = (
        "You are a fiction writer for a literary magazine. House style: "
        "write vivid, concrete prose in third person past tense; ground every "
        "scene in sensory detail; prefer short declarative sentences; avoid "
        "cliches, adverbs, and abstract summary; every story must contain one "
        "specific object described twice, once early and once transformed at "
        "the end; end on an image, not a moral. Stories are 150-200 words.\n\n"
        "Example story:\nThe kettle had a dent shaped like a comma. Marta "
        "filled it anyway, watching steam climb the kitchen window. Outside, "
        "the ferry horn counted the morning into pieces. She wrote her "
        "brother's address on the back of a receipt and weighed it down with "
        "a spoon. The tide would be out by noon. She wanted to be on it, or "
        "in the letter, or anywhere the kettle's whistle could not follow. "
        "When it sang she poured two cups from habit and drank hers standing. "
        "The second cup cooled beside the sink, a small gray lake. She "
        "rinsed the kettle, packed it last, and carried the box downstairs. "
        "On the curb the comma-shaped dent caught the sun, a bright pause in "
        "the middle of the street's long sentence.\n\n"
    )
    combos = [
        (s, q, w)
        for s in WRITING_SUBJECTS
        for q in WRITING_QUIRKS
        for w in WRITING_SETTINGS
    ]
    prompts = [
        preamble + f"Write a story about {s} who {q}, set in {w}.\nStory:"
        for s, q, w in combos[:num_questions]
    ]
    return Domain(
        name="writing",
        prompts=prompts,
        preamble=preamble,
        stop=["\n\n\n"],
        max_tokens=max_tokens,
        num_shots=num_shots,
        meta={"preamble_chars": len(preamble), "num_pool": len(prompts)},
    )


DOMAIN_BUILDERS = {
    "gsm8k": build_gsm8k_domain,
    "mbpp": build_mbpp_domain,
    "squad": build_squad_domain,
    "writing": build_writing_domain,
}
