# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frames rendered from the per-stream template must be byte-identical to the
frames the response models would have produced, for any delta text."""

import json
import random

import pytest

from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import (
    _SSE_PROBE_TEXT,
    build_delta_frame_template,
)

CREATED = 1706000000


def template_frame(
    request_id: str, created_time: int, model_name: str, index: int, text: str
) -> str:
    prefix, infix, suffix = build_delta_frame_template(
        request_id, created_time, model_name
    )
    return f"{prefix}{index}{infix}{json.dumps(text, ensure_ascii=False)}{suffix}"


def model_frame(
    request_id: str, created_time: int, model_name: str, index: int, text: str
) -> str:
    chunk = CompletionStreamResponse(
        id=request_id,
        object="text_completion",
        created=created_time,
        model=model_name,
        choices=[
            CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=None,
                finish_reason=None,
                stop_reason=None,
                prompt_token_ids=None,
                token_ids=None,
            )
        ],
    )
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world",
        ' "quoted" and \\backslashed\\ ',
        "line\nbreaks\r\ttabs",
        "\x00\x01\x1f\x7f",
        "js separators   ",
        "unicode é 日本語 한글",
        "emoji 🚀🎉",
        "astral \U0001d549\U0010ffff",
        "nonchars ￿￾",
    ],
)
@pytest.mark.parametrize(
    ("request_id", "model_name", "index"),
    [
        ("cmpl-abc123", "meta-llama/Llama-3.1-8B", 0),
        ('id "with" quotes\\', "модель 🎯", 7),
    ],
)
def test_template_frame_matches_model_frame(
    request_id: str, model_name: str, index: int, text: str
):
    args = (request_id, CREATED, model_name, index, text)
    assert template_frame(*args) == model_frame(*args)


def test_template_frame_matches_model_frame_fuzz():
    rng = random.Random(0)
    pool = [c for c in range(1, 0x3000) if not (0xD800 <= c <= 0xDFFF)]
    pool += [0x1F600, 0x1D549, 0x10FFFF]
    for _ in range(50):
        text = "".join(chr(rng.choice(pool)) for _ in range(rng.randint(0, 40)))
        args = (
            "cmpl-" + text[:8],
            CREATED,
            "m/" + text[:4],
            rng.randint(0, 99),
            text,
        )
        assert template_frame(*args) == model_frame(*args)


def test_template_is_none_when_sentinels_are_ambiguous():
    """An id or model name carrying the index sentinel must disable the
    template rather than emit a frame split at the wrong offset."""
    assert build_delta_frame_template("cmpl-987654321", CREATED, "m") is None
    assert build_delta_frame_template("cmpl-x", CREATED, "m-987654321") is None


def test_template_handles_probe_text_in_request_fields():
    """A quote inside a field is escaped, so embedded probe text cannot forge
    the quoted marker; a field equal to the probe text duplicates it and must
    disable the template instead of splitting the frame at the wrong offset."""
    request_id = f'cmpl-"{_SSE_PROBE_TEXT}"'
    args = (request_id, CREATED, "m", 0, "hi")
    assert template_frame(*args) == model_frame(*args)
    assert build_delta_frame_template("cmpl-x", CREATED, _SSE_PROBE_TEXT) is None
