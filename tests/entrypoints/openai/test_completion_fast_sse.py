# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-equality property test for the fast SSE completion streaming path.

The fast path in vllm/entrypoints/openai/completion/serving.py renders
plain delta chunks from a per-request string template instead of
constructing Pydantic models. Frames must stay byte-identical to
``"data: " + chunk.model_dump_json(exclude_unset=True) + "\\n\\n"``.

When vllm is not importable (torch missing), inline replicas of the two
protocol models (identical field declarations and order) and of the
template builder are used so the serialization property still runs; a
full install exercises the real code.
"""

import json
import random
from typing import Any

import pytest

try:
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionResponseStreamChoice,
        CompletionStreamResponse,
    )
    from vllm.entrypoints.openai.completion.serving import _fast_sse_template
except ImportError:
    from pydantic import BaseModel, ConfigDict

    class OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    class CompletionResponseStreamChoice(OpenAIBaseModel):  # type: ignore[no-redef]
        index: int
        text: str
        logprobs: Any | None = None
        finish_reason: str | None = None
        stop_reason: int | str | None = None
        prompt_token_ids: list[int] | None = None
        token_ids: list[int] | None = None

    class CompletionStreamResponse(OpenAIBaseModel):  # type: ignore[no-redef]
        id: str
        object: str = "text_completion"
        created: int = 0
        model: str = ""
        choices: list[CompletionResponseStreamChoice] = []
        usage: Any | None = None
        system_fingerprint: str | None = None
        metrics: Any | None = None

    def _fast_sse_template(
        request_id: str, created_time: int, model_name: str
    ) -> tuple[str, str]:
        id_json = json.dumps(request_id, ensure_ascii=False)
        model_json = json.dumps(model_name, ensure_ascii=False)
        prefix = (
            f'data: {{"id":{id_json},"object":"text_completion",'
            f'"created":{created_time},"model":{model_json},'
            f'"choices":[{{"index":'
        )
        suffix = (
            ',"logprobs":null,"finish_reason":null,"stop_reason":null,'
            '"prompt_token_ids":null,"token_ids":null}]}\n\n'
        )
        return prefix, suffix


def _fast_frame(
    request_id: str, created_time: int, model_name: str, index: int, text: str
) -> str:
    prefix, suffix = _fast_sse_template(request_id, created_time, model_name)
    return f'{prefix}{index},"text":{json.dumps(text, ensure_ascii=False)}{suffix}'


def _pydantic_frame(
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


EDGE_TEXTS = [
    "",
    "Hello, world",
    ' "quoted" and \\backslashed\\ ',
    "line\nbreaks\r\ttabs",
    "\x00\x01\x1f\x7f",
    "js separators \u2028\u2029",
    "unicode é 日本語 한글",
    "emoji 🚀🎉",
    "astral \U0001d549\U0010ffff",
    "nonchars \uffff\ufffe",
]


@pytest.mark.parametrize("text", EDGE_TEXTS)
@pytest.mark.parametrize(
    ("request_id", "model_name", "index"),
    [
        ("cmpl-abc123", "meta-llama/Llama-3.1-8B", 0),
        ('id "with" quotes\\', "модель\u2028🎯", 7),
    ],
)
def test_fast_frame_matches_pydantic(
    request_id: str, model_name: str, index: int, text: str
):
    args = (request_id, 1234567890, model_name, index, text)
    assert _fast_frame(*args) == _pydantic_frame(*args)


def test_fast_frame_matches_pydantic_fuzz():
    rng = random.Random(0)
    pool = [c for c in range(1, 0x3000) if not (0xD800 <= c <= 0xDFFF)]
    pool += [0x1F600, 0x1D549, 0x10FFFF]
    for _ in range(50):
        text = "".join(chr(rng.choice(pool)) for _ in range(rng.randint(0, 40)))
        rid, model, idx = "cmpl-" + text[:8], "m/" + text[:4], rng.randint(0, 99)
        args = (rid, 1706000000, model, idx, text)
        assert _fast_frame(*args) == _pydantic_frame(*args)
