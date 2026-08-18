# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-chunk render cost + byte-identity, measured on the image's own stack."""

import importlib
import json
import os
import time

from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)

RID = "cmpl-3a1f9b2c4d5e6f708192a3b4c5d6e7f8"
CT = 1706000000
MODEL = "Qwen/Qwen3-0.6B"
DELTA = " the quick brown fox jumps over the"
N = 200000


def pydantic_frame(i, text):
    chunk = CompletionStreamResponse(
        id=RID,
        object="text_completion",
        created=CT,
        model=MODEL,
        choices=[
            CompletionResponseStreamChoice(
                index=i,
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


EDGE = [
    "",
    " fox jumps",
    ' "q" \\b ',
    "\n\r\t\x00\x1f\x7f",
    "  ",
    "日本語 한글 é",
    "🚀🎉",
    "\U0001d549\U0010ffff",
    "￿￾",
    "</script>",
    "a" * 400,
]


def timeit(fn, *a):
    fn(*a)
    t0 = time.perf_counter()
    for _ in range(N):
        fn(*a)
    return (time.perf_counter() - t0) / N * 1e6


print(f"{'pydantic build+dump':24s} {timeit(pydantic_frame, 0, DELTA):6.2f} us/chunk")

for mode, label in ((1, "PR template"), (2, "derived template")):
    os.environ["VLLM_SSE_FASTPATH"] = str(mode)
    import vllm.entrypoints.openai.completion.serving as sv

    importlib.reload(sv)
    pre, mid, suf = sv._fast_sse_template(RID, CT, MODEL)

    def fast(i, text, pre=pre, mid=mid, suf=suf):
        return f"{pre}{i}{mid}{json.dumps(text, ensure_ascii=False)}{suf}"

    bad = [t for t in EDGE if fast(3, t) != pydantic_frame(3, t)]
    print(
        f"{label:24s} {timeit(fast, 0, DELTA):6.2f} us/chunk   "
        f"byte-identical={not bad}{'  MISMATCH:' + repr(bad[:1]) if bad else ''}"
    )

os.environ["VLLM_SSE_FASTPATH"] = "2"
import vllm.entrypoints.openai.completion.serving as sv

importlib.reload(sv)
t0 = time.perf_counter()
for _ in range(20000):
    sv._fast_sse_template(RID, CT, MODEL)
print(f"{'derive cost':24s} {(time.perf_counter() - t0) / 20000 * 1e6:6.2f} us/REQUEST")
