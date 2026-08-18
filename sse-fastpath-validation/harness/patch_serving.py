# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the SSE fast-path into an installed vLLM's completion serving.py.

One installed file serves every arm; VLLM_SSE_FASTPATH selects behaviour at
import time so A/B arms differ only by environment, never by bytes on disk.

  0 = stock pydantic path (fast_sse never engages)
  1 = PR Etelis/vllm#3 variant  (hardcoded field template)
  2 = derived variant           (template probed from the pydantic serializer)
"""

import shutil
import sys

TARGET = sys.argv[1]
ORIG = TARGET + ".orig"

HELPER = '''

# --- SSE fast-path A/B experiment ---------------------------------------
_FAST_SSE_MODE = int(os.environ.get("VLLM_SSE_FASTPATH", "0"))
_FAST_SSE_TEXT = "zZ_vllm_sse_sentinel_Zz"
_FAST_SSE_IDX = 987654321


def _fast_sse_template(request_id, created_time, model_name):
    """Return (prefix, mid, suffix) for a plain-delta SSE frame, or None."""
    if _FAST_SSE_MODE == 1:
        id_json = json.dumps(request_id, ensure_ascii=False)
        model_json = json.dumps(model_name, ensure_ascii=False)
        prefix = (
            f'data: {{"id":{id_json},"object":"text_completion",'
            f'"created":{created_time},"model":{model_json},'
            f'"choices":[{{"index":'
        )
        suffix = (
            ',"logprobs":null,"finish_reason":null,"stop_reason":null,'
            '"prompt_token_ids":null,"token_ids":null}]}\\n\\n'
        )
        return prefix, ',"text":', suffix
    probe = CompletionStreamResponse(
        id=request_id,
        object="text_completion",
        created=created_time,
        model=model_name,
        choices=[
            CompletionResponseStreamChoice(
                index=_FAST_SSE_IDX,
                text=_FAST_SSE_TEXT,
                logprobs=None,
                finish_reason=None,
                stop_reason=None,
                prompt_token_ids=None,
                token_ids=None,
            )
        ],
    )
    frame = f"data: {probe.model_dump_json(exclude_unset=True)}\\n\\n"
    idx_marker = str(_FAST_SSE_IDX)
    txt_marker = json.dumps(_FAST_SSE_TEXT)
    if frame.count(idx_marker) != 1 or frame.count(txt_marker) != 1:
        return None
    head, rest = frame.split(idx_marker, 1)
    mid, tail = rest.split(txt_marker, 1)
    return head, mid, tail
# ------------------------------------------------------------------------
'''

SETUP = """
        fast_sse = bool(_FAST_SSE_MODE) and (
            not request.echo
            and request.logprobs is None
            and not request.return_token_ids
            and not include_continuous_usage
        )
        if fast_sse:
            _tmpl = _fast_sse_template(request_id, created_time, model_name)
            if _tmpl is None:
                fast_sse = False
            else:
                chunk_prefix, chunk_mid, chunk_suffix = _tmpl
"""

HOT = """                    if fast_sse and finish_reason is None:
                        yield (
                            f"{chunk_prefix}{i}{chunk_mid}"
                            f"{json.dumps(delta_text, ensure_ascii=False)}"
                            f"{chunk_suffix}"
                        )
                        continue

"""

# Always patch from the pristine original so the script is idempotent.
try:
    src = open(ORIG).read()
except FileNotFoundError:
    shutil.copy2(TARGET, ORIG)
    src = open(ORIG).read()

# 1. imports
anchor = "import io\n"
assert src.count(anchor) == 1, "import anchor"
src = src.replace(anchor, "import io\nimport json\nimport os\n", 1)

# 2. helper after logger
anchor = "logger = init_logger(__name__)\n"
assert src.count(anchor) == 1, "logger anchor"
src = src.replace(anchor, anchor + HELPER, 1)

# 3. per-request setup after should_include_usage
anchor = (
    "        include_usage, include_continuous_usage = should_include_usage(\n"
    "            stream_options, self.enable_force_include_usage\n"
    "        )\n"
)
assert src.count(anchor) == 1, "should_include_usage anchor"
src = src.replace(anchor, anchor + SETUP, 1)

# 4. hot path before the per-chunk model construction (first occurrence only)
anchor = "                    chunk = CompletionStreamResponse(\n"
assert src.count(anchor) == 1, "chunk anchor"
src = src.replace(anchor, HOT + anchor, 1)

compile(src, TARGET, "exec")
open(TARGET, "w").write(src)
print("patched OK ->", TARGET)
