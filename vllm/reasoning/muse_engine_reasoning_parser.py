# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from vllm.parser.engine.registered_adapters import MuseParserReasoningAdapter


class MuseEngineReasoningParser(MuseParserReasoningAdapter):  # type: ignore[valid-type, misc]
    @property
    def opens_reasoning_in_content(self) -> bool:
        # More messages follow the answer, so without a tool parser the engine
        # keeps parsing after reasoning ends.
        return True

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self._parser_engine.is_reasoning_end_streaming(input_ids, delta_ids)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self._parser_engine.count_reasoning_tokens(token_ids)
