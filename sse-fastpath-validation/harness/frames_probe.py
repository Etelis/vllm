# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count SSE frames actually emitted per generated token, under load.

The CPU-per-token metric only equals CPU-per-frame if the server emits one
frame per token. vLLM can merge undelivered deltas, and a slower frontend
would merge more -- which would make the stock arm look artificially cheap
per token. This measures frames/token directly so the two arms can be
compared on the same basis.

Usage: frames_probe.py <model> <concurrency> <requests> <max_tokens>
"""

import asyncio
import json
import sys

import aiohttp

MODEL, CONC, NREQ, MAXTOK = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
URL = "http://127.0.0.1:8000/v1/completions"
PROMPT = "Write a long detailed story about a lighthouse keeper and the sea. "


async def one(session, sem, idx):
    body = {
        "model": MODEL,
        "prompt": f"{PROMPT}(variant {idx}) ",
        "max_tokens": MAXTOK,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    frames = 0
    completion_tokens = None
    async with sem:
        async with session.post(URL, json=body) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                frames += 1
                obj = json.loads(payload)
                if obj.get("usage"):
                    completion_tokens = obj["usage"]["completion_tokens"]
    return frames, completion_tokens


async def main():
    sem = asyncio.Semaphore(CONC)
    timeout = aiohttp.ClientTimeout(total=1800)
    conn = aiohttp.TCPConnector(limit=CONC + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        results = await asyncio.gather(*(one(s, sem, i) for i in range(NREQ)))
    # the final usage-only chunk is a frame that carries no token
    total_frames = sum(f for f, _ in results)
    total_delta_frames = total_frames - len(results)
    total_tokens = sum(t or 0 for _, t in results)
    print(
        json.dumps(
            {
                "requests": len(results),
                "total_frames": total_frames,
                "delta_frames": total_delta_frames,
                "total_tokens": total_tokens,
                "frames_per_token": round(total_delta_frames / total_tokens, 5),
                "min_frames_req": min(f for f, _ in results),
                "max_frames_req": max(f for f, _ in results),
            },
            indent=2,
        )
    )


asyncio.run(main())
