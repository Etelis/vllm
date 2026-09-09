import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    started = time.monotonic()
    urls = [f"http://127.0.0.1:{p}" for p in (8100, 8110, 8120)]
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() - started < 1800:
            states = []
            for url in urls:
                try:
                    response = await client.get(url + "/v1/pd_role")
                    response.raise_for_status()
                    states.append(response.json())
                except (httpx.HTTPError, ValueError) as exc:
                    states.append({"error": repr(exc)})
            print(json.dumps({"elapsed": time.monotonic() - started, "states": states}), flush=True)
            if all(s.get("phase") == "ready" for s in states):
                assert [s["role"] for s in states] == ["prefill", "prefill", "decode"], states
                assert [s["epoch"] for s in states] == [0, 0, 0], states
                Path(args.run_dir, "ready.json").write_text(json.dumps(states, indent=2))
                return
            await asyncio.sleep(10)
    raise TimeoutError("Fleet did not become ready in 30 minutes")


asyncio.run(main())
