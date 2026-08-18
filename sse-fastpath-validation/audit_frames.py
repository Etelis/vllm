import json, glob, os

base = "/Users/itayetlis/vllm/.claude/worktrees/sse-fastpath-validation/sse-fastpath-validation"
dirs = [
    ("qwen0.6b", os.path.join(base, "results")),
    ("qwen4b", os.path.join(base, "results-run2/out_qwen4b")),
    ("qwen30", os.path.join(base, "results-run2/out_qwen30")),
    ("gptoss", os.path.join(base, "results-run2/out_gptoss")),
    ("ctrl", os.path.join(base, "results-run2/out_ctrl")),
]
for name, d in dirs:
    print("=== ", name, d)
    for f in sorted(glob.glob(os.path.join(d, "*.bench.json"))):
        try:
            b = json.load(open(f))
        except Exception as e:
            print(f, "ERR", e)
            continue
        tag = os.path.basename(f).replace(".bench.json", "")
        side = f.replace(".bench.json", ".json")
        cpu = None
        if os.path.exists(side):
            try:
                s = json.load(open(side))
            except Exception:
                s = None
            if isinstance(s, dict):
                cpu = {k: v for k, v in s.items() if not isinstance(v, (list, dict))}
        itl = b.get("mean_itl_ms")
        tpot = b.get("mean_tpot_ms")
        r = itl / tpot if (itl and tpot) else float("nan")
        print(
            "%-10s thr=%9.1f tot_out=%s dur=%7.2f itl=%.4f tpot=%.4f tok/frame=%.4f frames/tok=%.4f"
            % (tag, b.get("output_throughput", 0), b.get("total_output_tokens"), b.get("duration", 0), itl, tpot, r, 1 / r)
        )
        if cpu:
            print("           side:", cpu)
