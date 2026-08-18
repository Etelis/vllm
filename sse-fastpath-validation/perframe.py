import json, os, statistics as st

base = "/Users/itayetlis/vllm/.claude/worktrees/sse-fastpath-validation/sse-fastpath-validation"
sets = {
    "0.6B A": (os.path.join(base, "results"), ["a1_m0", "a2_m0"], ["a1_m2", "a2_m2"]),
    "0.6B A(hc)": (os.path.join(base, "results"), ["a1_m0", "a2_m0"], ["a1_m1", "a2_m1"]),
    "0.6B B": (os.path.join(base, "results"), ["b1_m0", "b3_m0"], ["b2_m2", "b4_m2"]),
    "4B A": (os.path.join(base, "results-run2/out_qwen4b"), ["a1_m0", "a2_m0"], ["a1_m2", "a2_m2"]),
    "4B B": (os.path.join(base, "results-run2/out_qwen4b"), ["b1_m0", "b3_m0"], ["b2_m2", "b4_m2"]),
    "30B A": (os.path.join(base, "results-run2/out_qwen30"), ["a1_m0", "a2_m0"], ["a1_m2", "a2_m2"]),
    "30B B": (os.path.join(base, "results-run2/out_qwen30"), ["b1_m0", "b3_m0"], ["b2_m2", "b4_m2"]),
    "120B A": (os.path.join(base, "results-run2/out_gptoss"), ["a1_m0", "a2_m0"], ["a1_m2", "a2_m2"]),
    "120B B": (os.path.join(base, "results-run2/out_gptoss"), ["b1_m0", "b3_m0"], ["b2_m2", "b4_m2"]),
}


def arm(d, tag):
    b = json.load(open(os.path.join(d, tag + ".bench.json")))
    s = json.load(open(os.path.join(d, tag + ".json")))
    tpf = b["mean_itl_ms"] / b["mean_tpot_ms"]
    upt = s["frontend_us_per_token"]
    return upt, upt * tpf, tpf


print("%-11s %-28s %-28s %s" % ("case", "per-TOKEN stock/fast/d/%", "per-FRAME stock/fast/d/%", "tok-per-frame stock/fast"))
for name, (d, m0, mf) in sets.items():
    s0 = [arm(d, t) for t in m0]
    sf = [arm(d, t) for t in mf]
    t0 = st.median([x[0] for x in s0]); tf = st.median([x[0] for x in sf])
    f0 = st.median([x[1] for x in s0]); ff = st.median([x[1] for x in sf])
    r0 = st.median([x[2] for x in s0]); rf = st.median([x[2] for x in sf])
    print("%-11s %6.2f %6.2f %6.2f %5.1f%%    %6.2f %6.2f %6.2f %5.1f%%     %.4f %.4f"
          % (name, t0, tf, t0 - tf, 100 * (t0 - tf) / t0, f0, ff, f0 - ff, 100 * (f0 - ff) / f0, r0, rf))
