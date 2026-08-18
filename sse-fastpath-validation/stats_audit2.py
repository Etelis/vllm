import json, math, os, statistics as st
base="/Users/itayetlis/vllm/.claude/worktrees/sse-fastpath-validation/sse-fastpath-validation"
dirs={"Qwen3-0.6B":base+"/results","Qwen3-4B":base+"/results-run2/out_qwen4b",
      "Qwen3-30B":base+"/results-run2/out_qwen30","gpt-oss-120b":base+"/results-run2/out_gptoss"}
groups={"A stock":["a1_m0","a2_m0"],"A hard":["a1_m1","a2_m1"],"A derived":["a1_m2","a2_m2"],
        "B stock":["b1_m0","b3_m0"],"B derived":["b2_m2","b4_m2"]}
KEYS=["median_itl_ms","mean_itl_ms","median_tpot_ms","mean_ttft_ms","duration"]
for m,d in dirs.items():
    print("="*80); print(m)
    for g,labels in groups.items():
        rows=[]
        for l in labels:
            p=os.path.join(d,l+".bench.json")
            if not os.path.exists(p): rows=None; break
            rows.append(json.load(open(p)))
        if not rows: continue
        out=[]
        for k in KEYS:
            v=[r[k] for r in rows]
            out.append(f"{k}={v[0]:.2f}/{v[1]:.2f} (spread {abs(v[0]-v[1])/st.mean(v)*100:.2f}%)")
        print(f"  {g:10s} "+"  ".join(out))
    # steady-state comparison on median ITL
    for phase,(s,f) in {"A":("A stock","A derived"),"B":("B stock","B derived")}.items():
        try:
            sv=[json.load(open(os.path.join(d,l+".bench.json")))["median_itl_ms"] for l in groups[s]]
            fv=[json.load(open(os.path.join(d,l+".bench.json")))["median_itl_ms"] for l in groups[f]]
        except FileNotFoundError: continue
        sm,fm=st.mean(sv),st.mean(fv)
        print(f"  -> Phase {phase} median-ITL stock {sm:.3f} ms vs derived {fm:.3f} ms  "
              f"=> implied steady-state tok/s change {(sm/fm-1)*100:+.2f}%  "
              f"(within-cell spreads {abs(sv[0]-sv[1])/sm*100:.2f}% / {abs(fv[0]-fv[1])/fm*100:.2f}%)")
# pooled FE us/tok deltas
print("="*80); print("POOLED FE us/token deltas (derived vs stock), one value per model/phase cell")
abs_d=[];rel_d=[]
for m,d in dirs.items():
    for phase,(s,f) in {"A":("A stock","A derived"),"B":("B stock","B derived")}.items():
        sv=st.mean([json.load(open(os.path.join(d,l+".json")))["frontend_us_per_token"] for l in groups[s]])
        fv=st.mean([json.load(open(os.path.join(d,l+".json")))["frontend_us_per_token"] for l in groups[f]])
        abs_d.append(fv-sv); rel_d.append((fv/sv-1)*100)
        print(f"  {m:14s} {phase}  {sv:6.2f} -> {fv:6.2f}   {fv-sv:+6.2f} us ({(fv/sv-1)*100:+6.2f}%)")
for name,vals in (("absolute us",abs_d),("relative %",rel_d)):
    mu=st.mean(vals); sd=st.stdev(vals); se=sd/math.sqrt(len(vals)); ci=2.365*se
    print(f"  pooled {name}: mean={mu:+.2f} sd={sd:.2f} n={len(vals)} 95%CI=[{mu-ci:+.2f},{mu+ci:+.2f}]  min={min(vals):+.2f} max={max(vals):+.2f}")
