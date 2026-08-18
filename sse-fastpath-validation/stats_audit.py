import json, math, os
base="/Users/itayetlis/vllm/.claude/worktrees/sse-fastpath-validation/sse-fastpath-validation"
cells = {
 "Qwen3-0.6B": (base+"/results", {"A":[("stock",["a1_m0","a2_m0"]),("hardcoded",["a1_m1","a2_m1"]),("derived",["a1_m2","a2_m2"])],
                                  "B":[("stock",["b1_m0","b3_m0"]),("derived",["b2_m2","b4_m2"])]}),
 "Qwen3-4B": (base+"/results-run2/out_qwen4b", None),
 "Qwen3-30B": (base+"/results-run2/out_qwen30", None),
 "gpt-oss-120b": (base+"/results-run2/out_gptoss", None),
}
default = {"A":[("stock",["a1_m0","a2_m0"]),("derived",["a1_m2","a2_m2"])],
           "B":[("stock",["b1_m0","b3_m0"]),("derived",["b2_m2","b4_m2"])]}
TCRIT=12.706  # two-sided 95%, df=1 (worst case for Welch with n=2/n=2)

def load(d,l): return json.load(open(os.path.join(d,l+".json")))

def stat(vals):
    m=sum(vals)/len(vals); rng=max(vals)-min(vals)
    se=rng/2.0   # n=2: s=range/sqrt(2), se=s/sqrt(2)=range/2
    return m,rng,se

for model,(d,spec) in cells.items():
    spec = spec or default
    print("="*92); print(model)
    for phase,arms in spec.items():
        print("  -- Phase "+phase)
        for metric,key,fmt in (("tok/s","output_throughput","%.1f"),("FE us/tok","frontend_us_per_token","%.2f"),("FEcores","frontend_cores","%.3f")):
            stats={}
            line=[]
            for name,labels in arms:
                vals=[load(d,l)[key] for l in labels]
                m,rng,se=stat(vals)
                stats[name]=(m,rng,se,vals)
                line.append(f"{name}: {vals} mean={fmt%m} range={fmt%rng} ({rng/m*100:.2f}%)")
            print(f"    {metric:9s} " + " | ".join(line))
            b=stats["stock"]
            for name,(m,rng,se,vals) in stats.items():
                if name=="stock": continue
                diff=m-b[0]; sed=math.sqrt(se**2+b[2]**2)
                t=diff/sed if sed else float('inf')
                ci=TCRIT*sed
                verdict='SIGNIFICANT' if abs(t)>TCRIT else 'NOT distinguishable from 0'
                print(f"      {name:10s} delta={diff:+9.2f} ({diff/b[0]*100:+6.2f}%)  t={t:+7.2f}  "
                      f"95%CI(df=1)=[{diff-ci:+.1f},{diff+ci:+.1f}] -> {verdict}  (CI +-{ci/b[0]*100:.1f}%)")
