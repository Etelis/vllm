# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one A/B arm: launch vLLM, pin the frontend, benchmark, measure CPU.

Usage: run_arm.py <mode 0|1|2> <label> <pin: 1|0> <num_prompts> <concurrency>

Emits /results/sse/out/<label>.json with the bench result plus the API-server
process CPU time consumed over the measured window.
"""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

MODE, LABEL, PIN, NPROMPTS, CONC = (
    int(sys.argv[1]),
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
)
OUT = os.environ.get("BENCH_OUT", "/results/sse/out")
os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-0.6B")
GPU_UTIL = os.environ.get("BENCH_GPU_UTIL", "0.85")
SERVE_EXTRA = os.environ.get("BENCH_SERVE_EXTRA", "")
PORT = 8000
HZ = os.sysconf("SC_CLK_TCK")


def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(") ", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / HZ  # utime + stime


def find_engine_pid(parent):
    # `comm` is truncated to 15 chars -> "VLLM::EngineCor"; match the prefix
    # and do not require a direct parent link.
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/comm") as f:
                if "EngineCor" in f.read():
                    return int(entry)
        except OSError:
            continue
    return None


def affinity(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


env = dict(os.environ, VLLM_SSE_FASTPATH=str(MODE), VLLM_USE_RUST_FRONTEND="0")
cmd = (
    f"vllm serve {MODEL} --port {PORT} --max-num-seqs 512 "
    f"--max-model-len 2048 --gpu-memory-utilization {GPU_UTIL} "
    f"--api-server-count 1 {SERVE_EXTRA}"
)
log = open(f"{OUT}/server_{LABEL}.log", "w")
proc = subprocess.Popen(shlex.split(cmd), env=env, stdout=log, stderr=subprocess.STDOUT)

# wait for readiness
deadline = time.time() + 2400
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        break
    except Exception:
        if proc.poll() is not None:
            sys.exit(f"server died, see server_{LABEL}.log")
        time.sleep(3)
else:
    sys.exit("server never became ready")

fe_pid = proc.pid
eng_pid = find_engine_pid(fe_pid)
print(f"[{LABEL}] frontend pid={fe_pid} engine pid={eng_pid} mode={MODE} pin={PIN}")

def allowed_cpus():
    """CPUs this container may actually use. Under a cgroup cpuset these are
    NOT 0..N-1 -- asking for a CPU outside the set makes taskset fail with
    EINVAL, which is silent unless the return code is checked."""
    cpus = []
    for part in affinity(os.getpid()).split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def pin(pid, cpu_list, what):
    spec = ",".join(str(c) for c in cpu_list)
    r = subprocess.run(
        ["taskset", "-acp", spec, str(pid)], capture_output=True, text=True
    )
    got = affinity(pid)
    if r.returncode != 0 or got != spec:
        sys.exit(
            f"FATAL: failed to pin {what} (pid {pid}) to {spec}: "
            f"rc={r.returncode} {r.stderr.strip()} -- affinity is {got}"
        )
    print(f"[{LABEL}] pinned {what} to {spec}")


if PIN:
    cpus = allowed_cpus()
    if len(cpus) < 12:
        sys.exit(f"FATAL: need >=12 usable CPUs to pin, container has {len(cpus)}")
    pin(fe_pid, cpus[:1], "frontend")
    if eng_pid:
        pin(eng_pid, cpus[1:9], "engine")
    CLIENT_CPUS = ",".join(str(c) for c in cpus[9:])
else:
    CLIENT_CPUS = None
print(f"[{LABEL}] affinity frontend={affinity(fe_pid)} engine={affinity(eng_pid)}")

# Capture raw SSE bytes for a fixed greedy request set, for cross-arm diffing.
sample_path = f"{OUT}/{LABEL}.sse"
with open(sample_path, "wb") as sf:
    for prompt in (
        "The capital of France is",
        "def fib(n):",
        "Ünïcödé ✓ 日本語 🚀 test",
    ):
        body = json.dumps(
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 48,
                "temperature": 0,
                "seed": 1234,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        sf.write(urllib.request.urlopen(req, timeout=120).read())

# Frames actually emitted per generated token, under load. vLLM merges
# undelivered deltas, so frames != tokens and the two arms need not coalesce
# equally; this puts the CPU comparison on a measured per-frame basis.
fp = subprocess.run(
    shlex.split(f"python3 frames_probe.py {MODEL} 128 256 256"),
    capture_output=True,
    text=True,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
try:
    frames_stats = json.loads(fp.stdout)
except Exception:
    frames_stats = {"error": (fp.stderr or fp.stdout)[-300:]}

bench = (
    f"vllm bench serve --model {MODEL} --port {PORT} --dataset-name random "
    f"--random-input-len 32 --random-output-len 512 --ignore-eos "
    f"--num-prompts {NPROMPTS} --max-concurrency {CONC} --percentile-metrics ttft,tpot,itl "
    f"--save-result --result-dir {OUT} --result-filename {LABEL}.bench.json"
)
if PIN and CLIENT_CPUS:
    bench = f"taskset -c {CLIENT_CPUS} " + bench

# warmup so the first arm isn't penalised by lazy init
subprocess.run(
    shlex.split(
        f"vllm bench serve --model {MODEL} --port {PORT} --dataset-name random "
        f"--random-input-len 32 --random-output-len 64 --ignore-eos "
        f"--num-prompts 64 --max-concurrency 32"
    ),
    capture_output=True,
)

# Idle draw of the frontend, so the CPU consumed during the benchmark's own
# startup (tokenizer load, dataset generation) can be subtracted out: the CPU
# window is wider than the request phase the benchmark reports.
idle_t0, idle_c0 = time.time(), cpu_seconds(fe_pid)
time.sleep(5)
idle_cps = (cpu_seconds(fe_pid) - idle_c0) / (time.time() - idle_t0)

t0, c0, e0 = time.time(), cpu_seconds(fe_pid), cpu_seconds(eng_pid) if eng_pid else 0
r = subprocess.run(shlex.split(bench), capture_output=True, text=True)
t1, c1, e1 = time.time(), cpu_seconds(fe_pid), cpu_seconds(eng_pid) if eng_pid else 0

stdout = r.stdout
res = {"label": LABEL, "mode": MODE, "pin": PIN, "conc": CONC, "nprompts": NPROMPTS}
for key, pat in (
    ("output_throughput", r"Output token throughput \(tok/s\):\s+([\d.]+)"),
    ("total_throughput", r"Total Token throughput \(tok/s\):\s+([\d.]+)"),
    ("req_throughput", r"Request throughput \(req/s\):\s+([\d.]+)"),
    ("mean_tpot_ms", r"Mean TPOT \(ms\):\s+([\d.]+)"),
    ("p99_tpot_ms", r"P99 TPOT \(ms\):\s+([\d.]+)"),
    ("mean_itl_ms", r"Mean ITL \(ms\):\s+([\d.]+)"),
    ("duration_s", r"Benchmark duration \(s\):\s+([\d.]+)"),
    ("total_output_tokens", r"Total generated tokens:\s+([\d.]+)"),
):
    m = re.search(pat, stdout)
    res[key] = float(m.group(1)) if m else None

res["frontend_cpu_s"] = round(c1 - c0, 3)
res["engine_cpu_s"] = round(e1 - e0, 3)
res["wall_s"] = round(t1 - t0, 2)
res["frontend_affinity"] = affinity(fe_pid)
res["engine_affinity"] = affinity(eng_pid)
res["frontend_idle_cores"] = round(idle_cps, 4)
res["frames"] = frames_stats
if res["total_output_tokens"]:
    res["frontend_us_per_token"] = round(
        (c1 - c0) / res["total_output_tokens"] * 1e6, 2
    )
if res["duration_s"]:
    # CPU attributable to the request phase: total minus the idle draw over
    # the part of the window that was not the request phase.
    idle_window = max(0.0, (t1 - t0) - res["duration_s"])
    bench_cpu = (c1 - c0) - idle_cps * idle_window
    res["frontend_cpu_s_corrected"] = round(bench_cpu, 3)
    # cores consumed by the frontend during the request phase; ~1.0 means it
    # is saturated on its single event loop and is therefore the bottleneck.
    res["frontend_cores"] = round(bench_cpu / res["duration_s"], 3)
    res["frontend_cores_raw"] = round((c1 - c0) / res["duration_s"], 3)
    if res["total_output_tokens"]:
        res["frontend_us_per_token_corrected"] = round(
            bench_cpu / res["total_output_tokens"] * 1e6, 2
        )

with open(f"{OUT}/{LABEL}.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
if r.returncode != 0:
    print("BENCH STDERR:", r.stderr[-2000:])

proc.send_signal(signal.SIGINT)
try:
    proc.wait(timeout=60)
except subprocess.TimeoutExpired:
    proc.kill()
