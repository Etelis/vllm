"""Stop this completed benchmark fleet before loading the next model."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "pgid": int(fields[2]),
        "start_ticks": int(fields[19]),
        "state": fields[0],
    }


def group_members(pgids):
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            info = process_info(int(entry.name))
            if info and info["pgid"] in pgids and info["state"] != "Z":
                members.append(info)
    return sorted(members, key=lambda info: info["pid"])


def signal_process(expected, sig):
    # Recheck group membership and process identity before targeting this PID.
    current = process_info(expected["pid"])
    if current is None or current["state"] == "Z":
        return False
    if any(current[key] != expected[key] for key in ("pid", "pgid", "start_ticks")):
        return False
    try:
        os.kill(expected["pid"], sig)
    except ProcessLookupError:
        return False
    return True


def gpu_compute_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})


def main(args):
    if not args.allow_failed_test:
        proof = json.loads((args.run_dir / "final-quiescence.json").read_text())
        assert proof["summary"]["all_six_cores_drained"], proof["summary"]
    with httpx.Client(timeout=60) as client:
        probes = []
        for port in (8100, 8110, 8120):
            response = client.get(f"http://127.0.0.1:{port}/pd_probe")
            response.raise_for_status()
            probes.append(response.json())
    assert all(p["frontend_role"]["active_requests"] == 0 for p in probes), probes
    assert all(c["drained"] for p in probes for c in p["core_roles"]), probes
    (args.run_dir / "shutdown-probes.json").write_text(json.dumps({
        "after_failed_test": args.allow_failed_test, "probes": probes,
    }, indent=2))

    parents = []
    for group in "abc":
        pid = int((args.run_dir / f"server-{group}.pid").read_text())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        assert b"vllm.entrypoints.cli.main serve" in command, (pid, command)
        info = process_info(pid)
        assert info and info["pgid"] == pid and info["state"] != "Z", info
        parents.append(info)
    pgids = {info["pgid"] for info in parents}
    assert len(pgids) == 3 and os.getpgrp() not in pgids, pgids
    report = {
        "server_pids": [info["pid"] for info in parents],
        "recorded_pgids": sorted(pgids),
        "after_failed_test": args.allow_failed_test,
        "forced_pids": [],
        "success": False,
    }
    try:
        for parent in parents:
            signal_process(parent, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while (remaining := group_members(pgids)) and time.monotonic() < deadline:
            time.sleep(0.25)
        report["survivors_after_graceful_wait"] = remaining

        deadline = time.monotonic() + 10
        while remaining:
            for info in remaining:
                if signal_process(info, signal.SIGKILL):
                    report["forced_pids"] = sorted(set(report["forced_pids"]) | {info["pid"]})
            time.sleep(0.1)
            remaining = group_members(pgids)
            if remaining and time.monotonic() >= deadline:
                raise TimeoutError(f"Processes remain in recorded groups: {remaining}")
        report["remaining_group_members"] = remaining

        deadline = time.monotonic() + 10
        while (gpu_pids := gpu_compute_pids()):
            report["remaining_gpu_compute_pids"] = gpu_pids
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GPU compute processes remain: {gpu_pids}")
            time.sleep(0.25)
        report["remaining_gpu_compute_pids"] = []
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.run_dir / "shutdown-result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-failed-test", action="store_true")
    main(parser.parse_args())
