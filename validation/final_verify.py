"""Check continuity from actual startup and successful final KV cleanup."""

import argparse
import json
import time
from pathlib import Path

import httpx

from summarize_results import metric_total
from validate_probe import compare


parser = argparse.ArgumentParser()
parser.add_argument("run_dir", nargs="?", default="/workspace/validation")
base = Path(parser.parse_args().run_dir)
static = json.loads((base / "final-static-results.json").read_text())
final = json.loads((base / "results.json").read_text())
edge = json.loads((base / "edge-results.json").read_text())
summary = static["summary"]
assert not summary["fatal_error"] and summary["worker_errors"] == [], summary
assert summary["requests"] > 0 and summary["errors"] == 0, summary
assert summary["output_mismatches"] == 0 and summary["transitions"] == 0, summary
for i in range(3):
    compare(static["startup_probes"][i], static["initial_probes"][i], 2)
    compare(static["initial_probes"][i], static["final_probes"][i], 2)
    compare(static["startup_probes"][i], final["startup_probes"][i], 2)
    compare(final["startup_probes"][i], final["final_probes"][i], 2)
compare(edge["probe_before"], edge["probe_after"], 2)
assert edge["held_lease_rollback_passed"]
assert edge["stale_epoch_status"] == 409
with httpx.Client(timeout=60) as client:
    deadline = time.monotonic() + 15
    while True:
        probes = []
        for port in (8100, 8110, 8120):
            response = client.get(f"http://127.0.0.1:{port}/pd_probe")
            response.raise_for_status()
            probes.append(response.json())
        if all(core["drained"] for p in probes for core in p["core_roles"]):
            break
        assert time.monotonic() < deadline, [p["core_roles"] for p in probes]
        time.sleep(0.1)
    metrics = [client.get(f"http://127.0.0.1:{port}/metrics").text for port in (8100, 8110, 8120)]
for i, probe in enumerate(probes):
    compare(final["final_probes"][i], probe, 2)
    assert probe["frontend_role"]["active_requests"] == 0
    assert probe["frontend_role"]["phase"] == "ready"
for metric in metrics:
    for name in (
        "vllm:nixl_num_failed_transfers_total",
        "vllm:nixl_num_failed_notifications_total",
        "vllm:nixl_num_kv_expired_reqs_total",
    ):
        assert metric_total(metric, name) == 0, name
result = {
    "startup_to_final_identity_continuity": True,
    "edge_case_identity_continuity": True,
    "all_six_cores_drained": True,
    "final_epochs": [p["frontend_role"]["epoch"] for p in probes],
    "final_roles": [p["frontend_role"]["role"] for p in probes],
    "failed_transfers_notifications_expired_leases": 0,
}
(base / "final-quiescence.json").write_text(json.dumps({"summary": result, "probes": probes, "metrics": metrics}, indent=2))
print(json.dumps(result, indent=2))
