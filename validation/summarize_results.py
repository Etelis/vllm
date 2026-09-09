"""Derive a compact, checkable report from the complete traffic artifact."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from validate_probe import validate_artifact


def metric_total(text, name):
    lines = [
        line
        for line in text.splitlines()
        if line.startswith(name + "{")
    ]
    assert lines, f"Missing metric series: {name}"
    engines = {re.search(r'engine="([^"]+)"', line).group(1) for line in lines}
    assert engines == {"0", "1"}, (name, engines)
    values = [float(line.rsplit(" ", 1)[1]) for line in lines]
    assert all(value >= 0 for value in values), (name, values)
    return sum(values)


def main(args):
    data = json.loads(Path(args.artifact).read_text())
    probe = validate_artifact(data, 2)
    summary = data["summary"]
    assert not summary["fatal_error"] and summary["errors"] == 0, summary
    assert summary["worker_errors"] == [], summary
    assert summary["output_mismatches"] == 0, summary
    reference_count = len(data["references"])
    records = data["records"][reference_count:]
    assert len(records) == summary["requests"]
    assert all(r["remote_blocks"] > 0 and r["remote_num_tokens"] > 0 for r in records)
    assert all(r["text"] == data["references"][str(r["prompt"])] for r in records)
    counters = {}
    for name in (
        "vllm:nixl_bytes_transferred_sum",
        "vllm:nixl_xfer_time_seconds_count",
        "vllm:nixl_num_failed_transfers_total",
        "vllm:nixl_num_failed_notifications_total",
        "vllm:nixl_num_kv_expired_reqs_total",
    ):
        values = [
            metric_total(after, name) - metric_total(before, name)
            for before, after in zip(data["metrics_before"], data["metrics_after"])
        ]
        assert all(value >= 0 for value in values), (name, values)
        counters[name] = {"by_group": values, "total": sum(values)}
    assert counters["vllm:nixl_bytes_transferred_sum"]["by_group"][1] > 0
    assert counters["vllm:nixl_xfer_time_seconds_count"]["by_group"][1] > 0
    for name, value in counters.items():
        if "failed" in name or "expired" in name:
            assert value["total"] == 0, (name, value)
    workers = [w for p in data["final_probes"] for w in p["workers"]]
    transitions = [
        {k: v for k, v in event.items() if k != "samples"}
        for event in data["transitions"]
    ]
    report = {
        "summary": summary,
        "source_revisions": sorted({w["source_revision"] for w in workers}),
        "reference_requests": reference_count,
        "tokens_per_request": data["args"]["tokens"],
        "requests_per_prompt": dict(sorted(Counter(r["prompt"] for r in records).items())),
        "identity_validation": probe,
        "workers": [
            {
                "pid": w["pid"],
                "gpu": w["gpu_name"],
                "graph_entries": len(w["graphs"]["entries"]),
                "captures": w["compilation_counters"]["num_cudagraph_captured"],
                "capture_triggers": w["compilation_counters"]["num_gpu_runner_capture_triggers"],
                "kv_buffers": len(w["kv_buffers"]),
            }
            for w in workers
        ],
        "nixl_metric_deltas": counters,
        "transitions": transitions,
        "all_transitions_have_unaffected_decode_chunks": all(
            e["unaffected_decode_chunks_during_transition"] > 0 for e in transitions
        ),
        "worst_unaffected_decode_gap_during_transition_seconds": max(
            e["unaffected_decode_max_gap_seconds"] for e in transitions
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "transitions"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
