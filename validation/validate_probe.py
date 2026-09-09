"""Check graph, allocation, process and communication continuity in exercise output.

Run: .venv/bin/python validate_probe.py results.json --ep-size 2
This checks metadata identity/counters, not tensor values or graph replay events.
"""

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_no_errors(value, path):
    if isinstance(value, dict):
        assert "inspection_error" not in value, (path, value)
        for key, child in value.items():
            check_no_errors(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            check_no_errors(child, f"{path}[{i}]")


def workers_by_ep_rank(probe, ep_size):
    check_no_errors(probe, "probe")
    workers = probe["workers"]
    assert len(workers) == ep_size, (len(workers), ep_size)
    indexed = {}
    for worker in workers:
        assert worker["probe_version"] == 2, (
            "Use validate_probe_v1.py for old artifacts"
        )
        ep = worker["communication"]["groups"]["_EP"]
        assert ep and len(ep["ranks"]) == ep_size, ep
        rank = ep["rank_in_group"]
        assert rank not in indexed, f"Duplicate EP rank {rank}"
        indexed[rank] = worker
        assert worker["pid"] > 0 and worker["process_start_ticks"] is not None
        assert worker["parameter_count"] > 0
        assert worker["parameter_pointer_fingerprint"]
        assert worker["kv_buffers"], f"Missing KV buffers on rank {rank}"
        assert worker["kv_pointer_fingerprint"] == fingerprint(worker["kv_buffers"])
        graph_rows = worker["graphs"]["entries"]
        assert graph_rows, f"No CUDA graph entries on rank {rank}"
        assert all(row.get("graph") or row.get("capture") for row in graph_rows), (
            "Uncaptured CUDA graph entry",
            rank,
            graph_rows,
        )
        assert worker["graphs"]["fingerprint"] == fingerprint(graph_rows)
        counters = worker["compilation_counters"]
        assert counters["num_cudagraph_captured"] > 0, counters
        assert counters["num_gpu_runner_capture_triggers"] > 0, counters
        assert ep["all2all_manager"]["type"].endswith("DeepEPV2All2AllManager"), ep
        assert ep["buffers"], f"No DeepEP buffers on rank {rank}"
        for buffer in ep["buffers"]:
            assert buffer["buffer"]["type"].endswith("ElasticBuffer"), buffer
            assert buffer["native_runtime"] and buffer["nccl_comm_handle"], buffer
            assert buffer["capacity"]["num_bytes"] > 0, buffer
            assert buffer["capacity"]["num_max_tokens_per_rank"] > 0, buffer
        groups = worker["communication"]["groups"]
        assert worker["communication"]["fingerprint"] == fingerprint(groups)
        workspace = worker["workspace"]
        assert workspace and workspace["locked"] and workspace["buffers"], workspace
        nixl = worker["nixl_registration"]
        metadata = nixl["metadata"]
        assert nixl["fingerprint"] == fingerprint(metadata), nixl
        for field in ("connector", "connector_worker", "nixl_wrapper", "native_agent"):
            assert metadata[field] and metadata[field]["id"] > 0, (field, metadata)
        assert metadata["registered_descriptors"], metadata
        assert all(desc["id"] > 0 for desc in metadata["registered_descriptors"])
        local_handle = metadata["initial_local_xfer_handle"]
        assert local_handle["native_handle"], metadata
        assert not local_handle.get("released"), local_handle
        if "native_agent" in local_handle:
            assert local_handle["native_agent"] == metadata["native_agent"], metadata
        assert metadata["src_blocks_data"]["data_ptr"] > 0, metadata
        assert metadata["src_blocks_data"]["shape"][0] > 0, metadata
        assert metadata["local_kv_base_addresses"], metadata
        assert all(address > 0 for address in metadata["local_kv_base_addresses"])
        assert metadata["handshake"]["agent_metadata_sha256"], metadata
    assert set(indexed) == set(range(ep_size)), sorted(indexed)
    if "core_roles" in probe:
        roles = probe["core_roles"]
        assert len(roles) == probe["engine_count"], roles
        assert len({r["dp_rank"] for r in roles}) == len(roles), roles
        assert len({(r["role"], r["epoch"]) for r in roles}) == 1, roles
        frontend = probe["frontend_role"]
        assert frontend["phase"] == "ready", frontend
        assert all(r["pending_role"] is None for r in roles), roles
        assert all(
            (r["role"], r["epoch"]) == (frontend["role"], frontend["epoch"])
            for r in roles
        ), (roles, frontend)
    return indexed


def compare(before, after, ep_size):
    initial = workers_by_ep_rank(before, ep_size)
    final = workers_by_ep_rank(after, ep_size)
    unchanged = (
        "probe_version",
        "source_revision",
        "hostname",
        "pid",
        "process_start_ticks",
        "rank",
        "local_rank",
        "device",
        "gpu_name",
        "model",
        "model_runner",
        "parameter_count",
        "parameter_pointer_fingerprint",
        "kv_buffers",
        "kv_pointer_fingerprint",
        "compilation_counters",
        "graphs",
        "communication",
        "workspace",
        "nixl_registration",
    )
    for rank in initial:
        for field in unchanged:
            assert initial[rank][field] == final[rank][field], (
                f"EP rank {rank}: {field} changed",
                initial[rank][field],
                final[rank][field],
            )


def validate_artifact(artifact, ep_size):
    startup = artifact["startup_probes"]
    initial, final = artifact["initial_probes"], artifact["final_probes"]
    assert len(startup) == len(initial) == len(final) == 3, "Expected three EP groups"
    assert artifact["transition_probes"], "No transition snapshots"
    assert len(artifact["transition_probes"]) == len(artifact["transitions"]), (
        "Each transition needs a worker snapshot"
    )
    for probe in startup + initial + final + artifact["transition_probes"]:
        assert "core_roles" in probe, "Missing all-core role status on patched run"
    for group, (before, after) in enumerate(zip(initial, final)):
        compare(startup[group], before, ep_size)
        compare(before, after, ep_size)
        print(
            f"Group {group}: {ep_size} workers preserve startup graphs and allocations"
        )
    for i, probe in enumerate(artifact["transition_probes"]):
        compare(initial[1], probe, ep_size)
        print(f"Transition {i}: all {ep_size} switched workers preserve identities")
    return {
        "groups": 3,
        "workers_per_group": ep_size,
        "transition_snapshots": len(artifact["transition_probes"]),
        "inspection_errors": 0,
        "changed_identities_or_counters": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--ep-size", type=int, default=2)
    args = parser.parse_args()
    result = validate_artifact(
        json.loads(Path(args.artifact).read_text()), args.ep_size
    )
    print(json.dumps(result, indent=2))
