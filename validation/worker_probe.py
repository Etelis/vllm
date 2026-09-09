"""Read-only worker evidence for planned P/D transitions on vLLM c7e9816c6a.

Add this directory to PYTHONPATH and launch with
--worker-extension-cls worker_probe.PDRoleSwitchProbe. Call
collective_rpc("pd_probe") at quiescent boundaries. Optionally add
--middleware worker_probe.ProbeMiddleware and GET /pd_probe. Snapshots
inspect host metadata only; they do not synchronize or launch GPU work.
Identity fingerprints do not checksum model/KV values or prove that a graph
replayed. Keep request/output checks and timing evidence separately.
"""

import dataclasses
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def _identity(obj):
    if obj is None:
        return None
    cls = type(obj)
    return {"id": id(obj), "type": f"{cls.__module__}.{cls.__qualname__}"}


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_metadata(tensor):
    return {
        "data_ptr": tensor.data_ptr(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _tensor_tree(value, path=""):
    import torch

    if isinstance(value, torch.Tensor):
        return [{"path": path, **_tensor_metadata(value)}]
    if isinstance(value, dict):
        return [
            row
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            for row in _tensor_tree(child, f"{path}/{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            row
            for i, child in enumerate(value)
            for row in _tensor_tree(child, f"{path}/{i}")
        ]
    return []


def _graphs(runner):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    rows = []
    for wrapper in list(CUDAGraphWrapper._all_instances):
        for descriptor, entry in list(wrapper.concrete_cudagraph_entries.items()):
            rows.append(
                {
                    "wrapper": _identity(wrapper),
                    "mode": str(wrapper.runtime_mode),
                    "pool": list(wrapper.graph_pool or []),
                    "descriptor": repr(descriptor),
                    "entry": _identity(entry),
                    "graph": _identity(entry.cudagraph),
                    "input_addresses": entry.input_addresses,
                }
            )
    for wrapper in list(BreakableCUDAGraphWrapper._all_instances):
        for descriptor, entry in list(wrapper.entries.items()):
            capture = entry.capture
            segments = [] if capture is None else capture.segments
            rows.append(
                {
                    "wrapper": _identity(wrapper),
                    "mode": "breakable",
                    "pool": list(wrapper.graph_pool or []),
                    "descriptor": repr(descriptor),
                    "entry": _identity(entry),
                    "capture": _identity(capture),
                    "segments": [
                        _identity(getattr(segment, "__self__", segment))
                        for segment in segments
                    ],
                    "input_addresses": entry.input_addresses,
                }
            )
    manager = getattr(runner, "cudagraph_manager", None)
    if manager is not None:
        for descriptor, graph in list(manager.graphs.items()):
            rows.append(
                {
                    "wrapper": _identity(manager),
                    "mode": str(manager.cudagraph_mode),
                    "descriptor": repr(descriptor),
                    "graph": _identity(graph),
                }
            )
    rows.sort(key=lambda row: (row["wrapper"]["id"], row["descriptor"]))
    return {"entries": rows, "fingerprint": _fingerprint(rows)}


def _communication():
    from vllm.distributed import parallel_state

    groups = {}
    for name in ("_TP", "_DP", "_EP", "_PP"):
        group = getattr(parallel_state, name, None)
        if group is None:
            groups[name] = None
            continue
        communicator = group.device_communicator
        manager = getattr(communicator, "all2all_manager", None)
        buffers = []
        cache = getattr(manager, "handle_cache", None)
        if cache is not None:
            with cache._lock:
                handles = list(cache._cache.values())
            for handle in handles:
                attrs = vars(handle)
                scalars = {
                    key: attrs[key]
                    for key in (
                        "num_max_tokens_per_rank",
                        "num_bytes",
                        "num_allocated_qps",
                        "num_ranks",
                        "allow_hybrid_mode",
                        "prefer_overlap_with_compute",
                        "allow_multiple_reduction",
                    )
                    if key in attrs
                }
                buffers.append(
                    {
                        "buffer": _identity(handle),
                        "native_runtime": _identity(attrs.get("runtime")),
                        "nccl_comm_handle": _identity(attrs.get("nccl_comm_handle")),
                        "capacity": scalars,
                    }
                )
        buffers.sort(key=lambda row: row["buffer"]["id"])
        groups[name] = {
            "group": _identity(group),
            "ranks": list(group.ranks),
            "rank_in_group": group.rank_in_group,
            "cpu_group": _identity(group.cpu_group),
            "device_group": _identity(group.device_group),
            "communicator": _identity(communicator),
            "pynccl_communicator": _identity(
                getattr(communicator, "pynccl_comm", None)
            ),
            "all2all_manager": _identity(manager),
            "buffers": buffers,
        }
    return {"groups": groups, "fingerprint": _fingerprint(groups)}


def _workspace():
    from vllm.v1.worker import workspace

    manager = workspace._manager
    if manager is None:
        return None
    return {
        "manager": _identity(manager),
        "locked": manager.is_locked(),
        "buffers": _tensor_tree(manager._current_workspaces),
    }


def _nixl_handle(handle):
    if isinstance(handle, int):
        return {"native_handle": handle}
    native_handle = getattr(handle, "_handle", None)
    return {
        "object": _identity(handle),
        "native_handle": native_handle
        if isinstance(native_handle, int)
        else _identity(native_handle),
        "native_agent": _identity(getattr(handle, "_agent", None)),
        "released": getattr(handle, "_released", None),
    }


def _nixl_registration():
    from vllm.distributed.kv_transfer import kv_transfer_state

    connector = kv_transfer_state._KV_CONNECTOR_AGENT
    worker = connector.connector_worker
    wrapper = worker.nixl_wrapper
    blocks = worker.src_blocks_data
    handshake = worker.xfer_handshake_metadata
    metadata = {
        "connector": _identity(connector),
        "connector_worker": _identity(worker),
        "nixl_wrapper": _identity(wrapper),
        "native_agent": _identity(wrapper.agent),
        "agent_name": wrapper.name,
        "engine_id": worker.engine_id,
        "block_size": worker.block_size,
        "registered_descriptor_list": _identity(worker._registered_descs),
        "registered_descriptors": [
            _identity(desc) for desc in worker._registered_descs
        ],
        "initial_local_xfer_handle": _nixl_handle(
            worker.src_xfer_handles_by_block_size[worker.block_size]
        ),
        "src_blocks_data": {
            "object": _identity(blocks),
            "data_ptr": blocks.__array_interface__["data"][0],
            "shape": list(blocks.shape),
            "strides": list(blocks.strides),
            "dtype": str(blocks.dtype),
        },
        "local_kv_base_addresses": list(
            worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank]
        ),
        "region_num_blocks": list(worker.region_num_blocks),
        "region_mem_types": list(worker.region_mem_types),
        "block_len_per_layer": list(worker.block_len_per_layer),
        "block_stride_per_layer": list(worker.block_stride_per_layer),
        "handshake": {
            "object": _identity(handshake),
            "compatibility_hash": handshake.compatibility_hash,
            "agent_metadata_sha256": hashlib.sha256(
                handshake.agent_metadata_bytes
            ).hexdigest(),
        },
    }
    return {"metadata": metadata, "fingerprint": _fingerprint(metadata)}


class PDRoleSwitchProbe:
    def pd_probe(self):
        import torch

        from vllm.compilation.counter import compilation_counter

        runner = self.model_runner
        model = self.get_model()
        params = [
            {"name": name, **_tensor_metadata(param)}
            for name, param in sorted(model.named_parameters())
        ]
        kv_buffers = _tensor_tree(getattr(runner, "kv_caches", []))
        try:
            process_start_ticks = (
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, IndexError):
            process_start_ticks = None
        result = {
            "probe_version": 2,
            "source_revision": os.environ.get("PD_TEST_SOURCE_REVISION"),
            "timestamp_ns": time.time_ns(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "process_start_ticks": process_start_ticks,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device),
            "model": _identity(model),
            "model_runner": _identity(runner),
            "parameter_count": len(params),
            "parameter_pointer_fingerprint": _fingerprint(params),
            "kv_buffers": kv_buffers,
            "kv_pointer_fingerprint": _fingerprint(kv_buffers),
            "compilation_counters": dataclasses.asdict(compilation_counter),
        }
        for name, operation in (
            ("graphs", lambda: _graphs(runner)),
            ("communication", _communication),
            ("workspace", _workspace),
            ("nixl_registration", _nixl_registration),
        ):
            try:
                result[name] = operation()
            except Exception as exc:
                result[name] = {"inspection_error": f"{type(exc).__name__}: {exc}"}
        return result


class ProbeMiddleware:
    """Optional private validation endpoint; forwards other requests untouched."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "GET"
            or scope.get("path") != "/pd_probe"
        ):
            return await self.app(scope, receive, send)

        import asyncio

        from starlette.responses import JSONResponse

        client = scope["app"].state.engine_client
        core = client.engine_core
        args = ("collective_rpc", "pd_probe", 30.0, (), None)
        try:
            call_all = getattr(core, "call_utility_all_async", None)
            if call_all is not None:
                results = await call_all(*args)
            else:
                results = await asyncio.gather(
                    *[
                        core._call_utility_async(*args, engine=engine)
                        for engine in core.core_engines
                    ]
                )
            payload = {
                "engine_count": len(results),
                "workers": [worker for result in results for worker in result],
            }
            role_state = getattr(scope["app"].state, "pd_role", None)
            if role_state is not None:
                payload["core_roles"] = await core.call_utility_all_async(
                    "get_pd_role_status"
                )
                payload["frontend_role"] = role_state.status()
            response = JSONResponse(payload)
        except Exception as exc:
            response = JSONResponse(
                {"inspection_error": f"{type(exc).__name__}: {exc}"},
                status_code=503,
            )
        await response(scope, receive, send)
