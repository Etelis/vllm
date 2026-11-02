# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault tolerance for DeepEP v2 builds without a rank mask ("ghost rank").

DeepEP v2 kernels wait on values that peers write into the local workspace,
and trap once ``num_gpu_timeout_secs`` passes. When an EP peer dies, a watchdog
writes the dead peer's part into the survivor's own workspace, so in-flight
kernels complete instead of trapping: its arrival in each NVLink barrier, and
"0 tokens" in its dispatch count slots. After recovery, a small kernel before
each dispatch and combine keeps playing the dead ranks from a dead count held on
the device, so CUDA graphs and JIT-compiled kernels stay valid; alternatively
the buffers are rebuilt over the survivors. Single NVLink domain only: peers are
watched by PID.

Offsets follow deep_ep/include/deep_ep/common/layout.cuh at the DeepEP commit
vLLM pins; a consistency check at registration rejects other layouts.
"""

import os
import threading
import time
import uuid

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_MAX_RANKS, _MAX_EXPERTS, _MAX_EXPERTS_PER_RANK = 1024, 2048, 256
# Buffers per board row (shared barrier counters of all ranks), then a flag
# a rank sets when it left a step on a fault
_MAX_GHOSTS = 16
_LEFT = _MAX_GHOSTS
# A rank whose last step started longer ago than this may be stuck: poll it
_POLL_AFTER_SECS = 1.0
# int32 signal[phase] at (2 + phase) * 4; int64 counter at 0
_OFF_SIGNALS = 8
# int64 receive rank counts, then expert counts
_OFF_RECV_COUNTS = 16 + 2 * (_MAX_RANKS + _MAX_EXPERTS) * 8
# WorkspaceLayout::get_num_bytes() aligned to 2 MiB, in front of the buffer
_WORKSPACE_BYTES = 14 << 20


class _CudaArray:
    def __init__(self, ptr: int, n: int, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": (n,),
            "typestr": typestr,
            "data": (ptr, False),
            "version": 2,
        }


def _device_view(ptr: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    typestr = {torch.int64: "<i8", torch.int32: "<i4"}[dtype]
    return torch.as_tensor(_CudaArray(ptr, n, typestr), device="cuda")


def _workspace_base(buffer, rank: int, num_ranks: int) -> int:
    # A fresh AGRS session's in-place tensor starts at buffer + n * rank.
    n = 4096
    buffer.agrs_set_config(n * num_ranks, 1)
    buffer.create_agrs_session()
    tensor = buffer.agrs_get_inplace_tensor((n,), torch.uint8)
    buffer.destroy_agrs_session()
    return tensor.data_ptr() - n * rank - _WORKSPACE_BYTES


@triton.jit
def _pre_op_kernel(
    counter_ptr,
    signals_ptr,
    counts_ptr,
    dead_slots_ptr,
    num_dead_ptr,
    num_slots,
    IS_DISPATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The op's NVLink barriers are counter and counter + 1: add the dead
    # ranks' arrivals (+1 or -1 by barrier sign). Live peers may be adding
    # theirs concurrently, hence atomics.
    pid = tl.program_id(0)
    num_dead = tl.load(num_dead_ptr)
    if (pid == 0) & (num_dead > 0):
        counter = tl.load(counter_ptr)
        for k in tl.static_range(2):
            barrier = counter + k
            sign = ((barrier >> 1) & 1).to(tl.int32)
            delta = num_dead * (1 - 2 * sign)
            tl.atomic_add(signals_ptr + (barrier & 1), delta)
    if IS_DISPATCH:
        # "0 tokens" (encoded -1) in the dead ranks' rank and expert count slots
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_slots
        dead = tl.load(dead_slots_ptr + offs, mask=mask, other=0)
        minus_one = tl.full([BLOCK], -1, tl.int64)
        tl.store(counts_ptr + offs, minus_one, mask=mask & (dead != 0))


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as f:
            return not any(line.startswith("State:") and "Z" in line for line in f)
    except FileNotFoundError:
        return False


class GhostBuffer:
    """Views into one ElasticBuffer's workspace."""

    def __init__(self, buffer, rank: int, num_ranks: int, num_experts: int):
        base = _workspace_base(buffer, rank, num_ranks)
        self.buffer = buffer
        self.num_ranks = num_ranks
        self.experts_per_rank = num_experts // num_ranks
        assert self.experts_per_rank <= _MAX_EXPERTS_PER_RANK
        self.counter = _device_view(base, 1, torch.int64)
        self.signals = _device_view(base + _OFF_SIGNALS, 2, torch.int32)
        self.counts = _device_view(
            base + _OFF_RECV_COUNTS,
            num_ranks * (1 + self.experts_per_rank),
            torch.int64,
        )
        self.workspace = _device_view(base, _WORKSPACE_BYTES // 8, torch.int64)
        # Dead ranks played before every op, kept on the device for CUDA graphs
        self.num_dead = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.dead_slots = torch.zeros(
            self.counts.numel(), dtype=torch.int8, device="cuda"
        )
        self.active = False
        # Compile both variants now (no-ops while nothing is dead)
        self._launch(True)
        self._launch(False)
        torch.accelerator.synchronize()
        counter, signals = int(self.counter.item()), self.signals.tolist()
        last = counter - 1
        target = num_ranks if (last >> 1) & 1 == 0 else 0
        if counter < 1 or signals[last & 1] != target:
            raise RuntimeError(
                "Unexpected DeepEP v2 workspace layout "
                f"(counter={counter}, signals={signals})"
            )

    def cover(self, dead_ranks: list[int]) -> None:
        """Play ``dead_ranks`` (buffer-world ranks) before every later op."""
        slots = torch.zeros(self.counts.numel(), dtype=torch.int8)
        for rank in dead_ranks:
            slots[rank] = 1
            lo = self.num_ranks + self.experts_per_rank * rank
            slots[lo : lo + self.experts_per_rank] = 1
        self.dead_slots.copy_(slots)
        self.num_dead.fill_(len(dead_ranks))
        self.active = bool(dead_ranks)

    def pre_op(self, is_dispatch: bool) -> None:
        """Run on the compute stream right before each dispatch and combine."""
        # Graphs captured while healthy keep the no-op launch for later faults
        if self.active or torch.cuda.is_current_stream_capturing():
            self._launch(is_dispatch)

    def _launch(self, is_dispatch: bool) -> None:
        block = 1024
        grid = (cdiv(self.counts.numel(), block) if is_dispatch else 1,)
        _pre_op_kernel[grid](
            self.counter,
            self.signals,
            self.counts,
            self.dead_slots,
            self.num_dead,
            self.counts.numel(),
            IS_DISPATCH=is_dispatch,
            BLOCK=block,
        )

    def add_arrivals(self, num_barriers: int, num_missing: int) -> None:
        """Pre-add missing arrivals for the next barriers (current stream)."""
        barriers = self.counter + torch.arange(num_barriers, device="cuda")
        sign = (barriers >> 1) & 1
        delta = ((1 - 2 * sign) * num_missing).to(torch.int32)
        self.signals.index_add_(0, barriers & 1, delta)


class GhostWatchdog:
    """Detects dead EP peers and releases the kernels stuck on them.

    The thread only issues async copies on a side stream from buffers allocated
    up front: allocations, pageable copies, default-stream work and kernel
    launches can all wait behind the stuck kernel.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        board_path: str,
        silent_secs: float,
        stable_secs: float = 0.2,
    ):
        self.stable_secs = stable_secs
        # A kernel stalled this long on live peers behind in the barrier count
        # plays them and flags a fault (like a rank mask's timeout)
        self.silent_secs = silent_secs
        # Every rank's barrier counters, by original EP rank and buffer
        self._rank = rank
        self._board = np.memmap(
            board_path, dtype=np.int64, mode="r+", shape=(world_size, _MAX_GHOSTS + 1)
        )
        # GPU polling of a healthy rank waits for steady state: a CUDA call from
        # this thread would invalidate a globally captured CUDA graph
        self.steady = False
        # Refreshed every step by the fault query; a healthy rank that keeps
        # stepping is never polled, so the watchdog stays off the GIL and GPU
        self.heartbeat = time.monotonic()
        # Set with any mask bit: the per-step fault query just returns it
        self.fault_flag = torch.zeros((), dtype=torch.bool, device="cuda")
        self._true_h = torch.ones((), dtype=torch.bool, pin_memory=True)
        self.ghosts: list[GhostBuffer] = []
        self.pids: list[int] = []
        # Original EP rank of each rank in the buffers' world
        self.ranks: list[int] = []
        # Buffer-world ranks to play: dead peers, plus peers that left a step
        self.dead: set[int] = set()
        self.extra: set[int] = set()
        # Dead ranks already played by the pre-op kernels
        self.covered: set[int] = set()
        # Per buffer: first barrier released without waiting for a stall
        self._safe_from: dict[int, int] = {}
        self.mask = torch.zeros(world_size, dtype=torch.int32, device="cuda")
        self._mask_h = torch.zeros(world_size, dtype=torch.int32, pin_memory=True)
        self._side = torch.cuda.Stream()
        self._event = torch.cuda.Event()
        self._counter_h = torch.empty(1, dtype=torch.int64, pin_memory=True)
        self._signals_h = torch.empty(2, dtype=torch.int32, pin_memory=True)
        self._value_h = torch.empty(1, dtype=torch.int32, pin_memory=True)
        self._minus_one_h = torch.full(
            (_MAX_EXPERTS_PER_RANK,), -1, dtype=torch.int64, pin_memory=True
        )
        self._device = torch.accelerator.current_device_index()
        # The side stream and event are shared with recovery-time reads
        self._copy_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_world(self, ranks: list[int], pids: list[int]) -> None:
        self.ranks, self.pids = ranks, pids
        self.ghosts, self.covered = [], set()
        self.clear_missing()

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="deepep-v2-ghost", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def clear_missing(self) -> None:
        self.dead, self.extra, self._safe_from = set(), set(), {}
        self._board[self._rank, _LEFT] = 0

    def announce_left(self) -> None:
        """This rank left a step on a fault: peers stuck on it play it at once
        instead of after silent_secs."""
        self._board[self._rank, _LEFT] = 1

    def set_mask(self, dead_ep_ranks: list[int]) -> None:
        """Mask in original EP numbering, kept equal to the replayed dead set."""
        self._mask_h.zero_()
        self._mask_h[dead_ep_ranks] = 1
        self.mask.copy_(self._mask_h)
        self.fault_flag.zero_()
        torch.accelerator.synchronize()

    def host_mask(self) -> torch.Tensor:
        """Mask in original EP numbering, from host memory (never waits on the GPU)."""
        return self._mask_h.clone()

    def read_counters(self) -> list[int]:
        values = []
        for ghost in self.ghosts:
            self._copy(self._counter_h, ghost.counter)
            values.append(int(self._counter_h[0]))
        return values

    def _copy(self, dst: torch.Tensor, src: torch.Tensor, timeout: float = 5.0):
        with self._copy_lock:
            with torch.cuda.stream(self._side):
                dst.copy_(src, non_blocking=True)
                self._event.record(self._side)
            deadline = time.monotonic() + timeout
            while not self._event.query():
                if time.monotonic() > deadline:
                    raise TimeoutError("DeepEP v2 ghost copy did not complete")
                time.sleep(0.0001)

    def _run(self) -> None:
        torch.accelerator.set_device_index(self._device)
        stalled: dict[int, tuple[tuple, float]] = {}
        ghosted: set[tuple[int, int]] = set()
        released, last_detect = 0, 0.0
        while not self._stop.is_set():
            missing: set[int] = set()
            try:
                if time.monotonic() - last_detect >= 0.02:
                    self._detect()
                    last_detect = time.monotonic()
                missing = self.dead | self.extra
                pending = time.monotonic() - self.heartbeat > _POLL_AFTER_SECS
                if not (missing or (self.steady and pending)):
                    stalled.clear()
                for idx, ghost in enumerate(
                    self.ghosts if missing or (self.steady and pending) else []
                ):
                    self._copy(self._counter_h, ghost.counter)
                    self._copy(self._signals_h, ghost.signals)
                    counter = int(self._counter_h[0])
                    signals = self._signals_h.tolist()
                    self._board[self._rank, idx] = counter
                    if missing:
                        released += self._release(
                            idx, ghost, counter, signals, len(missing), stalled, ghosted
                        )
                    else:
                        self._check_silent(idx, ghost, counter, signals, stalled)
            except torch.AcceleratorError:
                # The CUDA context is broken; nothing left to release
                logger.exception("[FT] DeepEP v2 ghost watchdog stopping")
                break
            except Exception:
                logger.exception("[FT] DeepEP v2 ghost watchdog error")
            # A forward pass has ~4 barriers per MoE layer to release
            time.sleep(0.0002 if missing else 0.02)
        if released:
            logger.info("[FT] DeepEP v2 ghost: released %d waits", released)

    def _detect(self) -> None:
        newly = [
            i
            for i, pid in enumerate(self.pids)
            if i not in self.dead and i not in self.covered and not _alive(pid)
        ]
        if not newly:
            return
        for i in newly:
            self._mask_h[self.ranks[i]] = 1
        self._copy(self.mask, self._mask_h)
        self._copy(self.fault_flag, self._true_h)
        # A dead rank can have arrived at most at the barrier a survivor was
        # entering when it died; later barriers can be released at once.
        for idx, ghost in enumerate(self.ghosts):
            self._copy(self._counter_h, ghost.counter)
            self._safe_from[idx] = int(self._counter_h[0]) + 1
        self.dead.update(newly)
        logger.warning(
            "[FT] DeepEP v2 ghost: EP ranks %s died; playing their part",
            [self.ranks[i] for i in newly],
        )

    def _check_silent(self, idx, ghost, counter, signals, stalled) -> None:
        state, now = (counter, tuple(signals)), time.monotonic()
        if idx not in stalled or stalled[idx][0] != state:
            stalled[idx] = (state, now)
            return
        stalled_for = now - stalled[idx][1]
        if stalled_for < self.stable_secs:
            return
        barrier = counter - 1
        phase, sign = barrier & 1, (barrier >> 1) & 1
        short = ghost.num_ranks - signals[phase] if sign == 0 else signals[phase]
        behind = [
            i
            for i, r in enumerate(self.ranks)
            if r != self._rank
            and i not in self.covered
            and self._board[r, idx] < counter
        ]
        if short == 0 or not behind:
            return
        left = all(self._board[self.ranks[i], _LEFT] for i in behind)
        if stalled_for < self.silent_secs and not left:
            return
        for i in behind:
            self._mask_h[self.ranks[i]] = 1
        self._copy(self.mask, self._mask_h)
        self._copy(self.fault_flag, self._true_h)
        self.extra |= set(behind)
        logger.warning(
            "[FT] DeepEP v2 ghost: EP ranks %s %s; playing their part",
            [self.ranks[i] for i in behind],
            "left on a fault" if left else f"silent for {stalled_for:.0f} s",
        )

    def _release(
        self, idx, ghost, counter, signals, num_missing, stalled, ghosted
    ) -> int:
        barrier = counter - 1
        if self.dead and barrier < self._safe_from.get(idx, 0):
            # The dead rank may have arrived here before dying. Once the kernel
            # has made no progress for stable_secs, every live rank has arrived.
            state, now = (counter, tuple(signals)), time.monotonic()
            if idx not in stalled or stalled[idx][0] != state:
                stalled[idx] = (state, now)
                return 0
            if now - stalled[idx][1] < self.stable_secs:
                return 0
        phase, sign = barrier & 1, (barrier >> 1) & 1
        short = ghost.num_ranks - signals[phase] if sign == 0 else signals[phase]
        # Exactly the played ranks are missing, so every live rank has arrived
        # and plain writes are race-free.
        if short == num_missing and (idx, barrier) not in ghosted:
            self._value_h[0] = signals[phase] + (1 - 2 * sign) * num_missing
            self._copy(ghost.signals[phase : phase + 1], self._value_h)
            ghosted.add((idx, barrier))
            logger.debug("[FT] DeepEP v2 ghost: arrivals at barrier %d", barrier)
            return 1
        if short == 0 and (idx, -barrier - 1) not in ghosted:
            # Past the barrier: maybe waiting for dispatch counts. Idempotent,
            # as the next dispatch would expect the same "0 tokens".
            ghosted.add((idx, -barrier - 1))
            epr = ghost.experts_per_rank
            for rank in self.dead | self.extra:
                self._copy(ghost.counts[rank : rank + 1], self._minus_one_h[:1])
                lo = ghost.num_ranks + epr * rank
                self._copy(ghost.counts[lo : lo + epr], self._minus_one_h[:epr])
            logger.debug("[FT] DeepEP v2 ghost: counts after barrier %d", barrier)
            return 1
        return 0


def same_host_peers(cpu_group) -> tuple[list[int], str] | None:
    """PIDs of all ranks in ``cpu_group`` and a shared counter board in
    /dev/shm, or None if the ranks span hosts."""
    import socket

    import torch.distributed as dist

    board = ""
    if cpu_group.rank() == 0:
        board = f"/dev/shm/vllm-deepep-v2-ghost-{uuid.uuid4().hex}"
        with open(board, "wb") as f:
            f.truncate(cpu_group.size() * (_MAX_GHOSTS + 1) * 8)
    entries: list = [None] * cpu_group.size()
    host = socket.gethostname()
    dist.all_gather_object(entries, (host, os.getpid(), board), cpu_group)
    if len({h for h, _, _ in entries}) != 1:
        return None
    return [pid for _, pid, _ in entries], entries[0][2]
