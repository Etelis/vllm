# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time
import uuid

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker, _DescriptorBuilder
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCK_SIZE_FACTORS = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:0"]
NUM_MAPPINGS = [3]
NUM_MAPPINGS_PER_GROUP = [2]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("block_size_factor", BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_shared_memory", [False, True])
@torch.inference_mode()
def test_transfer(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings: int,
    gpu_page_size_bytes: int,
    block_size_factor: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_tensors: int,
    seed: int,
    device: str,
    use_shared_memory: bool,
) -> None:
    set_random_seed(seed)

    # build CanonicalKVCacheTensor list: one per tensor
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for i in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # one group containing all tensors, one data ref per tensor
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]

    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    mmap_region: SharedOffloadRegion | None = None
    if use_shared_memory:
        cpu_page_size = round_up(
            gpu_page_size_bytes * num_tensors * block_size_factor,
            SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
        )
        mmap_region = SharedOffloadRegion(
            engine_id=str(uuid.uuid4()),
            num_blocks=num_cpu_blocks,
            rank=0,
            kv_bytes_per_block=cpu_page_size,
            cpu_page_size=cpu_page_size,
        )

    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
        mmap_region=mmap_region,
    )

    # select block mappings
    gpu_blocks = random.sample(range(num_gpu_blocks), num_mappings * block_size_factor)
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # expand cpu blocks to gpu-page granularity for uniform comparison:
    # each cpu block maps to block_size_factor consecutive sub-blocks
    cpu_blocks_expanded = [
        cpu_block * block_size_factor + j
        for cpu_block in cpu_blocks
        for j in range(block_size_factor)
    ]

    # maybe skip some GPU blocks to test reading/writing from the middle of a CPU block
    blocks_to_skip = block_size_factor - 1
    if blocks_to_skip > 0:
        gpu_blocks = gpu_blocks[blocks_to_skip:]
        cpu_blocks_expanded = cpu_blocks_expanded[blocks_to_skip:]

    # set transfer direction
    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        dst_to_src = dict(zip(cpu_blocks_expanded, gpu_blocks))
        num_dst_sub_blocks = num_gpu_blocks
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_to_src = dict(zip(gpu_blocks, cpu_blocks_expanded))
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    # clone src and dst tensors before transfer
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # call transfer function via public API
    start_time = time.time()
    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    # wait for transfer to complete
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            assert finished[0].transfer_size == (
                len(gpu_blocks)
                * sum([x.page_size_bytes for x in handler.kv_cache_groups_data_refs[0]])
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity.
    for src_tensor, dst_tensor, orig_dst_tensor in zip(
        handler.src_tensors,
        handler.dst_tensors,
        orig_dst_tensors,
    ):
        # view both GPU and CPU tensors as (n, gpu_page_size_bytes) for comparison.
        src_view = src_tensor.reshape(-1, gpu_page_size_bytes)
        dst_view = dst_tensor.reshape(-1, gpu_page_size_bytes)
        orig_dst_view = orig_dst_tensor.reshape(-1, gpu_page_size_bytes)
        for dst_sub_block in range(num_dst_sub_blocks):
            src_sub_block = dst_to_src.get(dst_sub_block)
            if src_sub_block is not None:
                expected = src_view[src_sub_block]
            else:
                expected = orig_dst_view[dst_sub_block]
            torch.testing.assert_close(dst_view[dst_sub_block].cpu(), expected.cpu())

    # Drop loop-variable refs so mmap_obj has no exported buffers at cleanup.
    del orig_tensor, tensor, src_tensor, dst_tensor, orig_dst_tensor
    del src_view, dst_view, orig_dst_view, expected

    worker.shutdown()
    if mmap_region:
        mmap_region.cleanup()


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings_per_group", NUM_MAPPINGS_PER_GROUP)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("block_size_factor", BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_transfer_multi_group(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings_per_group: int,
    gpu_page_size_bytes: int,
    block_size_factor: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    seed: int,
    device: str,
) -> None:
    """Test transfers with three KV cache groups:
    - Group 0: aligned transfer with num_mappings_per_group blocks
    - Group 1: zero blocks (empty group)
    - Group 2: unaligned CPU->GPU transfer (logical_offset=block_size_factor-1,
      causing the implementation to skip source sub-blocks) with
      num_mappings_per_group blocks
    """
    set_random_seed(seed)

    # 3 groups, each with 2 tensors
    num_groups = 3
    tensors_per_group = 2
    num_tensors = num_groups * tensors_per_group
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for _ in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=g * tensors_per_group + i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(tensors_per_group)
        ]
        for g in range(num_groups)
    ]

    canonical_kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors, group_data_refs=kv_cache_groups_data_refs
    )

    worker = CPUOffloadingWorker(
        kv_caches=canonical_kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
    )

    # group 0: aligned, group 1: empty, group 2: unaligned on CPU->GPU
    group_sizes_in_cpu_blocks = [num_mappings_per_group, 0, num_mappings_per_group]

    total_cpu_blocks = sum(group_sizes_in_cpu_blocks)
    total_gpu_blocks_needed = total_cpu_blocks * block_size_factor
    gpu_blocks_all = random.sample(range(num_gpu_blocks), total_gpu_blocks_needed)
    cpu_blocks_all = random.sample(range(num_cpu_blocks), total_cpu_blocks)

    # split gpu/cpu blocks per group
    gpu_blocks_per_group: list[list[int]] = []
    cpu_blocks_per_group: list[list[int]] = []
    gpu_offset = 0
    cpu_offset = 0
    for size in group_sizes_in_cpu_blocks:
        gpu_count = size * block_size_factor
        gpu_blocks_per_group.append(gpu_blocks_all[gpu_offset : gpu_offset + gpu_count])
        cpu_blocks_per_group.append(cpu_blocks_all[cpu_offset : cpu_offset + size])
        gpu_offset += gpu_count
        cpu_offset += size

    # expand cpu blocks to gpu-page granularity
    cpu_blocks_expanded_per_group = [
        [
            cpu_block * block_size_factor + j
            for cpu_block in cpu_blocks
            for j in range(block_size_factor)
        ]
        for cpu_blocks in cpu_blocks_per_group
    ]

    # skip sub-blocks from group 2 to test unaligned transfers.
    sub_blocks_to_skip = block_size_factor - 1  # e.g. 2 when block_size_factor=3
    if sub_blocks_to_skip > 0:
        gpu_blocks_per_group[2] = gpu_blocks_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]
        cpu_blocks_expanded_per_group[2] = cpu_blocks_expanded_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]

    # build flat gpu_blocks list and group_sizes in GPU blocks
    gpu_blocks: list[int] = []
    group_sizes: list[int] = []
    for gpu_blks in gpu_blocks_per_group:
        gpu_blocks.extend(gpu_blks)
        group_sizes.append(len(gpu_blks))

    # build flat cpu_blocks list
    cpu_blocks = []
    for cpu_blks in cpu_blocks_per_group:
        cpu_blocks.extend(cpu_blks)

    # block_indices: only relevant for unaligned transfers
    block_indices: list[int] = [0, 0, sub_blocks_to_skip]

    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        # per-group mapping: cpu sub-block -> gpu sub-block
        dst_to_src_per_group = [
            dict(zip(expanded, gpu_blks))
            for expanded, gpu_blks in zip(
                cpu_blocks_expanded_per_group, gpu_blocks_per_group
            )
        ]
        num_dst_sub_blocks = num_cpu_blocks * block_size_factor
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        # per-group mapping: gpu sub-block -> cpu sub-block
        dst_to_src_per_group = [
            dict(zip(gpu_blks, expanded))
            for gpu_blks, expanded in zip(
                gpu_blocks_per_group, cpu_blocks_expanded_per_group
            )
        ]
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            expected_bytes = sum(
                group_size * sum([x.page_size_bytes for x in data_refs])
                for group_size, data_refs in zip(
                    group_sizes, handler.kv_cache_groups_data_refs
                )
            )
            assert finished[0].transfer_size == expected_bytes
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity
    for group_idx, dst_to_src in enumerate(dst_to_src_per_group):
        group_tensor_offset = group_idx * tensors_per_group
        for tensor_idx in range(tensors_per_group):
            src_tensor = handler.src_tensors[group_tensor_offset + tensor_idx]
            dst_tensor = handler.dst_tensors[group_tensor_offset + tensor_idx]
            orig_dst_tensor = orig_dst_tensors[group_tensor_offset + tensor_idx]
            src_view = src_tensor.view(-1, gpu_page_size_bytes)
            dst_view = dst_tensor.view(-1, gpu_page_size_bytes)
            orig_dst_view = orig_dst_tensor.view(-1, gpu_page_size_bytes)
            for dst_sub_block in range(num_dst_sub_blocks):
                src_sub_block = dst_to_src.get(dst_sub_block)
                if src_sub_block is not None:
                    expected = src_view[src_sub_block]
                else:
                    expected = orig_dst_view[dst_sub_block]
                torch.testing.assert_close(
                    dst_view[dst_sub_block].cpu(), expected.cpu()
                )

    worker.shutdown()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_transfer_dma_store_chunked(
    default_vllm_config, monkeypatch, seed: int, device: str
) -> None:
    """Chunked DMA store submission must copy every descriptor exactly once.

    32 KiB pages select the DMA path; DMA_CHUNK=5 with
    3 blocks x 4 tensors = 12 descriptors submits sub-batches of 5+5+2.
    """
    monkeypatch.setattr(gpu_worker, "DMA_CHUNK", 5)
    set_random_seed(seed)

    num_tensors = 4
    num_gpu_blocks = 64
    num_cpu_blocks = 256
    page_size_bytes = 32 * 1024

    kv_cache_tensors = [
        CanonicalKVCacheTensor(
            tensor=torch.zeros(
                (num_gpu_blocks, page_size_bytes), dtype=torch.int8, device=device
            ),
            page_size_bytes=page_size_bytes,
        )
        for _ in range(num_tensors)
    ]
    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=page_size_bytes)
                for i in range(num_tensors)
            ]
        ],
    )
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches, block_size_factor=1, num_cpu_blocks=num_cpu_blocks
    )
    handler = worker._store_handler
    if current_platform.is_cuda():
        assert handler._chunk_dma_batches

    gpu_blocks = random.sample(range(num_gpu_blocks), 3)
    cpu_blocks = random.sample(range(num_cpu_blocks), 3)

    for tensor in handler.src_tensors:
        tensor.random_()
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    src_spec = GPULoadStoreSpec(
        gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(0,)
    )
    assert worker.submit_store(1, src_spec, CPULoadStoreSpec(cpu_blocks))
    worker.wait({1})

    block_map = dict(zip(cpu_blocks, gpu_blocks))
    for src_tensor, dst_tensor, orig_dst in zip(
        handler.src_tensors, handler.dst_tensors, orig_dst_tensors
    ):
        for cpu_block in range(num_cpu_blocks):
            gpu_block = block_map.get(cpu_block)
            expected = (
                src_tensor[gpu_block].cpu()
                if gpu_block is not None
                else orig_dst[cpu_block]
            )
            assert torch.equal(dst_tensor[cpu_block], expected)

    worker.shutdown()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_transfer_dma_load_chunked(
    default_vllm_config, monkeypatch, seed: int, device: str
) -> None:
    """Chunked DMA load submission must copy every descriptor exactly once,
    submitting sub-batches with STREAM source access order.

    32 KiB pages select the DMA path; DMA_CHUNK=5 with
    3 blocks x 4 tensors = 12 descriptors submits sub-batches of 5+5+2.
    Stride-2 block ids keep descriptors non-contiguous on both sides so
    merge_contiguous_descriptors cannot collapse them.
    """
    monkeypatch.setattr(gpu_worker, "DMA_CHUNK", 5)
    set_random_seed(seed)

    num_tensors = 4
    num_gpu_blocks = 64
    num_cpu_blocks = 256
    page_size_bytes = 32 * 1024

    kv_cache_tensors = [
        CanonicalKVCacheTensor(
            tensor=torch.zeros(
                (num_gpu_blocks, page_size_bytes), dtype=torch.int8, device=device
            ),
            page_size_bytes=page_size_bytes,
        )
        for _ in range(num_tensors)
    ]
    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=page_size_bytes)
                for i in range(num_tensors)
            ]
        ],
    )
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches, block_size_factor=1, num_cpu_blocks=num_cpu_blocks
    )
    handler = worker._load_handler
    if current_platform.is_cuda():
        assert handler._chunk_dma_batches

    gpu_blocks = [0, 2, 4]
    cpu_blocks = [1, 3, 5]

    calls: list[tuple[int, bool]] = []
    inner_swap = handler._swap_blocks_batch

    def counting_swap(src, dst, sizes, is_src_access_order_any):
        calls.append((len(sizes), is_src_access_order_any))
        inner_swap(src, dst, sizes, is_src_access_order_any=is_src_access_order_any)

    handler._swap_blocks_batch = counting_swap

    for tensor in handler.src_tensors:
        tensor.random_()
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    dst_spec = GPULoadStoreSpec(
        gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(0,)
    )
    assert worker.submit_load(1, CPULoadStoreSpec(cpu_blocks), dst_spec)
    worker.wait({1})

    if current_platform.is_cuda():
        assert [num_descs for num_descs, _ in calls] == [5, 5, 2]
        assert all(not order_any for _, order_any in calls)

    block_map = dict(zip(gpu_blocks, cpu_blocks))
    for src_tensor, dst_tensor, orig_dst in zip(
        handler.src_tensors, handler.dst_tensors, orig_dst_tensors
    ):
        for gpu_block in range(num_gpu_blocks):
            cpu_block = block_map.get(gpu_block)
            expected = (
                src_tensor[cpu_block]
                if cpu_block is not None
                else orig_dst[gpu_block].cpu()
            )
            assert torch.equal(dst_tensor[gpu_block].cpu(), expected)

    worker.shutdown()


def _naive_descriptor_fill(
    group_sizes,
    block_indices,
    src_blocks,
    dst_blocks,
    groups_refs,
    src_tensors,
    dst_tensors,
    src_factor,
    dst_factor,
):
    """Reference fill: one pure-python iteration per copy op."""

    def sub_block_ptr(tensor, blocks, factor, logical_idx):
        block_id = int(blocks[logical_idx // factor])
        sub_idx = logical_idx % factor
        sub_size = tensor.shape[1] // factor
        return tensor.data_ptr() + block_id * tensor.stride(0) + sub_idx * sub_size

    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    sizes: list[int] = []
    src_off = 0
    dst_off = 0
    total_bytes = 0
    for group_size, block_idx, refs in zip(group_sizes, block_indices, groups_refs):
        if group_size == 0:
            continue
        src_skip = block_idx % src_factor
        dst_skip = block_idx % dst_factor
        src_cnt = cdiv(group_size + src_skip, src_factor)
        dst_cnt = cdiv(group_size + dst_skip, dst_factor)
        group_src = src_blocks[src_off : src_off + src_cnt]
        group_dst = dst_blocks[dst_off : dst_off + dst_cnt]
        for ref in refs:
            for i in range(group_size):
                src_ptrs.append(
                    sub_block_ptr(
                        src_tensors[ref.tensor_idx], group_src, src_factor, src_skip + i
                    )
                )
                dst_ptrs.append(
                    sub_block_ptr(
                        dst_tensors[ref.tensor_idx], group_dst, dst_factor, dst_skip + i
                    )
                )
                sizes.append(ref.page_size_bytes)
                total_bytes += ref.page_size_bytes
        src_off += src_cnt
        dst_off += dst_cnt
    return src_ptrs, dst_ptrs, sizes, total_bytes


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("block_size_factor", [1, 3])
@pytest.mark.parametrize("uniform_layout", [True, False])
def test_descriptor_builder_matches_naive_reference(
    gpu_to_cpu: bool, block_size_factor: int, uniform_layout: bool
) -> None:
    """The vectorized descriptor fill must match a per-sub-block reference
    loop, including partial first CPU blocks (skip), zero-size groups,
    non-contiguous CPU tensors, tensors shared between refs, and the
    non-uniform-layout fallback (uniform_layout=False)."""
    rng = random.Random(0)
    gpu_page = 32 * block_size_factor
    cpu_page = gpu_page * block_size_factor

    group_sizes = [5, 0, 9]
    block_indices = [4, 0, 5]
    refs_per_group = [3, 2, 4]

    gpu_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []
    groups_refs: list[list[CanonicalKVCacheRef]] = []
    for num_refs in refs_per_group:
        refs: list[CanonicalKVCacheRef] = []
        for j in range(num_refs):
            if j == 1 and len(groups_refs) == 2:
                # two refs sharing one tensor, with a smaller unpadded page
                refs.append(
                    CanonicalKVCacheRef(
                        tensor_idx=refs[0].tensor_idx,
                        page_size_bytes=gpu_page - 16,
                    )
                )
                continue
            pad = 96 if not uniform_layout and not groups_refs and j == 0 else 32
            gpu_tensors.append(torch.zeros((64, gpu_page), dtype=torch.int8))
            cpu_tensors.append(
                torch.zeros((32, cpu_page + pad), dtype=torch.int8)[:, :cpu_page]
            )
            refs.append(
                CanonicalKVCacheRef(
                    tensor_idx=len(gpu_tensors) - 1, page_size_bytes=gpu_page
                )
            )
        groups_refs.append(refs)

    src_factor = 1 if gpu_to_cpu else block_size_factor
    dst_factor = block_size_factor if gpu_to_cpu else 1
    src_tensors = gpu_tensors if gpu_to_cpu else cpu_tensors
    dst_tensors = cpu_tensors if gpu_to_cpu else gpu_tensors

    src_blocks_list: list[int] = []
    dst_blocks_list: list[int] = []
    for group_size, block_idx in zip(group_sizes, block_indices):
        if group_size == 0:
            continue
        src_cnt = cdiv(group_size + block_idx % src_factor, src_factor)
        dst_cnt = cdiv(group_size + block_idx % dst_factor, dst_factor)
        src_blocks_list += [rng.randrange(24) for _ in range(src_cnt)]
        dst_blocks_list += [rng.randrange(24) for _ in range(dst_cnt)]
    src_blocks = np.array(src_blocks_list, dtype=np.int64)
    dst_blocks = np.array(dst_blocks_list, dtype=np.int64)

    builder = _DescriptorBuilder(
        src_tensors=src_tensors,
        dst_tensors=dst_tensors,
        src_block_size_factor=src_factor,
        dst_block_size_factor=dst_factor,
        kv_cache_groups_data_refs=groups_refs,
    )
    ref_src, ref_dst, ref_sizes, ref_bytes = _naive_descriptor_fill(
        group_sizes,
        block_indices,
        src_blocks,
        dst_blocks,
        groups_refs,
        src_tensors,
        dst_tensors,
        src_factor,
        dst_factor,
    )

    num_copy_ops = builder.num_copy_ops(group_sizes)
    assert num_copy_ops == len(ref_sizes)

    all_src = torch.empty(num_copy_ops, dtype=torch.int64).numpy()
    all_dst = torch.empty(num_copy_ops, dtype=torch.int64).numpy()
    all_sizes = torch.empty(num_copy_ops, dtype=torch.int64).numpy()
    num_bytes = builder.fill(
        group_sizes,
        block_indices,
        src_blocks,
        dst_blocks,
        all_src,
        all_dst,
        all_sizes,
    )

    assert num_bytes == ref_bytes
    assert all_src.tolist() == ref_src
    assert all_dst.tolist() == ref_dst
    assert all_sizes.tolist() == ref_sizes
