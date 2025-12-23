import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

import mori
import deep_ep
from utils import (calc_diff, init_dist, inplace_unique, per_token_cast_back)


NUM_SMs = 8
        # case 2560: case_macro(2560); \
        # case 5120: case_macro(5120); \
        # case 4096: case_macro(4096); \
        # case 7168: case_macro(7168); \
PRESET_SETTINGS = [
    {
        'name': 'baseline',
        'num_tokens': 8,
        'hidden': 2560,
        'num_topk': 4,
        'num_experts': 16,
        'seed': 0,
        'log_values': False,
        'num_processes': 2,
    },
    # {
    #     'name': 'baseline_2',
    #     'num_tokens': 16,
    #     'hidden': 8,
    #     'num_topk': 8,
    #     'num_experts': 32,
    #     'seed': 0,
    #     'log_values': False,
    #     'num_processes': 2,
    # },
    # {
    #     'name': 'setting_0',
    #     'num_tokens': 16,
    #     'hidden': 256,
    #     'num_topk': 8,
    #     'num_experts': 32,
    #     'seed': 17,
    #     'log_values': False,
    #     'num_processes': 2,
    # },
    # {
    #     'name': 'setting_1',
    #     'num_tokens': 128,
    #     'hidden': 4096,
    #     'num_topk': 8,
    #     'num_experts': 64,
    #     'seed': 17,
    #     'log_values': False,
    #     'num_processes': 4,
    # },
    # {
    #     'name': 'setting_2',
    #     'num_tokens': 128,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 42,
    #     'log_values': False,
    #     'num_processes': 8,
    # },
    # {
    #     'name': 'setting_3',
    #     'num_tokens': 2048,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 47,
    #     'log_values': False,
    #     'num_processes': 8,
    # },
]


def warn_allclose(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5, rank: Optional[int] = None, *, log_values: bool = True) -> bool:
    same = torch.allclose(a, b, rtol=rtol, atol=atol)
    if rank is None or rank == 0:
        if not same:
            diff = torch.abs(a - b)
            max_diff = torch.max(diff)
            print(f'[warning] {name} mismatch: max diff {max_diff:.6e}', flush=True)
        else:
            print(f'[debug] {name} match.', flush=True)
        print(f'[info] {name} tensor deep_ep shape {tuple(a.shape)}', flush=True)
        print(f'[info] {name} tensor mori shape {tuple(b.shape)}', flush=True)
        if log_values:
            print(a.cpu(), flush=True)
            print(b.cpu(), flush=True)
        else:
            print(f'[info] {name} tensor values suppressed (log_values False).', flush=True)
    return same


def _round_up_num_experts(base: int, num_ranks: int) -> int:
    per_rank = max((base + num_ranks - 1) // num_ranks, 1)
    return per_rank * num_ranks


def compare_buffers(local_rank: int, num_local_ranks: int, setting: dict, run_path: str):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, backend='gloo')
    num_experts = _round_up_num_experts(setting['num_experts'], num_ranks)
    num_tokens = setting['num_tokens']
    hidden = setting['hidden']
    num_topk = setting['num_topk']
    log_values = setting.get('log_values', True)
    run_deep = run_path in ('deep', 'both')
    run_mori = run_path in ('mori', 'both')

    if rank == 0:
        print(f"[info] running setting '{setting['name']}' with num_experts={num_experts}, num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}", flush=True)


    buffer_deep = None
    if run_deep:
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
        if local_rank == 0:
            print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
        buffer_deep = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                                     num_qps_per_rank=num_experts // num_ranks)

    buffer_mori = None
    if run_mori:
        buffer_mori = mori.Buffer(group, int(1e9), int(1e9), low_latency_mode=True,
                                  num_qps_per_rank=num_experts // num_ranks,
                                  max_num_inp_token_per_rank=num_tokens,
                                  num_experts_per_token=num_topk,
                                  gpu_per_node=num_local_ranks)

    torch.manual_seed(setting.get('seed', 0))
    torch.cuda.manual_seed_all(setting.get('seed', 0))

    device = torch.device('cuda', torch.cuda.current_device())
    
    # Data generation
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    print(f"[debug] rank {rank} x shape={tuple(x.shape)} topk_idx shape={tuple(topk_idx.shape)} topk_weights shape={tuple(topk_weights.shape)}",
            flush=True)

    # Low Latency Dispatch
    use_fp8 = False
    
    # DeepEP
    deep_packed_recv_x = deep_packed_recv_count = deep_handle = None
    if run_deep:
        deep_packed_recv_x, deep_packed_recv_count, deep_handle, deep_event, deep_hook = \
            buffer_deep.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                             use_fp8=use_fp8, async_finish=False)
    
    # Mori
    mori_packed_recv_x = mori_packed_recv_count = mori_handle = None
    if run_mori:
        mori_packed_recv_x, mori_packed_recv_count, mori_handle, mori_event, mori_hook = \
            buffer_mori.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                             use_fp8=use_fp8, async_finish=False, topk_weights=topk_weights)

    mismatch = False
    
    if run_deep and run_mori:
        # Compare counts
        mismatch |= not warn_allclose('packed_recv_count', deep_packed_recv_count, mori_packed_recv_count, rank=rank, log_values=log_values)

        # Compare received data (sorted)
        num_local_experts = deep_packed_recv_count.size(0)
        for i in range(num_local_experts):
            count = deep_packed_recv_count[i].item()
            if count > 0:
                deep_data = deep_packed_recv_x[i, :count]
                mori_data = mori_packed_recv_x[i, :count]
                
                # Sort for comparison
                deep_idx = torch.argsort(deep_data[:, 0])
                mori_idx = torch.argsort(mori_data[:, 0])
                
                deep_data_sorted = deep_data[deep_idx]
                mori_data_sorted = mori_data[mori_idx]
                
                if not torch.allclose(deep_data_sorted, mori_data_sorted, atol=1e-5):
                    if rank == 0:
                        print(f"[warning] recv_x mismatch at expert {i}", flush=True)
                        diff = (deep_data_sorted - mori_data_sorted).abs().max()
                        print(f"  max diff: {diff}", flush=True)
                    mismatch = True
    elif rank == 0:
        print(f"[info] skipping cross-buffer dispatch comparison (path={run_path}).", flush=True)

    # Low Latency Combine
    # Use identity as GEMM
    deep_combined_x = None
    if run_deep:
        deep_sim_gemm_x = deep_packed_recv_x.clone()
        deep_combined_x, deep_combine_event, deep_combine_hook = \
            buffer_deep.low_latency_combine(deep_sim_gemm_x, topk_idx, topk_weights, deep_handle, async_finish=False)
    
    mori_combined_x = None
    if run_mori:
        mori_sim_gemm_x = mori_packed_recv_x.clone()
        mori_combined_x, mori_combine_event, mori_combine_hook = \
            buffer_mori.low_latency_combine(mori_sim_gemm_x, topk_idx, topk_weights, mori_handle, async_finish=False)
        
    if run_deep and run_mori:
        mismatch |= not warn_allclose('combined_x', deep_combined_x, mori_combined_x, rank=rank, log_values=log_values)
    elif rank == 0:
        print(f"[info] skipping cross-buffer combine comparison (path={run_path}).", flush=True)

    dist.barrier()
    if rank == 0:
        if run_deep and run_mori:
            if mismatch:
                print('[warning] DeepEP and MORI buffers had mismatches during comparison.', flush=True)
            else:
                print('DeepEP and MORI buffers dispatch/combine outputs match across ranks.', flush=True)
        else:
            print(f"[info] completed run_path={run_path} without cross-buffer comparison.", flush=True)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Compare DeepEP and MORI low-latency buffers.')
    parser.add_argument('--path', choices=('deep', 'mori', 'both'), default='both',
                        help='Select which buffer path(s) to run: deep-only, mori-only, or both (default).')
    args = parser.parse_args()

    for setting in PRESET_SETTINGS:
        num_processes = setting.get('num_processes', 2)
        print('-------------------------------------------------------------------------', flush=True)
        print(f"[info] spawning comparison for setting '{setting['name']}' (num_processes={num_processes})", flush=True)
        
        mp.spawn(compare_buffers, args=(num_processes, setting, args.path), nprocs=num_processes)
        print('*************************************************************************', flush=True)


if __name__ == '__main__':
    main()
