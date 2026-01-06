import argparse
import os
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
    # {
    #     'name': 'baseline',
    #     'num_tokens': 8,
    #     'hidden': 2560,
    #     'num_topk': 4,
    #     'num_experts': 16,
    #     'seed': 0,
    #     'log_values': False,
    #     'num_processes': 2,
    # },
    # {
    #     'name': 'setting_0',
    #     'num_tokens': 64,
    #     'hidden': 4096,
    #     'num_topk': 8,
    #     'num_experts': 128,
    #     'seed': 29,
    #     'log_values': False,
    #     'num_processes': 4,
    # },
    # {
    #     'name': 'setting_1',
    #     'num_tokens': 64,
    #     'hidden': 4096,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 31,
    #     'log_values': False,
    #     'num_processes': 8,
    # },
    {
        'name': 'setting_2_0',
        'num_tokens': 32,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 288,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
    },
    {
        'name': 'setting_2_1',
        'num_tokens': 64,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 288,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
    },
    {
        'name': 'setting_2_2',
        'num_tokens': 128,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 288,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
    },
]


def warn_allclose(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-2, rank: Optional[int] = None, *, log_values: bool = True) -> bool:
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


def _lex_argsort(matrix: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(matrix.size(0), device=matrix.device)
    for col in range(matrix.size(1) - 1, -1, -1):
        col_vals = matrix[idx, col]
        perm = torch.argsort(col_vals, stable=True)
        idx = idx[perm]
    return idx


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
    num_nodes = int(os.getenv('WORLD_SIZE', 2))

    if rank == 0:
        print(f"[info] running setting '{setting['name']}' with num_experts={num_experts}, num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, num_nodes = {num_nodes}, num_ranks = {num_ranks} ", flush=True)
        print(f"[info] group.rank()={group.rank()} , group.size()={group.size()} ", flush=True)


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
                                  num_qps_per_rank=num_experts // num_ranks)

    torch.manual_seed(setting.get('seed', 0))
    torch.cuda.manual_seed_all(setting.get('seed', 0))

    device = torch.device('cuda', torch.cuda.current_device())
    
    # Data generation

    row_values = torch.arange(num_tokens, dtype=torch.float32, device=device)
    row_values = (row_values + rank * num_tokens) * 0.1
    x = row_values.unsqueeze(1).expand(num_tokens, hidden).to(torch.bfloat16)
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x = x_pure_rand
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') 
    # topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    # print(f"[debug] rank {rank} x shape={tuple(x.shape)} topk_idx shape={tuple(topk_idx.shape)} topk_weights shape={tuple(topk_weights.shape)}",
    #         flush=True)

    use_fp8 = False

    def clone_low_latency_inputs():
        return {
            'x': x.clone(),
            'topk_idx': topk_idx.clone(),
            'topk_weights': topk_weights.clone(),
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'use_fp8': use_fp8,
            'async_finish': False,
        }

    def bench_once(buffer):
        inputs = clone_low_latency_inputs()
        torch.cuda.synchronize()
        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        combine_start = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)

        dispatch_start.record()
        dispatch_kwargs = dict(
            x=inputs['x'],
            topk_idx=inputs['topk_idx'],
            num_max_dispatch_tokens_per_rank=inputs['num_tokens'],
            num_experts=inputs['num_experts'],
            use_fp8=inputs['use_fp8'],
            async_finish=inputs['async_finish'],
        )
        if isinstance(buffer, mori.Buffer):
            dispatch_kwargs['topk_weights'] = inputs['topk_weights']
        packed_recv_x, packed_recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(**dispatch_kwargs)
        dispatch_end.record()
        torch.cuda.synchronize()
        combine_start.record()
        buffer.low_latency_combine(packed_recv_x, inputs['topk_idx'], inputs['topk_weights'], handle,
                                   async_finish=False)
        combine_end.record()
        torch.cuda.synchronize()
        return dispatch_start.elapsed_time(dispatch_end), combine_start.elapsed_time(combine_end)

    def benchmark_low_latency(name: str, buffer, *, num_warmups: int = 1, num_iters: int = 5):
        if buffer is None:
            return None
        for _ in range(num_warmups):
            bench_once(buffer)
        times = [bench_once(buffer) for _ in range(num_iters)]
        dispatch_times = [t[0] for t in times]
        combine_times = [t[1] for t in times]
        stats = {
            'dispatch_avg_ms': sum(dispatch_times) / len(dispatch_times),
            'dispatch_min_ms': min(dispatch_times),
            'dispatch_max_ms': max(dispatch_times),
            'combine_avg_ms': sum(combine_times) / len(combine_times),
            'combine_min_ms': min(combine_times),
            'combine_max_ms': max(combine_times),
        }
        if rank == 0:
            print(f"[perf] {name} dispatch avg={stats['dispatch_avg_ms']:.3f} ms (min={stats['dispatch_min_ms']:.3f}, max={stats['dispatch_max_ms']:.3f})", flush=True)
            print(f"[perf] {name} combine  avg={stats['combine_avg_ms']:.3f} ms (min={stats['combine_min_ms']:.3f}, max={stats['combine_max_ms']:.3f})", flush=True)
        return stats

    # Low Latency Dispatch
    # DeepEP
    deep_packed_recv_x = deep_packed_recv_count = deep_handle = None
    if run_deep:
        deep_packed_recv_x, deep_packed_recv_count, deep_handle, deep_event, deep_hook = \
            buffer_deep.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                             use_fp8=use_fp8, async_finish=False)
        deep_src_info = deep_handle[0]
        if rank == 0:
            print(f"[debug] DeepEP low-latency dispatch src_info shape: {deep_src_info.shape}", flush=True)
            # print(f"[debug] DeepEP low-latency dispatch src_info: {deep_src_info.cpu()}", flush=True)

    
    # Mori
    mori_packed_recv_x = mori_packed_recv_count = mori_handle = None
    if run_mori:
        mori_packed_recv_x, mori_packed_recv_count, mori_handle, mori_event, mori_hook = \
            buffer_mori.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                             use_fp8=use_fp8, async_finish=False, topk_weights=topk_weights)
                                            
        mori_src_info = deep_handle[0]
        if rank == 0:
            print(f"[debug] MORI low-latency dispatch src_info shape: {mori_src_info.shape}", flush=True)
            # print(f"[debug] MORI low-latency dispatch src_info: {mori_src_info.cpu()}", flush=True)

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
                deep_idx = _lex_argsort(deep_data)
                mori_idx = _lex_argsort(mori_data)
                
                deep_data_sorted = deep_data[deep_idx]
                mori_data_sorted = mori_data[mori_idx]
                
                if not torch.allclose(deep_data_sorted, mori_data_sorted, atol=1e-2, rtol=1e-2):
                    if rank == 0:
                        print(f"[warning] recv_x mismatch at expert {i}", flush=True)
                        diff = (deep_data_sorted - mori_data_sorted).abs().max()
                        print(f"  max diff: {diff}", flush=True)
                        
                        print('  deep_ep recv_x:', deep_data_sorted.cpu(), flush=True)
                        print('  mori   recv_x:', mori_data_sorted.cpu(), flush=True)
                    mismatch = True
                else:
                    if rank == 0:
                        if log_values:
                            print(f"[debug] recv_x expert {i} match.", flush=True)
                            print('  deep_ep recv_x:', deep_data_sorted.cpu(), flush=True)
                            print('  mori   recv_x:', mori_data_sorted.cpu(), flush=True)
                
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
    deep_perf = benchmark_low_latency('DeepEP', buffer_deep, num_warmups=5, num_iters=50)
    mori_perf = benchmark_low_latency('MORI', buffer_mori, num_warmups=5, num_iters=50)
    dist.barrier()
    if rank == 0 and deep_perf and mori_perf:
        dispatch_ratio = mori_perf['dispatch_avg_ms'] / max(deep_perf['dispatch_avg_ms'], 1e-6)
        combine_ratio = mori_perf['combine_avg_ms'] / max(deep_perf['combine_avg_ms'], 1e-6)
        print(f"[perf] MORI/DeepEP dispatch avg ratio: {dispatch_ratio:.3f}x", flush=True)
        print(f"[perf] MORI/DeepEP combine  avg ratio: {combine_ratio:.3f}x", flush=True)

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
    # python  tests/test_compare_buffers_low_latency.py --path both

    for setting in PRESET_SETTINGS:
        num_processes = setting.get('num_processes', 2)
        print('-------------------------------------------------------------------------', flush=True)
        print(f"[info] spawning comparison for setting '{setting['name']}' (num_processes={num_processes})", flush=True)
        
        mp.spawn(compare_buffers, args=(num_processes, setting, args.path), nprocs=num_processes)
        print('*************************************************************************', flush=True)


if __name__ == '__main__':
    main()
