import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

import mori
import deep_ep
from utils import (calc_diff, init_dist, inplace_unique, per_token_cast_back)


PRESET_SETTINGS = [
    # {
    #     'name': 'setting_1',
    #     'num_tokens': 64,
    #     'hidden': 4096,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 31,
    #     'log_values': False,
    #     'num_processes': 8,
    #     'use_fp8' : True,
    # },
    # {
    #     'name': 'setting_2_0',
    #     'num_tokens': 32,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 288,
    #     'seed': 42,
    #     'log_values': False,
    #     'num_processes': 8,
    # },
    # {
    #     'name': 'setting_2_1',
    #     'num_tokens': 64,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 288,
    #     'seed': 42,
    #     'log_values': False,
    #     'num_processes': 8,
    # },
    {
        'name': 'setting_2_2',
        'num_tokens': 128,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 288,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
        'use_fp8' : False,
    },
    {
        'name': 'setting_2_3',
        'num_tokens': 128,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 288,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
        'use_fp8' : True,
        'use_gpu_ll_layout_transform' : False,
    },
    {
        'name': 'setting_2_4',
        'num_tokens': 128,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 288,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
        'use_fp8' : True,
        'use_gpu_ll_layout_transform' : True,
        'enable_' : True,
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


def _get_global_stats(val: float, group) -> dict:
    t_avg = torch.tensor(val, device='cuda', dtype=torch.float32)
    dist.all_reduce(t_avg, op=dist.ReduceOp.SUM, group=group)
    avg = t_avg.item() / dist.get_world_size(group)
    
    t_min = torch.tensor(val, device='cuda', dtype=torch.float32)
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN, group=group)
    mn = t_min.item()
    
    t_max = torch.tensor(val, device='cuda', dtype=torch.float32)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=group)
    mx = t_max.item()
    
    return {'avg': avg, 'min': mn, 'max': mx}


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
    use_fp8 = setting.get('use_fp8', False)
    use_gpu_ll_layout_transform = setting.get('use_gpu_ll_layout_transform', True)
    enable_mori_profiling = setting.get('profiling', False)

    if rank == 0:
        print(f"[info] running setting '{setting['name']}' with num_experts={num_experts}, num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, num_nodes = {num_nodes}, num_ranks = {num_ranks}, use_fp8 = {use_fp8}, use_gpu_ll_layout_transform = {use_gpu_ll_layout_transform} ", flush=True)
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
                                  num_qps_per_rank=num_experts // num_ranks, 
                                  use_gpu_ll_layout_transform = use_gpu_ll_layout_transform,
                                  enable_profiling = enable_mori_profiling,
                                  )

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

    num_selections = (topk_idx != -1).sum().item()
    
    if use_fp8:
        # FP8 config from deep_ep: data + scale + ...
        # (hidden + hidden / 128 * 4 + 16)
        bytes_per_token_dispatch = (hidden + hidden / 128 * 4 + 16)
    else:
        bytes_per_token_dispatch = hidden * 2
        
    bytes_per_token_combine = hidden * 2
    
    num_dispatch_comm_bytes = num_selections * bytes_per_token_dispatch
    num_combine_comm_bytes = num_selections * bytes_per_token_combine

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
        
        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        combine_start = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)

        
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
        torch.cuda.synchronize()
        dispatch_start.record()
        packed_recv_x, packed_recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(**dispatch_kwargs)
        torch.cuda.synchronize()
        dist.barrier()
        dispatch_end.record()
        combine_start.record()
        buffer.low_latency_combine(packed_recv_x, inputs['topk_idx'], inputs['topk_weights'], handle,
                                   async_finish=False)
        torch.cuda.synchronize()
        combine_end.record()
        return dispatch_start.elapsed_time(dispatch_end), combine_start.elapsed_time(combine_end)

    def benchmark_low_latency(name: str, buffer, *, num_warmups: int = 1, num_iters: int = 5, dispatch_bytes: int = 0, combine_bytes: int = 0):
        if buffer is None:
            return None
        for _ in range(num_warmups):
            bench_once(buffer)
        times = [bench_once(buffer) for _ in range(num_iters)]
        
        # NOTE: times is a list of (dispatch_ms, combine_ms)
        dispatch_times = [t[0] for t in times]
        combine_times = [t[1] for t in times]
        
        local_dispatch_avg_ms = sum(dispatch_times) / len(dispatch_times)
        local_combine_avg_ms = sum(combine_times) / len(combine_times)
        
        local_dispatch_bw = dispatch_bytes / 1e9 / (local_dispatch_avg_ms / 1000) if local_dispatch_avg_ms > 0 else 0
        local_combine_bw = combine_bytes / 1e9 / (local_combine_avg_ms / 1000) if local_combine_avg_ms > 0 else 0
        
        stats = {
            'dispatch_ms': _get_global_stats(local_dispatch_avg_ms, group),
            'combine_ms': _get_global_stats(local_combine_avg_ms, group),
            'dispatch_bw': _get_global_stats(local_dispatch_bw, group),
            'combine_bw': _get_global_stats(local_combine_bw, group),
        }

        if rank == 0:
            def format_metric(m):
                return f"{m['avg']:.3f} (min={m['min']:.3f}, max={m['max']:.3f})"
                
            print(f"[perf] {name} dispatch time={format_metric(stats['dispatch_ms'])} ms | BW={format_metric(stats['dispatch_bw'])} GB/s", flush=True)
            print(f"[perf] {name} combine  time={format_metric(stats['combine_ms'])} ms | BW={format_metric(stats['combine_bw'])} GB/s", flush=True)
            
        return stats

    # Low Latency Dispatch
    # DeepEP
    deep_packed_recv_x = deep_packed_recv_count = deep_handle = None
    if run_deep:
        deep_packed_recv_x, deep_packed_recv_count, deep_handle, deep_event, deep_hook = \
            buffer_deep.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                             use_fp8=use_fp8, async_finish=False)
        deep_src_info = deep_handle[0]
        # deep_packed_recv_x = (deep_packed_recv_x[0], deep_packed_recv_x[1].contiguous()) if use_fp8 else deep_packed_recv_x
        deep_packed_recv_x = per_token_cast_back(deep_packed_recv_x[0].view(-1, hidden), deep_packed_recv_x[1].view(-1, hidden // 128)).view(deep_packed_recv_x[0].shape) \
                if use_fp8 else deep_packed_recv_x.clone()
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
        # mori_packed_recv_x = (mori_packed_recv_x[0], mori_packed_recv_x[1].contiguous()) if use_fp8 else mori_packed_recv_x
        mori_packed_recv_x = per_token_cast_back(mori_packed_recv_x[0].view(-1, hidden), mori_packed_recv_x[1].view(-1, hidden // 128)).view(mori_packed_recv_x[0].shape) \
                if use_fp8 else mori_packed_recv_x.clone()
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
                
            if count > 0:
                deep_data = deep_src_info[i, :count]
                mori_data = mori_src_info[i, :count]
                deep_data_sorted = torch.argsort(deep_data, stable=True)
                mori_data_sorted = torch.argsort(mori_data, stable=True)
                
                if not torch.allclose(deep_data_sorted, mori_data_sorted, atol=1e-2, rtol=1e-2):
                    if rank == 0:
                        print(f"[warning] src_info mismatch at expert {i}", flush=True)
                        diff = (deep_data_sorted - mori_data_sorted).abs().max()
                        print(f"  max diff: {diff}", flush=True)
                        
                        print('  deep_ep src_info:', deep_data_sorted.cpu(), flush=True)
                        print('  mori   src_info:', mori_data_sorted.cpu(), flush=True)
                    mismatch = True
                else:
                    if rank == 0:
                        if log_values:
                            print(f"[debug] src_info expert {i} match.", flush=True)
                            print('  deep_ep src_info:', deep_data_sorted.cpu(), flush=True)
                            print('  mori   src_info:', mori_data_sorted.cpu(), flush=True)
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
    
    if rank == 0 and run_mori and enable_mori_profiling:
        print('[info] MORI profiling breakdown (after single run):', flush=True)
        dispatch_stats = buffer_mori.get_profiling_breakdown_low_latency_dispatch()
        combine_stats = buffer_mori.get_profiling_breakdown_low_latency_combine()
        print('--- Dispatch ---', flush=True)
        print(f'Pre-process (ms) = {dispatch_stats["average"]["pre"]} | Core (ms) {dispatch_stats["average"]["core"]} | GPU Core (ms) = {dispatch_stats["average"]["gpu_core"]} | Post-process (ms) = {dispatch_stats["average"]["post"]} ', flush=True)
        print('--- Combine ---', flush=True)
        print(f'Pre-process (ms) = {combine_stats["average"]["pre"]} | Core (ms) {combine_stats["average"]["core"]} | GPU Core (ms) = {combine_stats["average"]["gpu_core"]} | Post-process (ms) = {combine_stats["average"]["post"]} ', flush=True)

    dist.barrier()
    buffer_deep.reset_profiling_data()
    deep_perf = benchmark_low_latency('DeepEP', buffer_deep, num_warmups=5, num_iters=50, 
                                      dispatch_bytes=num_dispatch_comm_bytes, combine_bytes=num_combine_comm_bytes)
    mori_perf = benchmark_low_latency('MORI', buffer_mori, num_warmups=5, num_iters=50,
                                      dispatch_bytes=num_dispatch_comm_bytes, combine_bytes=num_combine_comm_bytes)
    dist.barrier()
    if rank == 0 and deep_perf and mori_perf:
        dispatch_ratio = mori_perf['dispatch_ms']['avg'] / deep_perf['dispatch_ms']['avg'] if deep_perf['dispatch_ms']['avg'] != 0 else float('inf')
        combine_ratio = mori_perf['combine_ms']['avg'] / deep_perf['combine_ms']['avg'] if deep_perf['combine_ms']['avg'] != 0 else float('inf')
        print(f"[perf] MORI/DeepEP dispatch avg ratio: {dispatch_ratio:.3f}x", flush=True)
        print(f"[perf] MORI/DeepEP combine  avg ratio: {combine_ratio:.3f}x", flush=True)
    
    if rank == 0 and mori_perf and enable_mori_profiling:
        print('[info] MORI profiling breakdown:', flush=True)
        dispatch_stats = buffer_mori.get_profiling_breakdown_low_latency_dispatch()
        combine_stats = buffer_mori.get_profiling_breakdown_low_latency_combine()
        print('--- Dispatch ---', flush=True)
        print(f'Pre-process (ms) = {dispatch_stats["average"]["pre"]} | Core (ms) {dispatch_stats["average"]["core"]} | GPU Core (ms) = {dispatch_stats["average"]["gpu_core"]} | Post-process (ms) = {dispatch_stats["average"]["post"]} ', flush=True)
        print('--- Combine ---', flush=True)
        print(f'Pre-process (ms) = {combine_stats["average"]["pre"]} | Core (ms) {combine_stats["average"]["core"]} | GPU Core (ms) = {combine_stats["average"]["gpu_core"]} | Post-process (ms) = {combine_stats["average"]["post"]} ', flush=True)

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
    parser.add_argument('--profiling', action='store_true', default=False,
                        help='Enable profiling mode (not used in this test).')
    args = parser.parse_args()
    # python  tests/test_compare_buffers_low_latency.py --path both

    for setting in PRESET_SETTINGS:
        num_processes = setting.get('num_processes', 2)
        setting['profiling'] = args.profiling
        print('-------------------------------------------------------------------------', flush=True)
        print(f"[info] spawning comparison for setting '{setting['name']}' (num_processes={num_processes})", flush=True)
        
        mp.spawn(compare_buffers, args=(num_processes, setting, args.path), nprocs=num_processes)
        print('*************************************************************************', flush=True)


if __name__ == '__main__':
    main()
