import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

import mori
import deep_ep
from utils import init_dist, inplace_unique, per_token_cast_to_fp8, per_token_cast_back


NUM_SMs = 8

PRESET_SETTINGS = [
    {
        'name': 'baseline',
        'num_tokens': 8,
        'hidden': 8,
        'num_topk': 4,
        'num_experts': 16,
        'seed': 0,
        'log_values': True,
        'num_processes': 2,
    },
    {
        'name': 'baseline_2',
        'num_tokens': 16,
        'hidden': 8,
        'num_topk': 8,
        'num_experts': 32,
        'seed': 0,
        'log_values': False,
        'num_processes': 2,
    },
    {
        'name': 'setting_0',
        'num_tokens': 16,
        'hidden': 256,
        'num_topk': 8,
        'num_experts': 32,
        'seed': 17,
        'log_values': False,
        'num_processes': 2,
    },
    {
        'name': 'setting_1',
        'num_tokens': 128,
        'hidden': 4096,
        'num_topk': 8,
        'num_experts': 64,
        'seed': 17,
        'log_values': False,
        'num_processes': 4,
    },
    {
        'name': 'setting_2',
        'num_tokens': 128,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 256,
        'seed': 42,
        'log_values': False,
        'num_processes': 8,
    },
    {
        'name': 'setting_3',
        'num_tokens': 2048,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 256,
        'seed': 47,
        'log_values': False,
        'num_processes': 8,
    },
    {
        'name': 'setting_4',
        'num_tokens': 4096,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 256,
        'seed': 49,
        'log_values': False,
        'num_processes': 8,
    },
]


def compute_dispatch_meta(topk_idx: torch.Tensor, num_experts: int, num_ranks: int, num_tokens: int):
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device=topk_idx.device)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device=topk_idx.device)
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device=topk_idx.device)
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device=topk_idx.device)
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    rank_to_tokens = []
    for rank in range(num_ranks):
        slot_idx = token_idx_in_rank[:, rank]
        mask = slot_idx >= 0
        if not mask.any():
            continue
        tokens = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        ordered = tokens[torch.argsort(slot_idx[mask])]
        rank_to_tokens.append(ordered)
    mori_token_order = torch.cat(rank_to_tokens, dim=0) if rank_to_tokens else torch.empty((0,), dtype=torch.long, device=topk_idx.device)

    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, mori_token_order


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


def reorder_mori_outputs(recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                         token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_outputs guard: shape mismatch or empty order.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    if token_order.min() < 0 :
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_outputs guard: order contains invalid indices.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    unique_tokens = torch.unique(token_order)
    if unique_tokens.numel() != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_outputs guard: order contains repeated tokens.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    perm = torch.argsort(token_order)
    return recv_x[perm], recv_topk_idx[perm], recv_topk_weights[perm]


def revert_mori_outputs(recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                         token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] revert_mori_outputs guard: shape mismatch or empty order.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    if token_order.min() < 0 :
        if dist.get_rank() == 0:
            print('[warning] revert_mori_outputs guard: order contains invalid indices.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    unique_tokens = torch.unique(token_order)
    if unique_tokens.numel() != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] revert_mori_outputs guard: order contains repeated tokens.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    perm = torch.argsort(token_order)
    inverted = torch.empty_like(perm)
    inverted[perm] = torch.arange(perm.numel(), device=perm.device)
    return recv_x[inverted], recv_topk_idx[inverted], recv_topk_weights[inverted]

def reorder_mori_handle(handle: tuple[torch.Tensor, torch.Tensor], token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if token_order.numel() == 0 or handle[0].size(0) != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_handle guard: shape mismatch or empty order.', flush=True)
        return handle
    if token_order.min() < 0:
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_handle guard: order contains invalid indices.', flush=True)
        return handle
    unique_tokens = torch.unique(token_order)
    if unique_tokens.numel() != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_handle guard: order contains repeated tokens.', flush=True)
        return handle
    perm = torch.argsort(token_order)
    return handle[0][perm], handle[1]


def mask_mori_topk_by_rank(topk_idx: torch.Tensor, rank: int, num_experts: int, num_ranks: int) -> torch.Tensor:
    experts_per_rank = max(num_experts // num_ranks, 1)
    rank_start = rank * experts_per_rank
    rank_end = rank_start + experts_per_rank
    local_mask = (topk_idx >= rank_start) & (topk_idx < rank_end)
    masked = topk_idx.clone()
    masked[~local_mask] = -1
    return masked


def mask_mori_topk_weights_by_rank(topk_weights: torch.Tensor, topk_idx: torch.Tensor, rank: int, num_experts: int, num_ranks: int) -> torch.Tensor:
    experts_per_rank = max(num_experts // num_ranks, 1)
    rank_start = rank * experts_per_rank
    rank_end = rank_start + experts_per_rank
    local_mask = (topk_idx >= rank_start) & (topk_idx < rank_end)
    masked = topk_weights.clone()
    masked[~local_mask] = 0
    return masked

def _round_up_num_experts(base: int, num_ranks: int) -> int:
    per_rank = max((base + num_ranks - 1) // num_ranks, 1)
    return per_rank * num_ranks


def compare_buffers(local_rank: int, num_local_ranks: int, setting: dict):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, backend='gloo')
    num_experts = _round_up_num_experts(setting['num_experts'], num_ranks)
    num_tokens = setting['num_tokens']
    hidden = setting['hidden']
    num_topk = setting['num_topk']
    log_values = setting.get('log_values', True)

    if rank == 0:
        print(f"[info] running setting '{setting['name']}' with num_experts={num_experts}, num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}", flush=True)

    buffer_deep = deep_ep.Buffer(group, int(1e9), 0, low_latency_mode=False,
                                 num_qps_per_rank=num_experts // num_ranks)
    buffer_mori = mori.Buffer(group, int(1e9), int(1e9), low_latency_mode=False,
                              num_qps_per_rank=num_experts // num_ranks,
                              max_num_inp_token_per_rank=num_tokens,
                              num_experts_per_token=num_topk,
                              gpu_per_node=num_local_ranks)

    torch.manual_seed(setting.get('seed', 0))
    torch.cuda.manual_seed_all(setting.get('seed', 0))

    device = torch.device('cuda', torch.cuda.current_device())
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = deep_ep.Config(NUM_SMs, 8, nvl_buffer_size, 16, rdma_buffer_size)

    def make_dispatch_session():
        row_values = torch.arange(num_tokens, dtype=torch.float32, device=device)
        row_values = row_values + rank * num_tokens
        x = row_values.unsqueeze(1).expand(num_tokens, hidden).to(torch.bfloat16)
        x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        x = x_pure_rand
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
        topk_weights = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device='cuda')
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, mori_token_order = compute_dispatch_meta(
            topk_idx, num_experts, num_ranks, num_tokens)
        session = {
            'x': x,
            'num_tokens_per_rank': num_tokens_per_rank,
            'is_token_in_rank': is_token_in_rank,
            'num_tokens_per_expert': num_tokens_per_expert,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'config': config,
            'async_finish': False,
        }
        return session, mori_token_order

    base_dispatch_args, mori_token_order = make_dispatch_session()

    def clone_dispatch_args(args: dict):
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in args.items()}

    deep_output = buffer_deep.dispatch(**clone_dispatch_args(base_dispatch_args))
    mori_output = buffer_mori.dispatch(**clone_dispatch_args(base_dispatch_args))

    def normalize_result(result):
        recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, _ = result
        if isinstance(recv_x, tuple):
            recv_x = recv_x[0]
        return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle



    def bench_once(buffer):
        session, _ = make_dispatch_session()
        cloned_args = clone_dispatch_args(session)
        torch.cuda.synchronize()
        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        combine_start = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)

        dispatch_start.record()
        result = buffer.dispatch(**cloned_args)
        recv_x, _, recv_topk_weights, _, handle = normalize_result(result)
        dispatch_end.record()

        combine_start.record()
        buffer.combine(recv_x, handle, topk_weights=recv_topk_weights, config=session['config'])
        combine_end.record()
        torch.cuda.synchronize()
        dispatch_ms = dispatch_start.elapsed_time(dispatch_end)
        combine_ms = combine_start.elapsed_time(combine_end)
        return dispatch_ms, combine_ms

    def benchmark_buffer(name: str, buffer, *, num_warmups: int = 1, num_iters: int = 5):
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
            print(f"[perf] {name} dispatch avg={stats['dispatch_avg_ms']:.3f} ms (min={stats['dispatch_min_ms']:.3f} ms, max={stats['dispatch_max_ms']:.3f} ms)", flush=True)
            print(f"[perf] {name} combine  avg={stats['combine_avg_ms']:.3f} ms (min={stats['combine_min_ms']:.3f} ms, max={stats['combine_max_ms']:.3f} ms)", flush=True)
        return stats

    deep_recv_x, deep_topk_idx, deep_topk_weights, deep_num_list, deep_handle = normalize_result(deep_output)
    mori_recv_x, mori_topk_idx, mori_topk_weights, mori_num_list, mori_handle = normalize_result(mori_output)
    mori_recv_x_orig = mori_recv_x.clone()
    mori_topk_idx_orig = mori_topk_idx.clone()
    mori_topk_weights_orig = mori_topk_weights.clone()

    # mori_recv_x, mori_topk_idx, mori_topk_weights = reorder_mori_outputs(
    #     mori_recv_x, mori_topk_idx, mori_topk_weights, mori_handle[1])

    if rank== 0 and log_values:
        print('mori dispatch indices:', mori_handle[0].cpu(), flush=True)


    # if rank== 0 and log_values:
    #     print('reordered mori dispatch indices:', mori_handle[0].cpu(), flush=True)
    mismatch = False
    if rank== 0 and log_values:
        print('mori_token_order:', mori_token_order.cpu(), flush=True)

    if rank== 0 and log_values:
        print('mori src_token_pos:', mori_handle[1].cpu(), flush=True)


    if deep_num_list != mori_num_list:
        mismatch = True
        if rank == 0:
            print('[warning] num_tokens_per_expert_list mismatch', flush=True)
            if log_values:
                print('  deep_ep:', deep_num_list, flush=True)
                print('  mori  :', mori_num_list, flush=True)
    else:
        if rank == 0:
            print(f'[debug] rank {rank} num_tokens_per_expert_list match:', flush=True)
            if log_values:
                print('  deep_ep:', deep_num_list, flush=True)
                print('  mori  :', mori_num_list, flush=True)
    mori_topk_idx_filtered = mask_mori_topk_by_rank(mori_topk_idx, rank, num_experts, num_ranks)
    mori_topk_weights_filtered = mask_mori_topk_weights_by_rank(mori_topk_weights, mori_topk_idx_filtered, rank, num_experts, num_ranks)
    if not torch.equal(deep_topk_idx, mori_topk_idx_filtered):
        mismatch = True
        if rank == 0:
            print('[warning] topk indices mismatch', flush=True)
            if log_values:
                print('  deep_ep:', deep_topk_idx.cpu(), flush=True)
                print('  mori  :', mori_topk_idx_filtered.cpu(), flush=True)
    else:
        if rank == 0:
            print(f'[debug] rank {rank} topk indices match.', flush=True)
            if log_values:
                print('  deep_ep:', deep_topk_idx.cpu(), flush=True)
                print('  mori  :', mori_topk_idx.cpu(), flush=True)
    mismatch |= not warn_allclose('recv_x', deep_recv_x.float(), mori_recv_x.float(), rank=rank, log_values=log_values)
    mismatch |= not warn_allclose('recv_topk_weights', deep_topk_weights, mori_topk_weights_filtered, rank=rank, log_values=log_values)


    torch.cuda.synchronize()
    deep_combined_x, deep_combined_weights, _ = buffer_deep.combine(deep_recv_x, deep_handle,
                                                                    topk_weights=deep_topk_weights,
                                                                    config=config)
    mori_combined_x, mori_combined_weights, _ = buffer_mori.combine(mori_recv_x, mori_handle,
                                                                     topk_weights=mori_topk_weights,
                                                                     config=config)

    mismatch |= not warn_allclose('combined_x', deep_combined_x.float(), mori_combined_x.float(), rank=rank, log_values=log_values)
    # mismatch |= not warn_allclose('combined_topk_weights', deep_combined_weights, mori_combined_weights, rank=rank, log_values=log_values)


    dist.barrier()
    if rank == 0:
        if mismatch:
            print('[warning] DeepEP and MORI buffers had mismatches during comparison.', flush=True)
        else:
            print('DeepEP and MORI buffers dispatch/combine outputs match across ranks.', flush=True)


    dist.barrier()
    deep_perf = benchmark_buffer('DeepEP', buffer_deep,num_warmups=5, num_iters=50)
    mori_perf = benchmark_buffer('MORI', buffer_mori,num_warmups=5, num_iters=50)
    dist.barrier()
    if rank == 0 and deep_perf and mori_perf:
        dispatch_ratio = mori_perf['dispatch_avg_ms'] / max(deep_perf['dispatch_avg_ms'], 1e-6)
        combine_ratio = mori_perf['combine_avg_ms'] / max(deep_perf['combine_avg_ms'], 1e-6)
        print(f"[perf] MORI/DeepEP dispatch avg ratio: {dispatch_ratio:.3f}x", flush=True)
        print(f"[perf] MORI/DeepEP combine  avg ratio: {combine_ratio:.3f}x", flush=True)

    dist.destroy_process_group()


def main():
    for setting in PRESET_SETTINGS:
        num_processes = setting.get('num_processes', 2)
        print('-------------------------------------------------------------------------', flush=True)
        print(f"[info] spawning comparison for setting '{setting['name']}' (num_processes={num_processes})", flush=True)
        
        mp.spawn(compare_buffers, args=(num_processes, setting), nprocs=num_processes)
        print('*************************************************************************', flush=True)


if __name__ == '__main__':
    main()
