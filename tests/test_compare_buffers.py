import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional

import mori
import deep_ep
from utils import init_dist, inplace_unique


NUM_SMs = 8

PRESET_SETTINGS = [
    # {
    #     'name': 'baseline',
    #     'num_tokens': 8,
    #     'hidden': 8,
    #     'num_topk': 4,
    #     'num_experts': 16,
    #     'seed': 0,
    #     'log_values': True,
    #     'num_processes': 2,
    # },
    {
        'name': 'baseline_2',
        'num_tokens': 16,
        'hidden': 8,
        'num_topk': 8,
        'num_experts': 32,
        'seed': 0,
        'log_values': True,
        'num_processes': 2,
    },
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
    if token_order.min() < 0 or token_order.max() >= recv_x.size(0):
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_outputs guard: order contains invalid indices.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    unique_tokens = torch.unique(token_order)
    if unique_tokens.numel() != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_outputs guard: order contains repeated tokens.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    return recv_x[token_order], recv_topk_idx[token_order], recv_topk_weights[token_order]


def revert_mori_outputs(recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                         token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] revert_mori_outputs guard: shape mismatch or empty order.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    if token_order.min() < 0 or token_order.max() >= recv_x.size(0):
        if dist.get_rank() == 0:
            print('[warning] revert_mori_outputs guard: order contains invalid indices.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    unique_tokens = torch.unique(token_order)
    if unique_tokens.numel() != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] revert_mori_outputs guard: order contains repeated tokens.', flush=True)
        return recv_x, recv_topk_idx, recv_topk_weights
    original_x = torch.empty_like(recv_x)
    original_idx = torch.empty_like(recv_topk_idx)
    original_weights = torch.empty_like(recv_topk_weights)
    original_x[token_order] = recv_x
    original_idx[token_order] = recv_topk_idx
    original_weights[token_order] = recv_topk_weights
    return original_x, original_idx, original_weights

def reorder_mori_handle(handle: tuple[torch.Tensor, torch.Tensor], token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if token_order.numel() == 0 or handle[0].size(0) != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_handle guard: shape mismatch or empty order.', flush=True)
        return handle
    if token_order.min() < 0 or token_order.max() >= handle[0].size(0):
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_handle guard: order contains invalid indices.', flush=True)
        return handle
    unique_tokens = torch.unique(token_order)
    if unique_tokens.numel() != token_order.numel():
        if dist.get_rank() == 0:
            print('[warning] reorder_mori_handle guard: order contains repeated tokens.', flush=True)
        return handle
    return handle[0][token_order], handle[1]


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
    row_values = torch.arange(num_tokens, dtype=torch.float32, device=device)
    row_values = row_values + rank * num_tokens
    x = row_values.unsqueeze(1).expand(num_tokens, hidden).to(torch.bfloat16)
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    # x = x_pure_rand
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * (rank + 1.0)

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, mori_token_order = compute_dispatch_meta(
        topk_idx, num_experts, num_ranks, num_tokens)
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = deep_ep.Config(NUM_SMs, 8, nvl_buffer_size, 16, rdma_buffer_size)

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'config': config,
        'async_finish': False,
    }

    deep_output = buffer_deep.dispatch(**{k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                           for k, v in dispatch_args.items()})
    mori_output = buffer_mori.dispatch(**dispatch_args)

    def normalize_result(result):
        recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, _ = result
        if isinstance(recv_x, tuple):
            recv_x = recv_x[0]
        return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle

    deep_recv_x, deep_topk_idx, deep_topk_weights, deep_num_list, deep_handle = normalize_result(deep_output)
    mori_recv_x, mori_topk_idx, mori_topk_weights, mori_num_list, mori_handle = normalize_result(mori_output)
    mori_recv_x_orig = mori_recv_x.clone()
    mori_topk_idx_orig = mori_topk_idx.clone()
    mori_topk_weights_orig = mori_topk_weights.clone()

    mori_recv_x, mori_topk_idx, mori_topk_weights = reorder_mori_outputs(
        mori_recv_x, mori_topk_idx, mori_topk_weights, mori_handle[1])

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
    mori_recv_x_reverted, mori_topk_idx_reverted, mori_topk_weights_reverted = revert_mori_outputs(
        mori_recv_x, mori_topk_idx, mori_topk_weights, mori_handle[1])
    if rank == 0:
        if not torch.equal(mori_recv_x_reverted, mori_recv_x_orig):
            print('[warning] reverted recv_x deviates from original order', flush=True)
        if not torch.equal(mori_topk_idx_reverted, mori_topk_idx_orig):
            print('[warning] reverted topk_idx deviates from original order', flush=True)
        if not torch.equal(mori_topk_weights_reverted, mori_topk_weights_orig):
            print('[warning] reverted topk_weights deviates from original order', flush=True)
    mori_recv_x, mori_topk_idx, mori_topk_weights = mori_recv_x_reverted, mori_topk_idx_reverted, mori_topk_weights_reverted
    torch.cuda.synchronize()
    deep_combined_x, deep_combined_weights, _ = buffer_deep.combine(deep_recv_x, deep_handle,
                                                                    topk_weights=deep_topk_weights,
                                                                    config=config)
    mori_combined_x, mori_combined_weights, _ = buffer_mori.combine(mori_recv_x, mori_handle,
                                                                     topk_weights=mori_topk_weights,
                                                                     config=config)

    mismatch |= not warn_allclose('combined_x', deep_combined_x.float(), mori_combined_x.float(), rank=rank, log_values=log_values)
    mismatch |= not warn_allclose('combined_topk_weights', deep_combined_weights, mori_combined_weights, rank=rank, log_values=log_values)

    dist.barrier()
    if rank == 0:
        if mismatch:
            print('[warning] DeepEP and MORI buffers had mismatches during comparison.', flush=True)
        else:
            print('DeepEP and MORI buffers dispatch/combine outputs match across ranks.', flush=True)

    dist.destroy_process_group()


def main():
    for setting in PRESET_SETTINGS:
        num_processes = setting.get('num_processes', 2)
        print(f"[info] spawning comparison for setting '{setting['name']}' (num_processes={num_processes})", flush=True)
        mp.spawn(compare_buffers, args=(num_processes, setting), nprocs=num_processes)


if __name__ == '__main__':
    main()
