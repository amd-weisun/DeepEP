import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mori
import deep_ep
from utils import init_dist, inplace_unique


NUM_TOKENS = 8
HIDDEN = 8
NUM_TOPK = 4
NVL_BUFFER_SIZE = 256
NUM_SAMPLES = 8


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

    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def assert_allclose(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name} mismatch (max diff {torch.max(torch.abs(a - b)):.6e})")


def compare_buffers(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, backend='gloo')
    num_experts = (16 // num_ranks) * num_ranks

    buffer_deep = deep_ep.Buffer(group, int(1e9), 0, low_latency_mode=False,
                                 num_qps_per_rank=num_experts // num_ranks)
    buffer_mori = mori.Buffer(group, int(1e9), 0, low_latency_mode=False,
                              num_qps_per_rank=num_experts // num_ranks,
                              max_num_inp_token_per_rank=NUM_TOKENS,
                              num_experts_per_token=NUM_TOPK,
                              gpu_per_node=num_local_ranks)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device('cuda', torch.cuda.current_device())
    row_values = torch.arange(NUM_TOKENS, dtype=torch.float32, device=device)
    row_values = row_values + rank * NUM_TOKENS
    x = row_values.unsqueeze(1).expand(NUM_TOKENS, HIDDEN).to(torch.bfloat16)
    scores = torch.randn((NUM_TOKENS, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, NUM_TOPK, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((NUM_TOKENS, NUM_TOPK), dtype=torch.float32, device='cuda') * rank

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = compute_dispatch_meta(
        topk_idx, num_experts, num_ranks, NUM_TOKENS)

    config = deep_ep.Config(NUM_SAMPLES, 8, NVL_BUFFER_SIZE)

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

    assert deep_num_list == mori_num_list, 'num_tokens_per_expert_list differs'
    assert torch.equal(deep_topk_idx, mori_topk_idx), 'topk indices differ'
    assert_allclose('recv_x', deep_recv_x.float(), mori_recv_x.float())
    assert_allclose('recv_topk_weights', deep_topk_weights, mori_topk_weights)

    deep_combined_x, deep_combined_weights, _ = buffer_deep.combine(deep_recv_x, deep_handle,
                                                                    topk_weights=deep_topk_weights,
                                                                    config=config)
    mori_combined_x, mori_combined_weights, _ = buffer_mori.combine(mori_recv_x, mori_handle,
                                                                     topk_weights=mori_topk_weights,
                                                                     config=config)

    assert_allclose('combined_x', deep_combined_x.float(), mori_combined_x.float())
    assert_allclose('combined_topk_weights', deep_combined_weights, mori_combined_weights)

    dist.barrier()
    if rank == 0:
        print('DeepEP and MORI buffers dispatch/combine outputs match across ranks.')

    dist.destroy_process_group()


def main():
    num_processes = 2
    mp.spawn(compare_buffers, args=(num_processes,), nprocs=num_processes)


if __name__ == '__main__':
    main()
