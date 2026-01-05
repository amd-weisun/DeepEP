import argparse
import datetime
import math
import os
import queue
import threading
from typing import Optional

import deep_ep
import mori
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import init_dist, inplace_unique, create_grouped_scores


NUM_SMs = 8

PRESET_SETTINGS = [
    # {
    #     'name': 'setting_0',
    #     'num_tokens': 16,
    #     'hidden': 256,
    #     'num_topk': 8,
    #     'num_experts': 32,
    #     'seed': 17,
    #     'log_values': False,
    # },
    # {
    #     'name': 'setting_1',
    #     'num_tokens': 128,
    #     'hidden': 4096,
    #     'num_topk': 8,
    #     'num_experts': 64,
    #     'seed': 17,
    #     'log_values': False,
    # },
    # {
    #     'name': 'setting_1_1',
    #     'num_tokens': 128,
    #     'hidden': 4096,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 17,
    #     'log_values': False,
    # },
    # {
    #     'name': 'setting_2_0',
    #     'num_tokens': 16,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 42,
    #     'log_values': False,
    # },
    # {
    #     'name': 'setting_2_1',
    #     'num_tokens': 32,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 42,
    #     'log_values': False,
    # },

    # {
    #     'name': 'setting_2_2',
    #     'num_tokens': 64,
    #     'hidden': 7168,
    #     'num_topk': 8,
    #     'num_experts': 256,
    #     'seed': 42,
    #     'log_values': False,
    # },
    {
        'name': 'setting_2a_debug',
        'num_tokens': 128,
        'hidden': 128,
        'num_topk': 8,
        'num_experts': 16,
        'seed': 42,
        'log_values': False,
    },
    {
        'name': 'setting_2a_debug',
        'num_tokens': 128,
        'hidden': 128,
        'num_topk': 8,
        'num_experts': 32,
        'seed': 42,
        'log_values': False,
    },
    {
        'name': 'setting_2b_debug',
        'num_tokens': 128,
        'hidden': 128,
        'num_topk': 8,
        'num_experts': 256,
        'seed': 42,
        'log_values': False,
    },
    {
        'name': 'setting_3',
        'num_tokens': 2048,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 256,
        'seed': 47,
        'log_values': False,
    },
    {
        'name': 'setting_4',
        'num_tokens': 4096,
        'hidden': 7168,
        'num_topk': 8,
        'num_experts': 256,
        'seed': 47,
        'log_values': False,
    },
]


class AsyncTensorDump:
    def __init__(self, file_path: str):
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file_path = file_path
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def log_tensor(self, label: str, tensor: torch.Tensor, context: str = ''):
        if tensor is None:
            return
        lines = _tensor_rows_as_lines(tensor)
        self._queue.put((label, context, lines))

    def log_orderless_tensor(self, label: str, tensor: torch.Tensor, context: str = ''):
        if tensor is None:
            return
        lines = _tensor_orderless_lines(tensor)
        self._queue.put((label, context, lines))

    def close(self):
        self._queue.put(None)
        self._thread.join()

    def _worker(self):
        with open(self._file_path, 'a') as handle:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                label, context, lines = item
                timestamp = datetime.datetime.utcnow().isoformat()
                prefix = f'[{timestamp}] {context} {label}\n'
                handle.write(prefix)
                handle.write('\n'.join(lines))
                handle.write('\n---\n')


def _format_row_for_logging(row: list[float]) -> str:
    if not row:
        return ''
    first = float(row[0])
    if len(row) > 1 and all(math.isclose(float(val), first, rel_tol=1e-6, abs_tol=1e-6) for val in row[1:]):
        return f'{first:.6f}'
    return ' '.join(f'{float(val):.6f}' for val in row)


def _tensor_rows_as_lines(tensor: torch.Tensor) -> list[str]:
    tens = tensor.detach().cpu()
    if tens.ndim == 0:
        return [f'{tens.item():.6f}']
    if tens.ndim == 1:
        return [_format_row_for_logging(tens.tolist())]
    lines: list[str] = []
    for row in tens:
        lines.append(_format_row_for_logging(row.tolist()))
    return lines


def _tensor_orderless_lines(tensor: torch.Tensor) -> list[str]:
    mat = tensor.detach().float()
    if mat.ndim == 0:
        return [f'{mat.item():.6f}']
    first_col = mat.reshape(mat.size(0), -1)[:, 0]
    sorted_vals = torch.sort(first_col)[0]
    return [f'{row.item():.6f}' for row in sorted_vals]


def _round_up_num_experts(base: int, num_ranks: int) -> int:
    per_rank = max((base + num_ranks - 1) // num_ranks, 1)
    return per_rank * num_ranks

def _build_all_rank_debug_data(
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    seed: int,
    device: torch.device = torch.device('cpu'),
):
    num_nodes = int(os.getenv('WORLD_SIZE', 2))
    total_values = num_ranks * num_tokens * hidden
    # values = torch.arange(total_values, dtype=torch.float32, device=device)
    
    data_gen = torch.Generator(device=device)
    data_gen.manual_seed(seed + num_ranks)
    values = torch.rand(total_values, dtype=torch.float32, device=device, generator=data_gen)

    all_rank_x = values.view(num_ranks, num_tokens, hidden).to(torch.bfloat16)
    rank_topk_idx = []
    rank_topk_weights = []
    for rank_id in range(num_ranks):
        idx_gen = torch.Generator(device=device)
        idx_gen.manual_seed(seed + rank_id * 127 + 1)
        # perm = torch.randperm(num_experts, generator=idx_gen, dtype=torch.long,  device=device)
        # topk_idx = perm[:num_topk].unsqueeze(0).expand(num_tokens, num_topk)
        # scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device,generator=data_gen).abs() + 1
        # topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
        group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
        group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
        masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
        topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]

        
        rank_topk_idx.append(topk_idx)

        weight_gen = torch.Generator(device=device)
        weight_gen.manual_seed(seed + num_ranks + rank_id * 211 + 3)
        weights = torch.rand((num_tokens, num_topk), generator=weight_gen, dtype=torch.float32, device=device)
        rank_topk_weights.append(weights)

    return torch.stack(rank_topk_idx, dim=0), torch.stack(rank_topk_weights, dim=0), all_rank_x


def compute_dispatch_meta(topk_idx: torch.Tensor, num_experts: int, num_ranks: int, num_tokens: int, num_local_ranks: int):
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

    num_nodes = max(num_ranks // max(num_local_ranks, 1), 1)
    rdma_rank_idx = rank_idx // max(num_local_ranks, 1)
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device=topk_idx.device)
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()

    return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank


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


def orderless_allclose(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-2, *, rank: Optional[int] = None, log_values: bool = True) -> bool:
    def summarize(tensor: torch.Tensor) -> torch.Tensor:
        mat = tensor.float().reshape(tensor.size(0), -1)
        return torch.sort(mat.sum(dim=1))[0]

    sum_a = summarize(a)
    sum_b = summarize(b)
    same = torch.allclose(sum_a, sum_b, rtol=rtol, atol=atol)
    if rank is None or rank == 0:
        if not same:
            max_diff = torch.max(torch.abs(sum_a - sum_b))
            print(f'[warning] {name} orderless mismatch: max diff {max_diff:.6e}', flush=True)
        else:
            print(f'[debug] {name} orderless match.', flush=True)
        if log_values:
            print(sum_a.cpu(), flush=True)
            print(sum_b.cpu(), flush=True)
        else:
            print(f'[info] {name} orderless tensor values suppressed (log_values False).', flush=True)
    return same


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





def compare_buffers(local_rank: int, num_local_ranks: int, backend: str, setting: dict, run_path: str):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, backend='gloo')
    torch.manual_seed(setting.get('seed', 0))
    torch.cuda.manual_seed_all(setting.get('seed', 0))
    num_experts = _round_up_num_experts(setting['num_experts'], num_ranks)
    num_tokens = setting['num_tokens']
    hidden = setting['hidden']
    num_topk = setting['num_topk']
    log_values = setting.get('log_values', True)

    num_nodes = int(os.getenv('WORLD_SIZE', 2))

    tensor_dumper: Optional[AsyncTensorDump] = None
    if run_path == 'both':
        sanitized_name = setting['name'].replace(' ', '_')
        dump_file = os.path.join('logs', 'internode', f'{sanitized_name}_{run_path}_rank{rank}.log')
        tensor_dumper = AsyncTensorDump(dump_file)

    if rank == 0:
        print(f"[info] running setting '{setting['name']}' with num_experts={num_experts}, num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, num_nodes = {num_nodes}, num_ranks = {num_ranks}", flush=True)
        print(f"[info] group.rank()={group.rank()} , group.size()={group.size()} ", flush=True)
        
    buffer_deep = deep_ep.Buffer(group, int(1e9), int(1e9) if (num_nodes > 1) else 0, low_latency_mode=False,
                                 num_qps_per_rank=max(num_experts // num_ranks, 1))
    buffer_mori = mori.Buffer(group, int(1e9), int(1e9), low_latency_mode=False,
                              num_qps_per_rank=max(num_experts // num_ranks, 1))

    
    device = torch.device('cuda', torch.cuda.current_device())
    all_rank_topk_idx, all_rank_topk_weights, all_rank_x = _build_all_rank_debug_data(
        num_ranks, num_tokens, hidden, num_experts, num_topk, setting.get('seed', 0)
    )
    local_x = all_rank_x[rank].to(device)
    topk_idx = all_rank_topk_idx[rank].to(device)
    topk_weights = all_rank_topk_weights[rank].to(device)

    # Data generation
    # torch.manual_seed(setting.get('seed', 0))
    # torch.cuda.manual_seed_all(setting.get('seed', 0))

    # device = torch.device('cuda', torch.cuda.current_device())
    
   

    # row_values = torch.arange(num_tokens, dtype=torch.float32, device=device)
    # row_values = row_values + rank * num_tokens
    # # local_x = row_values.unsqueeze(1).expand(num_tokens, hidden).to(torch.bfloat16)
    # local_x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    # scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    # topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    # topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') 


    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank = compute_dispatch_meta(
        topk_idx, num_experts, num_ranks, num_tokens, num_local_ranks)
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = deep_ep.Config(NUM_SMs, 8, nvl_buffer_size, 16, rdma_buffer_size)

    dispatch_args = {
        'x': local_x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'config': config,
        'async_finish': False,
    }

    run_deep = run_path in ('deep', 'both')
    run_mori = run_path in ('mori', 'both')

    deep_output = buffer_deep.dispatch(**{k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                           for k, v in dispatch_args.items()}) if run_deep else None
    mori_output = buffer_mori.dispatch(**dispatch_args) if run_mori else None

    def normalize_result(result):
        recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, _ = result
        if isinstance(recv_x, tuple):
            recv_x = recv_x[0]
        return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle

    deep_recv_x = deep_topk_idx = deep_topk_weights = deep_num_list = deep_handle = None
    mori_recv_x = mori_topk_idx = mori_topk_weights = mori_num_list = mori_handle = None
    if run_deep:
        deep_recv_x, deep_topk_idx, deep_topk_weights, deep_num_list, deep_handle = normalize_result(deep_output)
    if run_mori:
        mori_recv_x, mori_topk_idx, mori_topk_weights, mori_num_list, mori_handle = normalize_result(mori_output)

    mismatch = False
    if run_deep and run_mori:
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

        recv_match = warn_allclose('recv_x', deep_recv_x.float(), mori_recv_x.float(), rank=rank, log_values=log_values)
        mismatch |= not recv_match
        if not recv_match and tensor_dumper is not None:
            context = f"{setting['name']} rank{rank} recv_x"
            tensor_dumper.log_tensor('recv_x/deep_ep', deep_recv_x.float(), context)
            tensor_dumper.log_tensor('recv_x/mori', mori_recv_x.float(), context)
            tensor_dumper.log_orderless_tensor('recv_x/deep_ep_orderless', deep_recv_x.float(), context)
            tensor_dumper.log_orderless_tensor('recv_x/mori_orderless', mori_recv_x.float(), context)
            # if rank == 0:
            #     tensor_dumper.log_tensor('all_rank_x', all_rank_x.float(), context)
            #     tensor_dumper.log_tensor('all_rank_topk_idx', all_rank_topk_idx.to(torch.float32), context)
            #     tensor_dumper.log_tensor('all_rank_topk_weights', all_rank_topk_weights, context)
        mismatch |= not warn_allclose('recv_topk_weights', deep_topk_weights, mori_topk_weights_filtered, rank=rank, log_values=log_values)
    elif rank == 0:
        print(f'[info] Running only {"DeepEP" if run_deep else "MORI"} path; skipping cross-checks.', flush=True)

    torch.cuda.synchronize()
    dispatch_indices = mori_handle[0]
    if rank == 0:
        print(f'[info]dispatch_indices shape: {dispatch_indices.shape}', flush=True)
    
    if run_deep:
        deep_combined_x, deep_combined_weights, _ = buffer_deep.combine(deep_recv_x, deep_handle,
                                                                        topk_weights=deep_topk_weights,
                                                                        config=config)
    if run_mori:
        mori_combined_x, mori_combined_weights, _ = buffer_mori.combine(mori_recv_x, mori_handle,
                                                                         topk_weights=mori_topk_weights,
                                                                         config=config)

    if run_deep and run_mori:
        combined_match = warn_allclose('combined_x', deep_combined_x.float(), mori_combined_x.float(), rank=rank, log_values=log_values)
        mismatch |= not combined_match
        if not combined_match and tensor_dumper is not None:
            context = f"{setting['name']} rank{rank} combined_x"
            tensor_dumper.log_tensor('combined_x/deep_ep', deep_combined_x.float(), context)
            tensor_dumper.log_orderless_tensor('combined_x/deep_ep_orderless', deep_combined_x.float(), context)
            tensor_dumper.log_tensor('combined_x/mori', mori_combined_x.float(), context)
            tensor_dumper.log_orderless_tensor('combined_x/mori_orderless', mori_combined_x.float(), context)
        mismatch |= not orderless_allclose('combined_x', deep_combined_x, mori_combined_x, rank=rank, log_values=log_values)

    dist.barrier()
    if rank == 0:
        if mismatch:
            print('[warning] DeepEP and MORI buffers had mismatches during comparison.', flush=True)
        else:
            print('DeepEP and MORI buffers dispatch/combine outputs match across ranks.', flush=True)
    else:
        if rank == 0:
            print('Dispatch/combine finished for the selected path.', flush=True)

    dist.destroy_process_group()
    if tensor_dumper is not None:
        tensor_dumper.close()


def main():
    parser = argparse.ArgumentParser(description='Compare deepEP and MORI buffers in an internode setting')
    parser.add_argument('--backend', type=str, choices=['mpi', 'nccl', 'gloo'], default='gloo',
                        help='Backend for distributed communication (nccl/gloo via mp.spawn, mpi via mpiexec/mpirun)')
    parser.add_argument('--num-local-ranks', type=int, default=8,
                        help='Number of local ranks (GPUs) per node for spanning the test')
    parser.add_argument('--path', type=str, choices=['deep', 'mori', 'both'], default='both',
                        help='Select which buffer implementation to run for debugging')
    args = parser.parse_args()

    if args.backend == 'mpi':
        rank_env = int(os.getenv('RANK', '0'))
        local_rank = rank_env % args.num_local_ranks
        for setting in PRESET_SETTINGS:
            if rank_env == 0:
                print('-------------------------------------------------------------------------', flush=True)
                print(f"[info] launching '{setting['name']}' with backend mpi", flush=True)
            compare_buffers(local_rank, args.num_local_ranks, args.backend, setting, args.path)
    else:
        for setting in PRESET_SETTINGS:
            num_processes = args.num_local_ranks
            print('-------------------------------------------------------------------------', flush=True)
            print(f"[info] launching '{setting['name']}' with backend {args.backend} and {num_processes} local ranks", flush=True)
            mp.spawn(compare_buffers, args=(num_processes, args.backend, setting, args.path), nprocs=num_processes)
            print('*************************************************************************', flush=True)


if __name__ == '__main__':
    main()
