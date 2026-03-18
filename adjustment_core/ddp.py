import os
import torch
import torch.distributed as dist


def _parse_use_ddp_arg(value: str) -> str:
    if value is None:
        return 'auto'
    v = str(value).strip().lower()
    if v in {'auto', 'true', 'false'}:
        return v
    raise ValueError(f"Invalid use_ddp value: {value}. Expected one of: auto/true/false")


def setup_distributed(use_ddp: str = 'auto'):
    """
    Initialize distributed runtime.

    Args:
        use_ddp: 'auto' | 'true' | 'false'

    Returns:
        (local_rank, world_size, ddp_enabled)
    """
    mode = _parse_use_ddp_arg(use_ddp)
    env_world_size = int(os.getenv('WORLD_SIZE', '1'))
    has_cuda = torch.cuda.is_available()

    if mode == 'true':
        ddp_enabled = True
    elif mode == 'false':
        ddp_enabled = False
    else:
        ddp_enabled = env_world_size > 1

    if ddp_enabled:
        backend = 'nccl' if has_cuda else 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        if has_cuda:
            torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1

    return local_rank, world_size, ddp_enabled


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def barrier():
    if is_distributed():
        dist.barrier()


def broadcast_object_list(obj_list, src: int = 0):
    if is_distributed():
        dist.broadcast_object_list(obj_list, src=src)


def all_reduce_sum(tensor: torch.Tensor):
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

def broadcast_tensor(tensor: torch.Tensor, src: int = 0):
    if is_distributed():
        dist.broadcast(tensor, src=src)


def destroy_distributed():
    if is_distributed():
        dist.destroy_process_group()
