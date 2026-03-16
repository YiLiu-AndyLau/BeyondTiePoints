import os
import torch
import torch.distributed as dist

def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank
