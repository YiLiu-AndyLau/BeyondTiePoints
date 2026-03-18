import torch
from pykeops.torch import LazyTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict
import time
import os

from rpc import RPCModelParameterTorch
from rs_image import RSImage

from adjustment_core.utils import TqdmLogger
import adjustment_core.ddp as ddp


class SingleDeviceModel:
    """Single-device wrapper with a DDP-like interface."""
    def __init__(self, module):
        self.module = module


def warp_local(local: torch.Tensor, dem: torch.Tensor, rpc_src: RPCModelParameterTorch, rpc_dst: RPCModelParameterTorch,
              affine_matrix: torch.Tensor):
    affine_matrix_double = affine_matrix.to(torch.double)
    ones = torch.ones(local.shape[0], 1).to(device=local.device, dtype=local.dtype)
    local_homo = torch.cat([local, ones], dim=-1)
    trans_local = local_homo.to(torch.double) @ affine_matrix_double.T

    lats, lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:, 1], trans_local[:, 0], dem)
    samps, lines = rpc_dst.RPC_OBJ2PHOTO(lats, lons, dem)
    warped_local = torch.stack([lines, samps], dim=-1).to(torch.float32)
    return warped_local


def feature_sampling(feature: torch.Tensor, conf: torch.Tensor, local: torch.Tensor, query: torch.Tensor, k=4):
    point_base = LazyTensor(local.contiguous().unsqueeze(0))
    query_lazy = LazyTensor(query.contiguous().unsqueeze(1))
    dist_ij: LazyTensor = ((query_lazy - point_base) ** 2).sum(-1)
    dists, idxs = dist_ij.Kmin_argKmin(K=k, dim=1)

    locals_kmin = local[idxs]
    dists = torch.cdist(query.unsqueeze(1), locals_kmin, p=2).squeeze(1)

    valid_mask = (dists.min(dim=1).values < 64)
    dists = dists[valid_mask]
    idxs = idxs[valid_mask]

    if dists.shape[0] == 0:
        return None, None, valid_mask

    dists_sum = torch.sum(dists, dim=1, keepdim=True).clamp_min(1e-6)
    dists_ratio = dists / dists_sum
    reverse_dists_ratio = 1. / (dists_ratio + 1e-6)
    weights = reverse_dists_ratio / torch.sum(reverse_dists_ratio, dim=1, keepdim=True)

    feature_sample_p3d = feature[idxs]
    feature_sample_pd = torch.sum(feature_sample_p3d * weights.unsqueeze(-1), dim=1).to(torch.float32)

    conf_sample_p3 = conf[idxs]
    conf_sample_p = torch.sum(conf_sample_p3 * weights, dim=1).to(torch.float32).detach()

    return feature_sample_pd, conf_sample_p, valid_mask


def fit_affine_bundle(args,
                      local_shared_grids,
                      images: List[RSImage],
                      model_ddp: DDP,
                      optimizer_r: torch.optim.Adam,
                      optimizer_t: torch.optim.Adam,
                      scheduler_r,
                      scheduler_t,
                      local_rank: int,
                      world_size: int,
                      patience: int,
                      current_level: int
                      ) -> List[Dict[str, torch.Tensor]]:
    """Optimize affine parameters with loss-based early stopping."""

    num_images = len(images)
    if num_images < 2 and local_rank == 0:
        print("Error: Need at least 2 images for bundle adjustment.")
        return []

    logger = TqdmLogger(args, args.max_iter, current_level, local_rank)

    if local_rank == 0:
        start_time = time.perf_counter()
        loss_log = ""

    best_model_state = []
    if local_rank == 0:
        min_metric_val = float('inf')
        patience_counter = 0
        loss_threshold = args.min_loss_threshold
        print(f"Starting optimization with criterion='loss', patience={patience}.")
        print(f"Using min_loss_threshold={loss_threshold}")

    stop_signal = torch.tensor(0.0, device=local_rank if torch.cuda.is_available() else 'cpu')

    for iter in range(args.max_iter):

        if optimizer_r:
            optimizer_r.zero_grad()
        if optimizer_t:
            optimizer_t.zero_grad()

        local_total_loss = torch.tensor(0.0, device=local_rank if torch.cuda.is_available() else 'cpu')
        num_valid_grids = 0

        for grid in local_shared_grids:
            grid_avg_loss = grid.calculate_all_pairs_loss(model_ddp, images, local_rank if torch.cuda.is_available() else 'cpu')

            if not torch.isnan(grid_avg_loss) and not torch.isinf(grid_avg_loss) and grid_avg_loss > 0:
                local_total_loss = local_total_loss + grid_avg_loss
                num_valid_grids += 1
            else:
                if grid_avg_loss > 0:
                    print(f"[Rank{local_rank}]: Detect invalid loss:{grid_avg_loss.item()} in Grid {grid.id}")

        if num_valid_grids > 0:
            local_total_loss = local_total_loss / num_valid_grids

        if local_total_loss > 0:
            local_total_loss.backward()

        if optimizer_r:
            optimizer_r.step()
        if optimizer_t:
            optimizer_t.step()

        global_loss_sum = local_total_loss.clone().detach()
        ddp.all_reduce_sum(global_loss_sum)
        global_avg_loss = (global_loss_sum / world_size).item()

        metric_log_dict = {}

        if local_rank == 0:
            current_metric_val = global_avg_loss
            current_threshold = args.min_loss_threshold
            metric_log_dict['min_met'] = f"{min_metric_val:.4f}"

            if current_metric_val > 0:
                if (min_metric_val - current_metric_val) > current_threshold:
                    min_metric_val = current_metric_val
                    patience_counter = 0

                    best_model_state = []
                    for sub_model in model_ddp.module.models:
                        best_model_state.append({
                            'R': sub_model.R.data.clone(),
                            'T': sub_model.T.data.clone()
                        })
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                print(f"--- Early stopping triggered at iter {iter + 1} based on 'loss' ---")
                print(f"Loss ({global_avg_loss:.4f}) did not improve by {args.min_loss_threshold} for {patience} iterations. Min loss: {min_metric_val:.4f}")
                stop_signal.fill_(1.0)

            metric_log_dict['lr_t'] = scheduler_t.get_last_lr()[0] if scheduler_t else 0
            metric_log_dict['lr_r'] = scheduler_r.get_last_lr()[0] if scheduler_r else 0

            logger.update(iter, global_avg_loss, metric_log_dict, patience_counter, patience)
            loss_log += f"{global_avg_loss:.6f}\n"

        ddp.broadcast_tensor(stop_signal, src=0)

        if stop_signal.item() == 1.0:
            if local_rank == 0:
                print(f"Rank {local_rank}: Received stop signal. Breaking optimization loop.")
            break

        if scheduler_r:
            scheduler_r.step()
        if scheduler_t:
            scheduler_t.step()

    logger.close()

    if local_rank == 0:
        print("Bundle adjustment optimization finished for this level.")
        end_time = time.perf_counter()
        print(f"Time Cost:{(end_time - start_time):.2f}")
        with open(os.path.join(args.debug_output_path, f'loss_log_level_{current_level}.txt'), 'w') as f:
            f.write(loss_log)

    return best_model_state
