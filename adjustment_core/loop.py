import torch
from pykeops.torch import LazyTensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Dict
import time
import os

# 假设的外部依赖
from rpc import RPCModelParameterTorch
from rs_image import RSImage

# 从同一核心模块导入
from adjustment_core.validation import calculate_error_report
from adjustment_core.utils import TqdmLogger
# [FIXED] 移除此处的循环导入
# from adjustment_core.data import SharedGrid # 仅用于类型提示


def warp_local(local:torch.Tensor,dem:torch.Tensor,rpc_src:RPCModelParameterTorch,rpc_dst:RPCModelParameterTorch,affine_matrix:torch.Tensor):
    # RPC内部计算是float64，但affine_matrix是float32，需要转换
    affine_matrix_double = affine_matrix.to(torch.double)
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    
    # 转换为double进行RPC计算
    trans_local = local_homo.to(torch.double) @ affine_matrix_double.T

    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32) # 输出转回float32
    return warped_local

def feature_sampling(feature:torch.Tensor, conf:torch.Tensor, local:torch.Tensor, query:torch.Tensor,k = 4):
    point_base = LazyTensor(local.contiguous().unsqueeze(0))
    query_lazy = LazyTensor(query.contiguous().unsqueeze(1))
    dist_ij:LazyTensor = ((query_lazy - point_base) ** 2).sum(-1)
    dists,idxs = dist_ij.Kmin_argKmin(K = k, dim=1)

    locals_kmin = local[idxs] # n,k,2
    dists = torch.cdist(query.unsqueeze(1),locals_kmin,p=2).squeeze(1)

    valid_mask = (dists.min(dim=1).values < 64)
    dists = dists[valid_mask]
    idxs = idxs[valid_mask]
    
    if dists.shape[0] == 0: # 如果没有有效的点
        return None, None, valid_mask

    dists_ratio = dists / torch.sum(dists,dim=1,keepdim=True) # n,k
    reverse_dists_ratio = 1. / (dists_ratio + 1e-6)
    weights = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)

    feature_sample_p3d = feature[idxs]
    feature_sample_pd = torch.sum(feature_sample_p3d * weights.unsqueeze(-1),dim=1).to(torch.float32)

    conf_sample_p3 = conf[idxs]
    conf_sample_p = torch.sum(conf_sample_p3 * weights,dim=1).to(torch.float32).detach()

    return feature_sample_pd,conf_sample_p,valid_mask

def fit_affine_bundle(args,
                      local_shared_grids, # [FIXED] 使用字符串前向引用
                      images: List[RSImage], 
                      model_ddp: DDP, 
                      optimizer_r: torch.optim.Adam, 
                      optimizer_t: torch.optim.Adam, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank:int, 
                      world_size:int,
                      patience: int,           
                      overlapping_pairs: List[Tuple[int, int]],
                      current_level: int 
                      ) -> List[Dict[str, torch.Tensor]]: 
    """
    (已修改) 使用DDP并行计算损失并优化仿射矩阵，支持基于loss或error的早停。
    (Refactored) 使用 TqdmLogger 统一处理日志记录。
    (Refactored) 使用 calculate_error_report 统一处理误差计算。
    """
    
    num_images = len(images)
    if num_images < 2 and local_rank == 0 and not args.auto:
        print("Error: Need at least 2 images for bundle adjustment.")
        return [] 
    
    # --- [Refactored] 初始化 Logger ---
    logger = TqdmLogger(args, args.max_iter, current_level, local_rank)

    if local_rank == 0:
        start_time = time.perf_counter()
        loss_log = ""
    
    # ---初始化早停和最佳模型变量 ---
    best_model_state = [] 
    if local_rank == 0:
        min_metric_val = float('inf') 
        patience_counter = 0
        criterion = args.stop_criterion
        loss_threshold = args.min_loss_threshold
        error_threshold = args.min_error_threshold
        
        if not args.auto:
            print(f"Starting optimization with criterion='{criterion}', patience={patience}.")
            if criterion == 'loss':
                print(f"Using min_loss_threshold={loss_threshold}")
            else: # criterion == 'error'
                print(f"Using min_error_threshold={error_threshold}m")
                # (check_error_during_train 已经在主文件中被强制启用)
                       
    stop_signal = torch.tensor(0.0, device=local_rank)

    # 4. 迭代优化
    for iter in range(args.max_iter):
        
        # [FIXED] 检查优化器是否存在
        if optimizer_r:
            optimizer_r.zero_grad()
        if optimizer_t:
            optimizer_t.zero_grad()
        
        local_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_grids = 0 
        
        # 5. 只在 *本地* 的格网子集上循环
        if len(local_shared_grids) == 0:
            pass 
        else:
            for grid in local_shared_grids:
                grid_avg_loss = grid.calculate_all_pairs_loss(model_ddp, images, local_rank)
                
                if not torch.isnan(grid_avg_loss) and not torch.isinf(grid_avg_loss) and grid_avg_loss > 0:
                    local_total_loss = local_total_loss + grid_avg_loss
                    num_valid_grids += 1
                else:
                    if grid_avg_loss > 0: 
                        print(f"[Rank{local_rank}]: Detect invalid loss:{grid_avg_loss.item()} in Grid {grid.id}")

            if num_valid_grids > 0:
                local_total_loss = local_total_loss / num_valid_grids
            
        # 7. 反向传播
        if local_total_loss > 0: # [FIXED] 仅在loss有效时反向传播
            local_total_loss.backward()
        
        if optimizer_r:
            optimizer_r.step()
        if optimizer_t:
            optimizer_t.step()
            
        # 1. 获取全局平均损失 (所有进程都需要)
        global_loss_sum = local_total_loss.clone().detach()
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
        global_avg_loss = (global_loss_sum / world_size).item() 
        
        # 2. Rank 0 进行决策
        metric_log_dict = {} # 用于存储本轮日志信息
        
        if local_rank == 0:
            
            mean_err = 0.0 # 初始化
            
            # 确定是否需要在本轮计算 error
            # (check_error_during_train 已在主函数中根据 stop_criterion 自动设置)
            should_calculate_error = args.check_error_during_train and (iter + 1) % 10 == 0
            
            # 计算 error (如果需要)
            if should_calculate_error:
                # 精度检查逻辑
                original_params_list = [img.rpc.adjust_params.clone() for img in images]
                original_params_inv_list = [img.rpc.adjust_params_inv.clone() for img in images]
                try:
                    with torch.no_grad():
                        for i in range(1, num_images): 
                            current_A_i = model_ddp.module.get_affine(i).detach()
                            images[i].rpc.Update_Adjust(current_A_i) 
                    
                    # --- [Refactored] 调用统一的验证函数 ---
                    # (仅在训练中调用，不需要详细打印，所以 verbose=False)
                    error_report = calculate_error_report(images, overlapping_pairs, verbose=False)
                    mean_err = error_report['mean']
                    metric_log_dict['err(m)'] = mean_err # 存入日志
                    # ---
                finally:
                    with torch.no_grad():
                        for i in range(num_images):
                            images[i].rpc.adjust_params = original_params_list[i]
                            images[i].rpc.adjust_params_inv = original_params_inv_list[i]

            # 确定本轮用于判断的指标和阈值
            current_metric_val = 0.0
            current_threshold = 0.0
            perform_check_this_iter = False 
            
            if args.stop_criterion == 'loss':
                current_metric_val = global_avg_loss
                current_threshold = args.min_loss_threshold
                perform_check_this_iter = True # loss 每轮都检查
                metric_log_dict['min_met'] = f"{min_metric_val:.4f}"
            elif args.stop_criterion == 'error' and should_calculate_error: # 只有计算了 error 的轮次才检查
                current_metric_val = mean_err 
                current_threshold = args.min_error_threshold
                perform_check_this_iter = True
                metric_log_dict['min_met'] = f"{min_metric_val:.2f}m"
            elif args.stop_criterion == 'error':
                metric_log_dict['min_met'] = f"{min_metric_val:.2f}m" # 保持显示
            
            # 执行判断 (仅在 perform_check_this_iter 为 True 时)
            if perform_check_this_iter and current_metric_val > 0: # 增加 > 0 检查，防止 error 为 0 时误判
                # 检查是否有显著改善 (注意: error 是越小越好)
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
                    # 没有显著改善
                    patience_counter += 1
            elif args.stop_criterion == 'error' and not should_calculate_error:
                 # 如果是 error 标准，但本轮未计算 error，则不增加 patience 计数器
                 pass
            elif perform_check_this_iter and current_metric_val <= 0 and args.stop_criterion == 'error':
                if not args.auto:
                    print(f"  Warning: Mean error is {current_metric_val:.4f}. Skipping best model check for this iteration.")


            # 检查是否需要早停
            if patience_counter >= patience:
                if not args.auto:
                    print(f"--- Early stopping triggered at iter {iter+1} based on '{args.stop_criterion}' ---")
                    if args.stop_criterion == 'loss':
                        print(f"Loss ({global_avg_loss:.4f}) did not improve by {args.min_loss_threshold} for {patience} iterations. Min loss: {min_metric_val:.4f}")
                    else: # error
                         print(f"Mean Error ({current_metric_val:.4f}m) did not improve by {args.min_error_threshold}m for {patience} check intervals. Min error: {min_metric_val:.4f}m")
                stop_signal.fill_(1.0) 

            # --- [Refactored] 日志记录 ---
            # 更新学习率 (非 auto 模式下需要)
            if not args.auto:
                 metric_log_dict['lr_t'] = scheduler_t.get_last_lr()[0] if scheduler_t else 0
                 metric_log_dict['lr_r'] = scheduler_r.get_last_lr()[0] if scheduler_r else 0
            
            # 调用 logger
            logger.update(iter, global_avg_loss, metric_log_dict, patience_counter, patience)
            loss_log += f"{global_avg_loss:.2f}\n"

        
        # 3.广播停止信号
        dist.broadcast(stop_signal, src=0)

        # 4.检查停止信号
        if stop_signal.item() == 1.0:
            if local_rank == 0 and not args.auto:
                print(f"Rank {local_rank}: Received stop signal. Breaking optimization loop.")
            break 
        
        if scheduler_r:
            scheduler_r.step()
        if scheduler_t:
            scheduler_t.step()

    # 优化循环结束
    logger.close() # [Refactored] 关闭 logger
    
    if local_rank == 0 and not args.auto:
        print("Bundle adjustment optimization finished for this level.")
        end_time = time.perf_counter()
        print(f"Time Cost:{(end_time - start_time):.2f}")
        with open(os.path.join(args.debug_output_path,f'loss_log_level_{current_level}.txt'),'w') as f:
            f.write(loss_log)

    return best_model_state

