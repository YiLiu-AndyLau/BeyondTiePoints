import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import sys
from typing import List, Dict, Tuple

# 假设的外部依赖
from rs_image import RSImage
from model.encoder import EncoderDino


def visualize_grid_selection(args, all_candidate_info: List[Dict], selected_diags: List[np.ndarray], ref_image: RSImage, level: int):
    """
    绘制格网选择示意图 (仅在 Rank 0 上调用)
    
    Args:
        args: 命令行参数
        all_candidate_info: 包含所有 *有效* 候选格网信息(diag, center, score)的字典列表
        selected_diags: 最终被选中的格网(diag)的列表
        ref_image: 参考影像 (例如 images[0]), 用于确定地理边界
        level: [新] 当前金字塔层级
    """
    # [修改] auto 模式下不运行
    if args.auto:
        return
        
    print(f"Rank 0: Generating grid selection visualization for Level {level}...")
    try:
        # 1. 获取参考影像的地理边界
        min_x = ref_image.corner_xys[:, 0].min()
        max_x = ref_image.corner_xys[:, 0].max()
        min_y = ref_image.corner_xys[:, 1].min()
        max_y = ref_image.corner_xys[:, 1].max()
        
        geo_w = max_x - min_x
        geo_h = max_y - min_y
        
        if geo_w == 0 or geo_h == 0:
            print("Rank 0: Invalid geographic bounds for visualization.")
            return

        # 2. 创建画布
        vis_h = 1000 # 固定高度
        aspect_ratio = geo_w / geo_h
        vis_w = int(vis_h * aspect_ratio)
        canvas = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255 # 白色背景

        # 3. 定义地理坐标到画布像素坐标的映射
        def geo_to_canvas(xy: np.ndarray) -> Tuple[int, int]:
            px = int((xy[0] - min_x) / geo_w * (vis_w - 1))
            py = int((max_y - xy[1]) / geo_h * (vis_h - 1)) # Y轴翻转 (地图坐标 -> 图像坐标)
            return (px, py)

        # 4. 绘制所有候选格网 (浅灰色)
        for info in all_candidate_info:
            diag = info['diag']
            # 构造完整的4个角点
            corners_geo = np.array([
                diag[0], [diag[1,0], diag[0,1]],
                diag[1], [diag[0,0], diag[1,1]]
            ])
            canvas_corners = [geo_to_canvas(pt) for pt in corners_geo]
            cv2.polylines(canvas, [np.array(canvas_corners, dtype=np.int32)], isClosed=True, color=(200, 200, 200), thickness=1)

        # 5. 绘制所有选中的格网 (绿色)
        selected_diags_set = {tuple(d.flatten()) for d in selected_diags}
        
        for diag in selected_diags:
            corners_geo = np.array([
                diag[0], [diag[1,0], diag[0,1]],
                diag[1], [diag[0,0], diag[1,1]]
            ])
            canvas_corners = [geo_to_canvas(pt) for pt in corners_geo]
            cv2.polylines(canvas, [np.array(canvas_corners, dtype=np.int32)], isClosed=True, color=(0, 200, 0), thickness=2) # 亮绿色

        # 6. 保存图像
        output_path = os.path.join(args.debug_output_path, f'grid_selection_visualization_level_{level}.png')
        cv2.imwrite(output_path, canvas)
        print(f"Rank 0: Saved grid selection visualization to {output_path}")

    except Exception as e:
        print(f"Rank 0: FAILED to generate grid visualization. Error: {e}")

def select_grids_by_confidence(args, 
                             all_candidate_diags: List[np.ndarray], 
                             num_to_select: int, 
                             images: List[RSImage], 
                             current_window_size: float, 
                             local_rank: int,
                             encoder: EncoderDino  # [Refactored] 接收 encoder
                             ) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    (新) 从候选格网中，通过置信度评估和空间NMS，筛选出指定数量的格网。
    (Refactored) Encoder 作为参数传入，不再内部加载。
    """
    
    if not args.auto:
        print(f"Rank {local_rank}: Strategy = Confidence-based selection...")
        print(f"Rank {local_rank}: Using pre-loaded encoder for grid quality assessment...")

    # [Refactored] 移除 EncoderDino 的加载
    encoder_assess = encoder.eval() # 确保是 eval 模式
    
    transform_assess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    
    all_valid_grids_info = [] 
    
    if not args.auto:
        print(f"Rank {local_rank}: Assessing quality for {len(all_candidate_diags)} candidate grids...")
    resample_size = 1024
    
    grid_iter = all_candidate_diags
    if args.auto:
        # (修改) 明确 position=1, leave=False
        grid_iter = tqdm(all_candidate_diags, desc=f"Lvl Assess Grids", leave=False, position=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',file=sys.stdout)

    with torch.no_grad():
        for diag in grid_iter:
            total_conf_score = 0.0
            overlapping_img_count = 0
            
            for img in images:
                try:
                    corners_geo = np.array([
                        diag[0], [diag[1,0], diag[0,1]],
                        diag[1], [diag[0,0], diag[1,1]]
                    ])
                    corners_sampline = img.xy_to_sampline(corners_geo)

                    if (corners_sampline.min() < 0 or 
                        corners_sampline[:, 0].max() > img.W or 
                        corners_sampline[:, 1].max() > img.H):
                        continue 

                    img_patch, _ = img.resample_image_by_sampline(corners_sampline, 
                                                                        (resample_size, resample_size), 
                                                                        need_local=True) 

                    img_tensor = transform_assess(img_patch)[None].cuda(local_rank)
                    _, conf = encoder_assess(img_tensor)

                    total_conf_score += conf.sum().item()
                    overlapping_img_count += 1
                    
                except Exception as e:
                    continue
            
            if overlapping_img_count >= 2:
                average_conf = total_conf_score / overlapping_img_count
                center_xy = diag.mean(axis=0)
                all_valid_grids_info.append({
                    'score': average_conf,
                    'center': center_xy,
                    'diag': diag
                })
    
    selected_tasks = []

    if num_to_select > 0 and len(all_valid_grids_info) > num_to_select:
        if not args.auto:
            print(f"Rank {local_rank}: Found {len(all_valid_grids_info)} valid grids. Selecting {num_to_select} using spatial selection...")
        
        all_valid_grids_sorted = sorted(all_valid_grids_info, key=lambda x: x['score'], reverse=True)
        candidate_grids_for_nms = all_valid_grids_sorted.copy()
        
        selected_grids_info_nms = [] 
        suppression_radius = current_window_size * 1.5 
        
        if not args.auto:
            print(f"Rank {local_rank}: Using suppression radius {suppression_radius:.2f} m...")

        while len(candidate_grids_for_nms) > 0 and len(selected_grids_info_nms) < num_to_select:
            best_grid = candidate_grids_for_nms.pop(0)
            selected_grids_info_nms.append(best_grid)
            
            remaining_grids = []
            for grid_info in candidate_grids_for_nms:
                distance = np.linalg.norm(best_grid['center'] - grid_info['center'])
                if distance > suppression_radius:
                    remaining_grids.append(grid_info)
            candidate_grids_for_nms = remaining_grids 
        
        num_selected_by_nms = len(selected_grids_info_nms)
        
        if num_selected_by_nms < num_to_select:
            if not args.auto:
                print(f"Rank 0: NMS 选中了 {num_selected_by_nms} 个格网 (目标: {num_to_select})。")
                print(f"Rank 0: 正在从高置信度列表中补齐剩余格网...")
            
            num_to_backfill = num_to_select - num_selected_by_nms
            selected_diags_set = {info['diag'].tostring() for info in selected_grids_info_nms}
            backfill_grids_info = []
            
            for grid_info in all_valid_grids_sorted:
                if grid_info['diag'].tostring() not in selected_diags_set:
                    backfill_grids_info.append(grid_info)
                    if len(backfill_grids_info) == num_to_backfill:
                        break
            
            if not args.auto:
                print(f"Rank 0: 已补齐 {len(backfill_grids_info)} 个格网。")
            final_selected_grids_info = selected_grids_info_nms + backfill_grids_info
        else:
            final_selected_grids_info = selected_grids_info_nms

        selected_tasks = [info['diag'] for info in final_selected_grids_info]
        
    else:
        if not args.auto:
             print(f"Rank {local_rank}: num_to_select ({num_to_select}) 为 0 或 >= 有效格网总数。使用所有 {len(all_valid_grids_info)} 个有效格网。")
        selected_tasks = [info['diag'] for info in all_valid_grids_info]

    # [Refactored] 不再删除 encoder，它由外部管理
    del transform_assess
    
    return selected_tasks, all_valid_grids_info

def select_grids_uniformly(args, 
                           all_candidate_diags: List[np.ndarray], 
                           num_to_select: int,
                           local_rank: int) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    (新) 从候选格网中，通过均匀抽样，筛选出指定数量的格网。
    """
    if not args.auto:
        print(f"Rank {local_rank}: Strategy = Uniform selection (fast, reproducible).")

    num_candidates = len(all_candidate_diags)
    all_valid_grids_info_for_vis = []
    selected_tasks = []

    if num_to_select > 0 and num_candidates > num_to_select:
        if not args.auto:
            print(f"Rank {local_rank}: Found {num_candidates} grids. Sorting and selecting {num_to_select} uniformly...")
        
        all_common_diags_list = list(all_candidate_diags)
        all_common_diags_list.sort(key=lambda diag: (diag.mean(axis=0)[0], diag.mean(axis=0)[1]))
        
        indices = np.linspace(0, num_candidates - 1, num_to_select, dtype=int)
        selected_tasks = [all_common_diags_list[i] for i in indices]
        
        selected_grids_map = {tuple(diag.flatten()) for diag in selected_tasks}
        for diag in all_common_diags_list: 
            is_selected = tuple(diag.flatten()) in selected_grids_map
            all_valid_grids_info_for_vis.append({
                'diag': diag,
                'center': diag.mean(axis=0),
                'score': 1 if is_selected else 0 
            })

    else:
        if not args.auto:
            print(f"Rank {local_rank}: Using all {num_candidates} grids (num_to_select ({num_to_select}) is 0 or >= num_candidates).")
        selected_tasks = list(all_candidate_diags) # 确保返回列表
        all_valid_grids_info_for_vis = [{'diag': diag, 'center': diag.mean(axis=0), 'score': 1} for diag in selected_tasks]

    return selected_tasks, all_valid_grids_info_for_vis


def subdivide_grids(parent_diags: List[np.ndarray]) -> List[np.ndarray]:
    """
    Takes a list of geographic grid diagonals (diags) and returns a new list
    containing the 4 sub-grids (quad-tree split) for each parent grid.
    """
    sub_grids = []
    for diag in parent_diags:
        # diag is np.array([[min_x, min_y], [max_x, max_y]])
        # 显式查找min/max，防止顺序问题
        min_x = np.min(diag[:, 0])
        max_x = np.max(diag[:, 0])
        min_y = np.min(diag[:, 1])
        max_y = np.max(diag[:, 1])
        
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0

        # 1. 左下 (Bottom-Left)
        sub_grids.append(np.array([[min_x, min_y], [mid_x, mid_y]]))
        # 2. 右下 (Bottom-Right)
        sub_grids.append(np.array([[mid_x, min_y], [max_x, mid_y]]))
        # 3. 左上 (Top-Left)
        sub_grids.append(np.array([[min_x, mid_y], [mid_x, max_y]]))
        # 4. 右上 (Top-Right)
        sub_grids.append(np.array([[mid_x, mid_y], [max_x, max_y]]))
        
    return sub_grids
