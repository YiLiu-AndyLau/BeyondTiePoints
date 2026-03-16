import numpy as np
import torch.distributed as dist
from typing import List, Tuple, Dict

# 假设的外部依赖
from rs_image import RSImage

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """计算两组 (lat, lon) 坐标之间的 Haversine 距离 (米)"""
    R = 6371000 
    lat1 = coords1[:, 0]
    lon1 = coords1[:, 1]
    lat2 = coords2[:, 0]
    lon2 = coords2[:, 1]

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

def check_pair_error(img_i: RSImage, img_j: RSImage) -> np.ndarray:
    """ 计算单对影像 (i, j) 之间的连接点误差"""
    
    if img_i.tie_points is None or img_j.tie_points is None:
        # print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Missing tie points.")
        return np.array([])
        
    if len(img_i.tie_points) != len(img_j.tie_points):
        # print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Mismatched tie points count.")
        return np.array([])
    
    if len(img_i.tie_points) == 0:
        return np.array([])

    # 投影 img_i 的连接点
    lines_i = img_i.tie_points[:,0]
    samps_i = img_i.tie_points[:,1]
    heights_i = img_i.dem[lines_i,samps_i]
    lats_i, lons_i = img_i.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
    coords_i = np.stack([lats_i, lons_i], axis=-1)
    
    # 投影 img_j 的连接点
    lines_j = img_j.tie_points[:,0]
    samps_j = img_j.tie_points[:,1]
    heights_j = img_j.dem[lines_j,samps_j]
    lats_j, lons_j = img_j.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
    coords_j = np.stack([lats_j, lons_j], axis=-1)
    
    # 计算地理距离
    distances = haversine_distance(coords_i, coords_j)
    return distances

def calculate_error_report(images: List[RSImage], 
                           overlapping_pairs: List[Tuple[int, int]], 
                           verbose: bool = True) -> Dict[str, float]:
    """
    [Refactored] 统一的误差计算和报告函数。
    计算所有重叠对的误差，并返回一个包含统计数据的字典。
    如果 verbose=True (且在 Rank 0)，则打印详细报告。
    """
    all_distances_list = []
    is_rank_0 = dist.get_rank() == 0
    
    if verbose and is_rank_0:
        print("\n" + "--- Global Error Report ---")
        
    for (i, j) in overlapping_pairs:
        distances = check_pair_error(images[i], images[j])
        if len(distances) > 0:
            all_distances_list.append(distances)
            if verbose and is_rank_0:
                print(f"Pair ({i}, {j}) | Points: {len(distances)} | Mean Error: {distances.mean():.4f} m | Median Error: {np.median(distances):.4f} m")

    if not all_distances_list:
        if verbose and is_rank_0:
            print("No valid tie points found for any overlapping pair. Cannot generate report.")
        # 返回一个包含 NaN/0 的字典
        keys = ['mean', 'median', 'max', 'rmse', '<1m_percent', '<3m_percent', '<5m_percent']
        report = {k: np.nan for k in keys}
        report['count'] = 0
        return report
        
    all_distances = np.concatenate(all_distances_list)
    total_points = len(all_distances)
    
    if total_points == 0:
         # (理论上不会到这里，但作为保险)
        keys = ['mean', 'median', 'max', 'rmse', '<1m_percent', '<3m_percent', '<5m_percent']
        report = {k: np.nan for k in keys}
        report['count'] = 0
        return report

    report = {
        'mean': float(np.mean(all_distances)),
        'median': float(np.median(all_distances)),
        'max': float(np.max(all_distances)),
        'rmse': float(np.sqrt(np.mean(all_distances**2))),
        'count': int(total_points),
        '<1m_percent': float(((all_distances < 1.0).sum() / total_points) * 100),
        '<3m_percent': float(((all_distances < 3.0).sum() / total_points) * 100),
        '<5m_percent': float(((all_distances < 5.0).sum() / total_points) * 100),
    }

    if verbose and is_rank_0:
        print("\n" + "--- Global Error Report (Summary) ---")
        print(f"Total tie points checked: {report['count']}")
        print(f"Mean Error:   {report['mean']:.4f} m")
        print(f"Median Error: {report['median']:.4f} m")
        print(f"Max Error:    {report['max']:.4f} m")
        print(f"RMSE:         {report['rmse']:.4f} m")
        print(f"< 1.0 m: {report['<1m_percent']:.2f} %")
        print(f"< 3.0 m: {report['<3m_percent']:.2f} %")
        print(f"< 5.0 m: {report['<5m_percent']:.2f} %")
    
    return report
