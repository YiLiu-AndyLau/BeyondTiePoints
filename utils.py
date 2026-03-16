import os
import cv2
from datetime import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.transform import AffineTransform
from skimage.measure import ransac
from PIL import Image
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import argparse
from rpc import RPCModelParameterTorch
from typing import Tuple
from sklearn.decomposition import PCA
from enum import Enum
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import ConnectionPatch
from matplotlib import pyplot as plt
import io
from shapely.geometry import Polygon
import math

def get_current_time():
    return datetime.now().strftime("%Y%m%d%H%M%S")
def crop_rect_from_image(image, rect_points, size):
    """
    从图像中截取矩形区域。

    参数:
    - image: 使用cv2.imread()读取的图像。
    - rect_points: 矩形的四个顶点坐标，按顺时针或逆时针顺序排列。
                  例如: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    返回:
    - cropped_image: 截取出的矩形图像。
    """
    # 将四个顶点转换为numpy数组
    rect = np.array(rect_points, dtype="float32")

    # 计算矩形的边界框的宽度和高度
    width_a = np.linalg.norm(rect[0] - rect[1])
    width_b = np.linalg.norm(rect[2] - rect[3])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(rect[0] - rect[3])
    height_b = np.linalg.norm(rect[1] - rect[2])
    max_height = int(max(height_a, height_b))

    # 目标矩形的四个角的坐标（仿射变换后的坐标）
    dst = np.array([[0, 0], [0, max_width-1], [max_height-1, max_width-1], [max_height-1, 0]], dtype="float32")

    rect_xy = np.array([[p[1],p[0]] for p in rect], dtype="float32")
    dst_xy = np.array([[p[1],p[0]]for p in dst], dtype="float32")

    # 计算仿射变换矩阵
    M = cv2.getPerspectiveTransform(rect_xy, dst_xy)
    M_inv = cv2.getPerspectiveTransform(dst, rect)

    # 使用仿射变换将图像中的矩形区域转换为目标矩形区域
    warped = cv2.warpPerspective(image.astype(np.float32), M, (max_width, max_height))

    if warped.shape[0] < size or warped.shape[1] < size:
        warped = cv2.resize(warped,(size,size))

    return warped.astype(np.float32),M_inv

def random_square_cut_and_affine(images, square_size, angle = None, margin = None):
    H, W = images[0].shape[:2]
    
    if angle is None:
        angle = np.random.uniform(-5,5)  # 随机旋转角度
    theta = np.deg2rad(angle)
    if margin is None:
        margin = int((np.abs(np.sin(theta)) + np.abs(np.cos(theta)))*square_size / 2) + 1
    center_x = np.random.uniform(margin + 1, W - margin - 1)
    center_y = np.random.uniform(margin + 1, H - margin - 1)
    # crop_upperleft = np.array([center_x - square_size / 2,center_y - square_size / 2],dtype=int)
    new_points = np.array([[-square_size / 2, -square_size / 2],
                        [ -square_size / 2, square_size / 2],
                        [ square_size / 2,  square_size / 2],
                        [square_size / 2,  -square_size / 2]])

    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    rotated_points = np.matmul(rotation_matrix,new_points.T).T + np.array([center_x, center_y])
    res = [crop_rect_from_image(image,rotated_points,square_size) for image in images]
    af_mat = res[0][1]

    return [i[0] for i in res],af_mat[:2,:],rotated_points


def estimate_affine_ransac(A, B, iterations=1000, threshold=0.1, whole=False, hp_num=3):
    max_inliers_num = 0
    best_affine_matrix = None

    def estimate_affine_transformation(A, B):
        # 中心化点集
        A_centered = A - np.mean(A, axis=0)
        B_centered = B - np.mean(B, axis=0)

        # 计算协方差矩阵
        H = A_centered.T @ B_centered

        # 进行奇异值分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = Vt.T @ U.T

        # 确保旋转矩阵的行列式为1
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # 计算平移
        t = np.mean(B, axis=0) - R @ np.mean(A, axis=0)

        # 组合成仿射变换矩阵
        affine_matrix = np.eye(3)
        affine_matrix[:2, :2] = R
        affine_matrix[:2, 2] = t

        return affine_matrix

    for _ in range(iterations):
        # 随机选择三个点
        sample_indices = random.sample(range(len(A)), hp_num)
        A_sample = A[sample_indices]
        B_sample = B[sample_indices]

        # 计算仿射变换矩阵
        affine_matrix = estimate_affine_transformation(A_sample, B_sample)

        # 计算在当前变换下的预测值
        B_pred = (affine_matrix[:2, :2] @ A.T).T + affine_matrix[:2, 2]

        # 计算内点
        distances = np.linalg.norm(B - B_pred, axis=1)
        inliers = distances < threshold

        # 更新最佳内点集
        if np.sum(inliers) > max_inliers_num:
            max_inliers_num = np.sum(inliers)
            if whole:
                whole_matrix = estimate_affine_transformation(A[inliers],B[inliers])
                B_pred = (whole_matrix[:2, :2] @ A.T).T + whole_matrix[:2, 2]
                # 计算内点
                distances = np.linalg.norm(B - B_pred, axis=1)
                inliers = distances < threshold
                if np.sum(inliers) > max_inliers_num:
                    max_inliers_num = np.sum(inliers)
                    best_affine_matrix = whole_matrix
                else:
                    best_affine_matrix = affine_matrix
            else:
                best_affine_matrix = affine_matrix

    return best_affine_matrix[:2,:], max_inliers_num

# def estimate_affine_ransac(points1, points2, threshold = 20):
#     model_robust, inliers = ransac((points1, points2), 
#                                 AffineTransform, 
#                                 min_samples=3, 
#                                 residual_threshold=threshold, 
#                                 max_trials=1000)
    
#     return model_robust.params[:2,:], len(inliers[inliers]) 

def calculate_errors(T_true, T_pred):
    # 提取旋转部分 (前2x2矩阵) 和 平移部分 (最后一列)
    R_true = T_true[:, :2]
    R_pred = T_pred[:, :2]
    
    t_true = T_true[:, 2]
    t_pred = T_pred[:, 2]
    
    # 计算平移误差（欧氏距离）
    translation_error = np.linalg.norm(t_pred - t_true)
    
    # 计算旋转角度
    theta_true = np.arctan2(R_true[1, 0], R_true[0, 0])
    theta_pred = np.arctan2(R_pred[1, 0], R_pred[0, 0])
    
    # 计算旋转误差（角度差异）
    rotation_error = np.degrees(np.abs(theta_pred - theta_true))
    
    return rotation_error, translation_error

class TableLogger():
    def __init__(self,folder_path:str,columns:list,prefix:str = 'log',name = None):
        self.df = pd.DataFrame(columns=columns)
        self.folder = folder_path
        if name is None:
            self.file_name = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        else:
            self.file_name = name
        self.path = os.path.join(self.folder,self.file_name)
        
        if not os.path.exists(self.path):
            self.df.to_csv(self.path,index=False)
        else:
            self.df = pd.read_csv(self.path)
    def update(self,row):
        try:
            self.df = self.df._append(row,ignore_index=True)
            self.df.to_csv(self.path,index=False)
        except:
            self.df = self.df.append(row,ignore_index=True)
            self.df.to_csv(self.path,index=False)

def average_downsample_matrix(matrix:np.ndarray, n):
    """
    对给定的numpy矩阵进行n倍平均下采样
    :param matrix: 输入的numpy矩阵
    :param n: 下采样倍数
    :return: 下采样后的矩阵
    """
    rows, cols = matrix.shape[:2]
    new_rows = rows // n
    new_cols = cols // n
    downsampled_matrix = np.zeros((new_rows, new_cols,*matrix.shape[2:]))
    for r in range(new_rows):
        for c in range(new_cols):
            downsampled_matrix[r, c] = matrix[r * n:(r + 1) * n, c * n:(c + 1) * n].mean(axis=(0,1))

    return downsampled_matrix

def get_coord_mat(H,W,downsample:int = 0):
    """
    return: [row,col] (H,W,2)
    """
    row_coords, col_coords = np.meshgrid(np.arange(0,H), np.arange(0,W), indexing='ij')
    coord_array = np.stack([row_coords, col_coords], axis=-1).astype(np.float32)  # (H,W, 2)
    if downsample > 0:
        coord_array = average_downsample_matrix(coord_array,downsample)
    return coord_array

def find_grids(quadrilaterals, side_length, offset_x=0.0, offset_y=0., grid_num = -1):
    # 检查输入是否有效
    if not isinstance(quadrilaterals, np.ndarray) or quadrilaterals.ndim != 3 or quadrilaterals.shape[1:] != (4, 2):
        raise ValueError("输入'quadrilaterals'必须是形状为 (N, 4, 2) 的Numpy数组。")
    if not isinstance(side_length, (int, float)) or side_length <= 0:
        raise ValueError("输入'side_length'必须是一个正数。")
    if quadrilaterals.shape[0] == 0:
        return np.empty((0, 2, 2)), None

    def _order_points_for_polygon(points):
        # 1. 计算质心
        centroid = np.mean(points, axis=0)
        
        # 2. 计算每个点相对于质心的角度
        angles = [math.atan2(p[1] - centroid[1], p[0] - centroid[0]) for p in points]
        
        # 3. 根据角度对点进行排序
        sorted_points = sorted(zip(points, angles), key=lambda item: item[1])
        
        # 返回排序后的点坐标
        return np.array([p for p, a in sorted_points])

    # --- 步骤 1: 将Numpy数组转换为Shapely多边形对象列表 ---
    try:
        # 已修改：在创建多边形前，先对其顶点进行排序，确保多边形有效。
        # .buffer(0) 仍然保留，作为处理其他潜在无效情况的最后防线。
        polygons = [Polygon(_order_points_for_polygon(q)).buffer(0) for q in quadrilaterals]
        polygons = [p for p in polygons if not p.is_empty]
        if not polygons:
             return np.empty((0, 2, 2)), None
    except Exception as e:
        raise ValueError(f"无法根据输入坐标创建多边形: {e}")

    # --- 步骤 2: 计算所有多边形的交集 ---
    intersection_area = polygons[0]
    for i in range(1, len(polygons)):
        intersection_area = intersection_area.intersection(polygons[i])
        if intersection_area.is_empty:
            return np.empty((0, 2, 2)), intersection_area

    if intersection_area.is_empty:
        return np.empty((0, 2, 2)), intersection_area

    # --- 步骤 3 & 4: 在交集的边界框内进行网格迭代 ---
    found_squares_coords = []
    minx, miny, maxx, maxy = intersection_area.bounds

    x = minx
    while x + side_length <= maxx:
        y = miny
        while y + side_length <= maxy:
            # --- 步骤 5: 创建候选正方形并检查是否被完全包含 ---
            square_poly = Polygon([
                (x, y),
                (x + side_length, y),
                (x + side_length, y + side_length),
                (x, y + side_length)
            ])

            if intersection_area.contains(square_poly):
                # 找到了一个有效的正方形。记录其左上角和右下角坐标，并应用偏移
                top_left_offset = [x + offset_x, y + side_length + offset_y]
                bottom_right_offset = [x + side_length + offset_x, y + offset_y]
                found_squares_coords.append([top_left_offset, bottom_right_offset])
            
            y += side_length
        x += side_length

    diags = np.array(found_squares_coords) if found_squares_coords else np.empty((0, 2, 2))

    if grid_num > 0:
        diags = diags[:grid_num]
    
    return diags

def kaiming_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

def norm_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.normal_(m.weight,0,0.001)
        if m.bias is not None:
            init.normal_(m.bias,0,0.001)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def warp_by_extend(points:torch.Tensor,extend:torch.Tensor):
    """
    points [x,y,h]
    return [y,x,h]
    """
    points = points.to(torch.double)
    extend = extend.to(torch.double).to(points.device)
    points[:,0] *= extend[6]
    points[:,1] *= extend[7] 
    points[:,2] = (points[:,2] + 1.) * (extend[9] - extend[8]) * 0.5 + extend[8]
    af_mat = extend[:6].reshape(2,3)
    R = af_mat[:2,:2]
    t = af_mat[:2,2]
    points[:,:2] = points[:,:2] @ R.T
    points[:,:2] = points[:,:2] + t
    points[:,[0,1]] = points[:,[1,0]]
    return points

def project_mercator(latlon:torch.Tensor):
    """
    (lat,lon) -> (y,x) N,2
    """
    r = 6378137.
    lon_rad = latlon[:,1] * torch.pi / 180.
    lat_rad = latlon[:,0] * torch.pi / 180.
    x = r * lon_rad
    y = r * torch.log(torch.tan(torch.pi / 4. + lat_rad / 2.))
    return torch.stack([y,x],dim=-1)

def mercator2lonlat(coord:torch.Tensor):
    """
    (y,x) -> (lat,lon) N,2
    """
    coord = torch.tensor(coord).to(torch.float64)
    r = 6378137.
    lon = (180. * coord[:,1]) / (torch.pi * r)
    lat = (2 * torch.atan(torch.exp(coord[:,0] / r)) - torch.pi * 0.5) * 180. / torch.pi
    return torch.stack([lat,lon],dim=-1)

def proj2photo(proj_coord:torch.Tensor,dem:torch.Tensor,rpc:RPCModelParameterTorch):
    lonlat_coord = mercator2lonlat(proj_coord)
    photo_coord = torch.stack(rpc.RPC_OBJ2PHOTO(lonlat_coord[:,0],lonlat_coord[:,1],dem),dim=1)
    return photo_coord

def bilinear_interpolate(array, points, use_cuda=False,device = None):
    """
    在矩阵上进行双线性插值采样，可选择在 CPU (NumPy) 或 GPU (PyTorch) 上运行。

    输入:
    - array: 二维 (H, W) 或三维 (H, W, C) 的 numpy 数组或 torch 张量。
    - points: (N, 2) 的浮点坐标数组或张量，每行表示一个坐标 [x, y]。
    - use_cuda:布尔值。如果为 True，则尝试使用 GPU (CUDA) 加速。

    输出:
    - 插值结果，形状为 (N,) 或 (N, C) 的 numpy 数组。
    """
    if device is None:
        device = 'cuda'
    # ----------- GPU (CUDA) 加速路径 -----------
    if use_cuda:
        # 检查 CUDA 是否可用，如果不可用则警告并回退到 CPU
        if not torch.cuda.is_available():
            print("警告：CUDA 不可用。将回退到 CPU (NumPy) 执行。")
            use_cuda = False
        else:
            device = torch.device(device)
            
            # 确保输入是 PyTorch 张量并移至 GPU
            # 使用 torch.as_tensor 避免不必要的数据拷贝
            arr_tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
            pts_tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
            
            # 将二维数组扩展为 (H, W, 1) 以统一处理
            if arr_tensor.dim() == 2:
                arr_tensor = arr_tensor.unsqueeze(-1)
            
            H, W, C = arr_tensor.shape
            x = pts_tensor[:, 0]
            y = pts_tensor[:, 1]
            
            # 计算整数坐标并约束边界
            # torch.floor 的结果是浮点数，需要转为长整型用于索引
            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            
            # 使用 torch.clamp 约束边界，等同于 np.clip
            x1 = torch.clamp(x0 + 1, 0, W - 1)
            x0 = torch.clamp(x0, 0, W - 1)
            y1 = torch.clamp(y0 + 1, 0, H - 1)
            y0 = torch.clamp(y0, 0, H - 1)
            
            # 提取四个角点的值，形状 (N, C)
            # PyTorch 的高级索引方式与 NumPy 相同
            Ia = arr_tensor[y0, x0, :]
            Ib = arr_tensor[y1, x0, :]
            Ic = arr_tensor[y0, x1, :]
            Id = arr_tensor[y1, x1, :]
            
            # 计算权重 (dx, dy 仍然是浮点数)
            dx = x - x0.float()
            dy = y - y0.float()
            
            wa = (1 - dx) * (1 - dy)
            wb = (1 - dx) * dy
            wc = dx * (1 - dy)
            wd = dx * dy
            
            # 加权求和 (使用 unsqueeze(1) 广播到所有通道)
            # wa[:, None] 在 PyTorch 中是 wa.unsqueeze(1)
            result_tensor = (
                wa.unsqueeze(1) * Ia +
                wb.unsqueeze(1) * Ib +
                wc.unsqueeze(1) * Ic +
                wd.unsqueeze(1) * Id
            )
            
            # 压缩多余的维度
            if arr_tensor.shape[-1] == 1 and arr_tensor.dim() == 3:
                result_tensor = result_tensor.squeeze(axis=1)
                
            # 将结果从 GPU 移回 CPU 并转换为 NumPy 数组
            return result_tensor.cpu().numpy()

    # ----------- CPU (NumPy) 原始路径 -----------
    # 如果 use_cuda 为 False，则执行原始逻辑
    array = np.asarray(array)
    points = np.asarray(points)
    
    # 记录原始维度以决定最终输出形状
    original_ndim = array.ndim
    
    if array.ndim == 2:
        array = array[..., np.newaxis]
    
    H, W, C = array.shape
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    x0 = np.clip(x0, 0, W - 1)
    
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    y0 = np.clip(y0, 0, H - 1)
    
    Ia = array[y0, x0, :]
    Ib = array[y1, x0, :]
    Ic = array[y0, x1, :]
    Id = array[y1, x1, :]
    
    dx = x - x0
    dy = y - y0
    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy
    
    result = (
        wa[:, None] * Ia +
        wb[:, None] * Ib +
        wc[:, None] * Ic +
        wd[:, None] * Id
    )
    
    if original_ndim == 2:
        result = result.squeeze(axis=1)
    
    return result

def downsample_average(
    input_tensor: torch.Tensor,
    downsample_factor: int,
    use_cuda: bool = False,
    device = 'cuda'
) -> torch.Tensor:
    """
    使用滑动窗口平均值对图像张量进行下采样。

    该函数接受一个形状为 (H, W) 或 (H, W, C) 的 PyTorch 图像张量，
    并使用一个 (downsample_factor x downsample_factor) 的窗口
    以 downsample_factor 为步长进行不重叠的滑动窗口下采样，
    并取窗口内的像素平均值。

    Args:
        input_tensor (torch.Tensor): 输入的图像张量，形状可以是 (H, W) [灰度图]
                                     或 (H, W, C) [彩色图]。
        downsample_factor (int): 下采样因子，将作为窗口大小和步长。例如，8 表示
                                 使用 8x8 的窗口下采样8倍。
        use_cuda (bool, optional): 如果为 True，则尝试使用 CUDA GPU 进行加速。
                                   如果 CUDA 不可用，将打印警告并回退到 CPU。
                                   默认为 False。

    Returns:
        torch.Tensor: 经过下采样后的图像张量。
                      如果输入是 (H, W)，输出是 (H/factor, W/factor)。
                      如果输入是 (H, W, C)，输出是 (H/factor, W/factor, C)。
                      输出张量将位于计算所用的设备上（CPU 或 CUDA）。

    Raises:
        ValueError: 如果输入张量的维度不是 2 或 3。
        TypeError: 如果输入不是一个 torch.Tensor。
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"输入必须是 torch.Tensor，但得到的是 {type(input_tensor)}")

    # --- 1. 检查和设置计算设备 (CPU or CUDA) ---
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device(device)
            # print("CUDA is available. Using GPU for acceleration.")
        else:
            device = torch.device('cpu')
            print("警告: 请求使用 CUDA，但 CUDA 不可用。将回退到 CPU。")
    else:
        device = torch.device('cpu')

    # 将输入张量移动到目标设备
    input_tensor = input_tensor.to(device)

    # --- 2. 预处理：将输入张量调整为 PyTorch 卷积层期望的格式 (N, C, H, W) ---
    # PyTorch 的 2D 卷积/池化层需要一个4D张量作为输入：(批量大小, 通道数, 高, 宽)
    input_dim = input_tensor.dim()
    if input_dim == 2:  # 灰度图 (H, W)
        is_grayscale = True
        # 扩展为 (1, 1, H, W)
        tensor_in = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_dim == 3:  # 彩色图 (H, W, C)
        is_grayscale = False
        # PyTorch 使用 "channels-first" (C, H, W) 格式，所以需要转换维度
        # (H, W, C) -> (C, H, W)，然后扩展为 (1, C, H, W)
        tensor_in = input_tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"输入张量的维度必须是 2 (H,W) 或 3 (H,W,C)，但得到的是 {input_dim}")

    # 确保输入张量是浮点数类型，以便计算平均值
    tensor_in = tensor_in.float()

    # --- 3. 定义并执行平均池化操作 ---
    # 使用 AvgPool2d 可以高效地完成滑动窗口平均操作
    # kernel_size 是窗口大小
    # stride 是滑动步长
    # 当 kernel_size 和 stride 相同时，窗口不会重叠
    pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor).to(device)
    downsampled_tensor = pool(tensor_in)

    # --- 4. 后处理：将输出张量恢复为原始格式 ---
    if is_grayscale:
        # (1, 1, H', W') -> (H', W')
        output_tensor = downsampled_tensor.squeeze(0).squeeze(0)
    else:
        # (1, C, H', W') -> (C, H', W') -> (H', W', C)
        output_tensor = downsampled_tensor.squeeze(0).permute(1, 2, 0)

    return output_tensor.cpu()

def downsample(arr:torch.Tensor,ds,use_cuda=False,show_detail=False,mode='mid',device = None):
    """
    mode: mid or avg
    """
    if ds <= 0:
        return arr
    # if len(arr.shape) < 4:
    #     arr = arr.unsqueeze(-1)
    arr_ds = []
    if show_detail:
        pbar = tqdm(total = len(arr))
    if device is None:
        device = 'cuda'
    for a in arr:
        if mode == 'mid':
            if len(a.shape) < 3:
                a = a.unsqueeze(-1)
            H,W = a.shape[:2]
            lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
            samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
            sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
            a = torch.tensor(bilinear_interpolate(a,sample_idxs,use_cuda=use_cuda))
            arr_ds.append(a.reshape(len(lines),len(samps),-1).squeeze())
        elif mode == 'avg':
            a_ds = downsample_average(a,ds,use_cuda)
            arr_ds.append(a_ds)
        else:
            raise ValueError("downsample mode should either be mid or avg")
        if show_detail:
            pbar.update(1)
    arr_ds = torch.stack(arr_ds,dim=0)
    return arr_ds

def print_hwc_matrix(matrix: np.ndarray, precision:int = 2):
    """
    将一个形状为 (H, W, C) 的 NumPy 数组在终端中以 H*W 矩阵的格式打印出来。
    增加了对浮点数格式化的支持。

    Args:
        matrix (np.ndarray): 一个三维的 NumPy 数组，形状为 (H, W, C)。
        precision (Optional[int], optional): 
            当数组是浮点类型时，指定要保留的小数位数。
            如果为 None，则使用默认的字符串表示。默认为 None。
    """
    # 检查输入是否为三维 NumPy 数组
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 3:
        print("错误：输入必须是一个三维的 NumPy 数组 (H, W, C)。")
        return

    # 获取数组的维度
    H, W, C = matrix.shape

    # 如果数组为空，则不打印
    if H == 0 or W == 0:
        print("[]")
        return
    
    string_elements = []
    for h in range(H):
        row_elements = []
        for w in range(W):
            vector = matrix[h, w]
            string_element = ""
            if precision is not None:
                # 如果指定了精度，对向量中的每个数字进行格式化
                try:
                    # 使用 f-string 的嵌套格式化功能
                    formatted_numbers = [f"{num:.{precision}f}" for num in vector]
                    string_element = f"[{' '.join(formatted_numbers)}]"
                except (ValueError, TypeError):
                    # 如果格式化失败（例如，数组不是数字类型），则退回默认方式
                    string_element = str(vector)
            else:
                # 未指定精度，使用 NumPy 默认的字符串转换
                string_element = str(vector)
            
            row_elements.append(string_element)
        string_elements.append(row_elements)

    # 找到所有字符串化后的元素中的最大长度，用于对齐
    max_len = max([len(s) for row in string_elements for s in row] or [0])

    # 打印带边框的矩阵
    print("┌" + "─" * (W * (max_len + 2) - 2) + "┐")
    for row in string_elements:
        print("│", end="")
        for element in row:
            # 使用 ljust 方法填充空格，使每个元素占据相同的宽度
            print(f"{element:<{max_len}}", end="  ")
        print("│")
    print("└" + "─" * (W * (max_len + 2) - 2) + "┘")

def apply_polynomial(x, coefs):
    y = torch.zeros_like(x)
    for i, c in enumerate(coefs):
        y = y + c * (x ** (len(coefs) - 1 - i))
    return y

def get_map_coef(target:np.ndarray,bins=1000,deg=20):
    extend_bins = int(bins * 0.1)
    src = np.linspace(0,1,bins)
    tgt = np.quantile(target,src)
    tgt = np.concatenate([2 * tgt[0] - tgt[:extend_bins][::-1],tgt,2 * tgt[-1] - tgt[-extend_bins:][::-1]],axis=0)
    src = np.linspace(-1,1,bins + 2 * extend_bins)
    coefs = np.polyfit(src,tgt,deg = deg)
    return coefs

def resample_from_quad(
    source_image: np.ndarray,
    quad_coords: np.ndarray,
    target_shape: Tuple[int, int],
    tile_size: int = 4096,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REPLICATE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据四边形角点重采样图像，并返回坐标映射。
    此最终版本通过“目标分块”和“源动态裁剪”解决了目标和源图像均可能超大尺寸的问题。

    Args:
        source_image (np.ndarray): 输入的源图像 (H, W) 或 (H, W, 3)。
        quad_coords (np.ndarray): 四边形的四个角点坐标 (4, 2), (row, col)。
        target_shape (tuple[int, int]): 目标输出图像的尺寸 (h, w)。
        tile_size (int, optional): 分块处理的块边长。
        interpolation (int, optional): 插值方法。
        border_mode (int, optional): 边界模式。

    Returns:
        tuple[np.ndarray, np.ndarray]: 重采样图像和坐标映射。
    """
    # 1. 输入验证和矩阵计算 (与之前相同)
    # ... (此处省略了与上一版相同的验证和矩阵计算代码，请直接复制过来)
    if source_image.ndim not in [2, 3]:
        raise ValueError("输入图像必须是二维 (灰度图) 或三维 (RGB/BGR图) 数组。")
    if quad_coords.shape != (4, 2):
        raise ValueError("角点坐标数组的形状必须是 (4, 2)。")

    h, w = target_shape
    src_h, src_w = source_image.shape[:2]
    
    src_points = quad_coords[:, ::-1].astype(np.float32)
    dst_points = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print("错误: 变换矩阵 M 是奇异矩阵，无法计算逆矩阵。")
        return None, None

    # 2. 准备空的输出画布 (与之前相同)
    if source_image.ndim == 3:
        output_channels = source_image.shape[2]
        resampled_image = np.zeros((h, w, output_channels), dtype=source_image.dtype)
    else:
        resampled_image = np.zeros((h, w), dtype=source_image.dtype)
    
    coordinate_map = np.zeros((h, w, 2), dtype=np.float32)

    # 3. 分块处理
    for y_start in range(0, h, tile_size):
        for x_start in range(0, w, tile_size):
            tile_h = min(tile_size, h - y_start)
            tile_w = min(tile_size, w - x_start)
            
            # a. 为当前块创建目标坐标网格并变换回源坐标系
            y_grid, x_grid = np.mgrid[y_start : y_start + tile_h, x_start : x_start + tile_w]
            target_coords_homo = np.stack((x_grid.ravel(), y_grid.ravel(), np.ones(tile_h * tile_w)), axis=1)
            source_coords_homo = target_coords_homo @ M_inv.T
            w_inv = 1.0 / (source_coords_homo[:, 2] + 1e-9)
            source_coords_xy = source_coords_homo[:, :2] * w_inv[:, np.newaxis]
            
            # b. 计算所需源区域的边界框 (Bounding Box)
            # map1 是 x 坐标 (col), map2 是 y 坐标 (row)
            map1 = source_coords_xy[:, 0]
            map2 = source_coords_xy[:, 1]
            
            # 计算边界并增加一点 padding，以防插值时访问到边界外
            padding = 2 
            src_x_min = max(0, int(np.floor(map1.min())) - padding)
            src_x_max = min(src_w, int(np.ceil(map1.max())) + padding)
            src_y_min = max(0, int(np.floor(map2.min())) - padding)
            src_y_max = min(src_h, int(np.ceil(map2.max())) + padding)

            # 如果所需区域完全在源图像外，则跳过
            if src_x_min >= src_w or src_x_max <= 0 or src_y_min >= src_h or src_y_max <= 0:
                continue

            # c. 裁剪源图像ROI和调整坐标
            source_roi = source_image[src_y_min:src_y_max, src_x_min:src_x_max]
            
            # 将绝对坐标调整为相对于 ROI 的坐标
            adjusted_map1 = (map1 - src_x_min).reshape(tile_h, tile_w).astype(np.float32)
            adjusted_map2 = (map2 - src_y_min).reshape(tile_h, tile_w).astype(np.float32)

            # d. 使用裁剪后的 ROI 和调整后的 map 调用 remap
            resampled_tile = cv2.remap(
                source_roi,         #  使用裁剪后的小图
                adjusted_map1,      #  使用调整后的坐标
                adjusted_map2,      #  使用调整后的坐标
                interpolation=interpolation,
                borderMode=border_mode
            )
            
            # e. 拼接结果
            resampled_image[y_start:y_start+tile_h, x_start:x_start+tile_w] = resampled_tile
            coordinate_map[y_start:y_start+tile_h, x_start:x_start+tile_w, 0] = map2.reshape(tile_h, tile_w)
            coordinate_map[y_start:y_start+tile_h, x_start:x_start+tile_w, 1] = map1.reshape(tile_h, tile_w)
            
    return resampled_image, coordinate_map

def vis_feat_pca(feat:np.ndarray,output_path = None):
    """
    feat shape:(H,W,C)
    """
    H,W,C = feat.shape
    feat = feat.reshape(-1,C)
    pca = PCA(n_components=3)
    feat = pca.fit_transform(feat)
    feat = 255. * (feat - feat.min()) / (feat.max() - feat.min())
    feat = feat.reshape(H,W,3).astype(np.uint8)
    if not output_path is None:
        cv2.imwrite(output_path,feat)
    else:
        return feat

def vis_feat_twin(feat1,feat2):
    H,W,C = feat1.shape
    # feat1 = feat1.permute(1,2,0).flatten(0,1).cpu().numpy()
    # feat2 = feat2.permute(1,2,0).flatten(0,1).cpu().numpy()
    feat1 = feat1.reshape(-1,C)
    feat2 = feat2.reshape(-1,C)
    feat = np.concatenate([feat1,feat2],axis=0)
    # tsne = TSNE(n_components=3, random_state=42,metric='cosine')
    # feat = tsne.fit_transform(feat)
    pca = PCA(n_components=3)
    
    feat = pca.fit_transform(feat)
    # feat = feat[:,:3]
    feat = (feat - feat.min()) / (feat.max() - feat.min())
    feat1 = feat[:H*W]
    feat2 = feat[H*W:]
    feat1 = feat1.reshape(H,W,3)
    feat2 = feat2.reshape(H,W,3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 在第一个子图中显示第一张图片
    ax1.imshow(feat1)
    ax1.axis('off')  # 关闭坐标轴
    ax1.set_title('Image 1')

    # 在第二个子图中显示第二张图片
    ax2.imshow(feat2)
    ax2.axis('off')  # 关闭坐标轴
    ax2.set_title('Image 2')

    # 调整布局
    plt.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image_buffer = fig.canvas.buffer_rgba()
    image_array = np.frombuffer(image_buffer, dtype=np.uint8)
    image_array = image_array.reshape(height, width, 4)[...,:3]

    return image_array
    
def vis_conf(conf:np.ndarray,img:np.ndarray,ds,div = .5,output_path = None):
    points = (get_coord_mat(conf.shape[0],conf.shape[1]) * ds + ds * .5).reshape(-1,2)
    scores = conf.reshape(-1)
    canvas_cont = deepcopy(img)
    canvas_div = deepcopy(img)

    def score_to_color_cont(score):
        red = int((1 - score) * 255)
        green = int(score * 255)
        return (red , green, 0)
    
    def score_to_color_div(score,div):
        if score >= div:
            return (0,255,0)
        else:
            return (255,0,0)
    
    for p,score in zip(points,scores):
        p = p.astype(int)
        color_cont = score_to_color_cont(score)
        color_div = score_to_color_div(score,div)

        cv2.circle(canvas_cont,(p[1],p[0]),radius=1,color=color_cont,thickness=-1)
        cv2.circle(canvas_div,(p[1],p[0]),radius=1,color=color_div,thickness=-1)
    
    if not output_path is None:
        cv2.imwrite(output_path.replace('.png','_cont.png'),cv2.cvtColor(canvas_cont,cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_path.replace('.png','_div.png'),cv2.cvtColor(canvas_div,cv2.COLOR_RGB2BGR))
    else:
        return canvas_cont,canvas_div

def visualize_subset_points(points1, points2, output_path, padding=50, point_radius=5):
    # 将两组点合并，以确定画布的整体尺寸
    all_points = np.vstack((points1, points2)) if points1.size > 0 and points2.size > 0 else \
                points1 if points1.size > 0 else points2

    min_x = np.min(all_points[:, 0])
    min_y = np.min(all_points[:, 1])

    # 计算所有点的最大 x 和 y 坐标
    max_x = np.max(all_points[:, 0]) - min_x
    max_y = np.max(all_points[:, 1]) - min_y

    points1[:,0] -= min_x
    points1[:,1] -= min_y
    points2[:,0] -= min_x
    points2[:,1] -= min_y
    
    

    # 根据最大坐标和边距计算画布尺寸
    canvas_width = int(max_x + padding * 2)
    canvas_height = int(max_y + padding * 2)

    # 创建一个白色画布 (BGR 格式)
    # np.ones 创建一个浮点数数组，乘以 255，然后转换为 uint8 类型
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 定义颜色 (OpenCV 使用 BGR 顺序)
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)

    # 绘制第二组点（红色）
    for point in points2:
        # 将坐标转换为整数元组，并加上边距
        center = (int(point[1]) + padding, int(point[0]) + padding)
        cv2.circle(canvas, center, point_radius, red_color, thickness=-1)

    # 绘制第一组点（绿色）
    for point in points1:
        # 将坐标转换为整数元组，并加上边距
        center = (int(point[1]) + padding, int(point[0]) + padding)
        cv2.circle(canvas, center, point_radius, green_color, thickness=-1) # thickness=-1 表示实心圆

    

    # 保存图像到指定路径
    cv2.imwrite(output_path, canvas)

def visualize_feature_correspondences(
    feature_map1: np.ndarray,
    feature_map2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    draw_lines: bool = True
) -> np.ndarray:
    """
    将两个特征图及其对应点进行可视化。

    该函数通过联合PCA将特征图降维并归一化到RGB空间，然后并排绘制它们。
    对应点会用相同的颜色在两张图上标记出来，以便于比较特征的相似性。

    Args:
        feature_map1 (np.ndarray): 第一个特征图，形状为 (H, W, D)。
        feature_map2 (np.ndarray): 第二个特征图，形状为 (H, W, D)。
        points1 (np.ndarray): 第一个特征图上的对应点坐标，形状为 (N, 2)，格式为 (x, y)。
        points2 (np.ndarray): 第二个特征图上的对应点坐标，形状为 (N, 2)，格式为 (x, y)。
        draw_lines (bool): 是否在对应点之间绘制连接线，默认为 True。

    Returns:
        np.ndarray: 一个包含最终可视化结果的RGB图像的NumPy数组。
    """
    # --- 1. 输入验证 ---
    if not (isinstance(feature_map1, np.ndarray) and isinstance(feature_map2, np.ndarray) and
            isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray)):
        raise TypeError("所有输入都必须是NumPy数组。")
        
    if feature_map1.shape[:2] != feature_map2.shape[:2]:
        raise ValueError("两个特征图的高度（H）和宽度（W）必须相同。")
    if feature_map1.shape[2] != feature_map2.shape[2]:
        raise ValueError("两个特征图的特征维度（D）必须相同。")
    if points1.shape != points2.shape:
        raise ValueError("两组对应点的数量和维度必须相同。")
    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError("坐标点数组的形状必须是 (N, 2)。")

    H, W, D = feature_map1.shape
    N = points1.shape[0]

    # --- 2. 联合PCA降维与归一化 ---
    # 为了公平比较，我们将两个特征图的数据合并在一起进行PCA和缩放
    fm1_reshaped = feature_map1.reshape((H * W, D))
    fm2_reshaped = feature_map2.reshape((H * W, D))
    all_features = np.concatenate([fm1_reshaped, fm2_reshaped], axis=0)

    # 应用PCA将特征降到3维
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(all_features)

    # # 使用MinMaxScaler将PCA结果归一化到[0, 1]范围，以便显示为RGB颜色
    # scaler = MinMaxScaler(feature_range=(1e-6, 1. - 1e-6))
    # features_normalized = scaler.fit_transform(features_pca)
    features_normalized = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min() + 1e-9)

    # 将处理后的数据分离并重塑为两个RGB图像
    img1_rgb = features_normalized[:H * W, :].reshape((H, W, 3))
    img2_rgb = features_normalized[H * W:, :].reshape((H, W, 3))

    # --- 3. 使用Matplotlib进行绘图 ---
    # 根据图像的宽高比动态计算画布大小
    fig_w = 12
    fig_h = fig_w * H / (W * 2) if W > 0 else 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=150)
    fig.tight_layout(pad=3.0)

    # 显示两个降维后的特征图
    ax1.imshow(img1_rgb)
    ax1.axis('off')

    ax2.imshow(img2_rgb)
    ax2.axis('off')

    # --- 4. 绘制对应点和连接线 ---
    if N > 0:
        # **修改后的逻辑**:
        # 1. 从降维后的RGB图像中直接采样，作为点的颜色
        # 2. 保留独立的颜色映射(jet)，仅用于绘制连接线，以唯一标识匹配对

        # 将浮点坐标转换为整数索引用于颜色采样
        # 注意：numpy索引是(行, 列)，对应于(y, x)
        points1_idx = np.round(points1).astype(int)
        points2_idx = np.round(points2).astype(int)
        
        # 裁剪索引以确保它们在图像边界内
        points1_idx[:, 0] = np.clip(points1_idx[:, 0], 0, W - 1)
        points1_idx[:, 1] = np.clip(points1_idx[:, 1], 0, H - 1)
        points2_idx[:, 0] = np.clip(points2_idx[:, 0], 0, W - 1)
        points2_idx[:, 1] = np.clip(points2_idx[:, 1], 0, H - 1)

        # 在对应点位置采样RGB颜色
        point_colors1 = img1_rgb[points1_idx[:, 1], points1_idx[:, 0]]
        point_colors2 = img2_rgb[points2_idx[:, 1], points2_idx[:, 0]]
        
        # 在两张图上分别绘制点，点的颜色是其背景特征的颜色
        # s是点的大小, edgecolor使其在各种背景下都可见
        ax1.scatter(points1[:, 0], points1[:, 1], c=point_colors1, s=25, edgecolor='white', linewidth=1.0, zorder=3)
        ax2.scatter(points2[:, 0], points2[:, 1], c=point_colors2, s=25, edgecolor='white', linewidth=1.0, zorder=3)

        # (可选) 在对应点之间绘制连接线
        if draw_lines:
            # 生成 N 个独特的颜色，专门用于连接线，以便唯一标识匹配对
            line_cmap = plt.get_cmap('jet')
            line_colors = line_cmap(np.linspace(0, 1, N))
            for i in range(N):
                con = ConnectionPatch(
                    xyA=points2[i], xyB=points1[i],
                    coordsA=ax2.transData, coordsB=ax1.transData,
                    axesA=ax2, axesB=ax1,
                    color=line_colors[i], linewidth=1.2,
                    linestyle='dashed', zorder=2
                )
                fig.add_artist(con)

    # --- 5. 将画布转换为NumPy数组 ---
    fig.canvas.draw()
    # 从buffer中获取RGB数据
    width, height = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((height, width, 4))[:,:,1:]

    # 关闭图形，防止在Jupyter等环境中自动显示
    plt.close(fig)

    return img_array

def visualize_obj_error(obj_P2: np.ndarray, pred_P2: np.ndarray, canvas_size: tuple = (800, 800), sample_k: int = 1000, ranges = None):
    """
    可视化坐标回归的误差，生成三种分析图像。

    Args:
        obj_P2 (np.ndarray): 真实的坐标数组, shape=(N, 2)。
        pred_P2 (np.ndarray): 预测的坐标数组, shape=(N, 2)。
        canvas_size (tuple): 输出图像的尺寸。
        sample_k (int): 为了避免图像过于杂乱，随机采样的点的数量。

    Returns:
        dict: 一个包含三种可视化图像 (numpy 数组) 的字典。
              {'quiver': quiver_plot, 'heatmap': error_heatmap, 'histogram': error_histogram}
    """
    def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
        """
        一个更稳定和高效的 Matplotlib Figure 转 NumPy 数组的函数。
        它直接从 canvas 缓冲区读取数据，避免了文件I/O和额外的库依赖。
        """
        # 1. 触发画布的绘制（render）
        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((height, width, 4))[:,:,1:]
        return img_array

    # 如果点太多，进行随机采样
    num_points = obj_P2.shape[0]
    if num_points > sample_k:
        indices = np.random.choice(num_points, sample_k, replace=False)
        obj_P2 = obj_P2[indices]
        pred_P2 = pred_P2[indices]

    # --- 数据准备 ---
    # 计算误差向量
    error_vectors = pred_P2 - obj_P2
    # 计算每个点的误差大小（欧氏距离）
    error_magnitudes = np.linalg.norm(error_vectors, axis=1)

    if ranges is None:
        all_points = np.vstack([obj_P2, pred_P2])
        min_coords = all_points.min(axis=0)
        max_coords = all_points.max(axis=0)
        range_coords = max_coords - min_coords
        
        # 防止范围为0
        range_coords[range_coords == 0] = 1

        # 将坐标缩放到画布尺寸
        obj_scaled = (obj_P2 - min_coords) / range_coords * np.array([canvas_size[1], canvas_size[0]]) * 0.9 + 0.05 * np.array([canvas_size[1], canvas_size[0]])
        pred_scaled = (pred_P2 - min_coords) / range_coords * np.array([canvas_size[1], canvas_size[0]]) * 0.9 + 0.05 * np.array([canvas_size[1], canvas_size[0]])
    else:
        obj_scaled = obj_P2
        pred_scaled = pred_P2
        
    
    visualizations = {}

    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 10))
    ax_scatter.scatter(obj_scaled[:, 0], obj_scaled[:, 1], c='blue', s=10, alpha=0.7, label='Ground Truth')
    ax_scatter.scatter(pred_scaled[:, 0], pred_scaled[:, 1], c='red', s=10, alpha=0.7, label='Prediction')
    ax_scatter.set_title('Ground Truth vs. Prediction Scatter Plot')

    if not ranges is None:
        ax_scatter.set_xlim(ranges[0][0],ranges[0][1])
        ax_scatter.set_ylim(ranges[1][0],ranges[1][1])

    ax_scatter.set_xlabel('X coordinate')
    ax_scatter.set_ylabel('Y coordinate')
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.legend()
    ax_scatter.grid(True)
    visualizations['scatter'] = fig_to_numpy(fig_scatter)
    plt.close(fig_scatter)
    return visualizations

class Status(Enum):
    NOT_INIT = 0
    WELL_TRAINED = 1
    BAD_TRAINED = 2
