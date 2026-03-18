import os
import time
import numpy as np
import cv2
from typing import List, Tuple, Dict
from scipy.interpolate import RegularGridInterpolator

import rasterio
from rasterio.transform import from_origin
from pyproj import CRS

import adjustment_core.ddp as ddp

from rs_image import RSImage


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def orthorectify_patch_mercator(rs_image: RSImage, 
                              grid_diag: np.ndarray, 
                              resolution: float, 
                              output_path: str = None) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Orthorectify one RSImage patch on a Mercator grid using adjusted RPC.
    If output_path is None, no file is written.
    """
    
    all_x = grid_diag[:, 0]
    all_y = grid_diag[:, 1]
    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)
    
    out_W = int(np.ceil((max_x - min_x) / resolution))
    out_H = int(np.ceil((max_y - min_y) / resolution))
    
    if out_W <= 0 or out_H <= 0:
        raise ValueError(f"Output size is zero or negative (W:{out_W}, H:{out_H})。Please check grid_diag and resolution.Grid Diag: {grid_diag}")

    transform = from_origin(min_x, max_y, resolution, resolution)
    
    out_x_coords = np.linspace(min_x + resolution / 2, max_x - resolution / 2, out_W)
    out_y_coords = np.linspace(max_y - resolution / 2, min_y + resolution / 2, out_H)
    
    out_xx, out_yy = np.meshgrid(out_x_coords, out_y_coords)
    
    H_src, W_src = rs_image.image.shape[:2]
    lines_src = np.arange(H_src)
    samps_src = np.arange(W_src)
    
    image_interpolator_r = RegularGridInterpolator((lines_src, samps_src), rs_image.image[..., 0], method='linear', bounds_error=False, fill_value=0)
    image_interpolator_g = RegularGridInterpolator((lines_src, samps_src), rs_image.image[..., 1], method='linear', bounds_error=False, fill_value=0)
    image_interpolator_b = RegularGridInterpolator((lines_src, samps_src), rs_image.image[..., 2], method='linear', bounds_error=False, fill_value=0)


    ortho_image = np.zeros((out_H, out_W, 3), dtype=rs_image.image.dtype)
    
    block_size = 1024
    for i in range(0, out_H, block_size):
        i_end = min(i + block_size, out_H)
        for j in range(0, out_W, block_size):
            j_end = min(j + block_size, out_W)
            
            block_xx = out_xx[i:i_end, j:j_end]
            block_yy = out_yy[i:i_end, j:j_end]
            
            xy_points = np.stack([block_xx.ravel(), block_yy.ravel()], axis=-1)
            
            try:
                sampline_pred = rs_image.xy_to_sampline(xy_points) 
            except Exception as e:
                print(f"Warning: xy_to_sampline projection failed (Grid: {output_path}): {e}")
                continue 
                
            points_to_sample = np.stack([sampline_pred[:, 1], sampline_pred[:, 0]], axis=-1) # (line, samp)
            
            pixel_vals_r = image_interpolator_r(points_to_sample).reshape(block_xx.shape)
            pixel_vals_g = image_interpolator_g(points_to_sample).reshape(block_xx.shape)
            pixel_vals_b = image_interpolator_b(points_to_sample).reshape(block_xx.shape)
            ortho_image[i:i_end, j:j_end] = np.stack([pixel_vals_r, pixel_vals_g, pixel_vals_b], axis=-1).astype(rs_image.image.dtype)

    if output_path is not None:
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=out_H,
            width=out_W,
            count=3,
            dtype=ortho_image.dtype,
            crs=CRS.from_epsg(3857), # Web Mercator
            transform=transform
        ) as dst:
            dst.write(ortho_image[..., 0], 1)
            dst.write(ortho_image[..., 1], 2)
            dst.write(ortho_image[..., 2], 3)
            
    return ortho_image, transform

def create_checkerboard(ortho1: np.ndarray, 
                        ortho2: np.ndarray, 
                        transform: rasterio.Affine,
                        output_path: str, 
                        block_size: int = 50):
    """
    Merge two aligned orthophotos into a checkerboard image.
    """
    if ortho1.shape != ortho2.shape:
        return

    H, W = ortho1.shape[:2]
    checkerboard_img = np.zeros_like(ortho1)

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            i_block = (i // block_size) % 2
            j_block = (j // block_size) % 2
            
            if i_block == j_block:
                checkerboard_img[i:min(i+block_size, H), j:min(j+block_size, W)] = \
                    ortho1[i:min(i+block_size, H), j:min(j+block_size, W)]
            else:
                checkerboard_img[i:min(i+block_size, H), j:min(j+block_size, W)] = \
                    ortho2[i:min(i+block_size, H), j:min(j+block_size, W)]
    
    try:
        if checkerboard_img.ndim == 3 and checkerboard_img.shape[2] == 3:
            checkerboard_img_bgr = cv2.cvtColor(checkerboard_img, cv2.COLOR_RGB2BGR)
        else:
            checkerboard_img_bgr = checkerboard_img

        cv2.imwrite(output_path, checkerboard_img_bgr)
    except Exception as e:
        print(f"create_checkerboard failed: {e}")


def load_imgs_bundle(args) -> List[RSImage]:
    """Load all images in the bundle."""
    base_path = os.path.join(args.root, 'adjust_images')
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if args.select_imgs != '-1':
        select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    else:
        select_img_idxs = range(len(img_folders))
    img_folders = [img_folders[i] for i in select_img_idxs]
    
    images:List[RSImage] = []
    if ddp.get_rank() == 0:
        print(f"[Rank {ddp.get_rank()}] Found {len(img_folders)} image folders. Loading all...")
        
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        try:
            images.append(RSImage(args, img_path, idx))
            images[-1].tie_points_height = images[0].tie_points_height
            if ddp.get_rank() == 0:
                print(f"[Rank {ddp.get_rank()}] Loaded image {idx} from {folder}.")
        except Exception as e:
            print(f"[Rank {ddp.get_rank()}] Failed to load image {idx} from {folder}: {e}")
    

    if ddp.get_rank() == 0:
        print(f"[Rank {ddp.get_rank()}] Successfully loaded {len(images)} images into memory.")
    return images

def find_overlapping_pairs(args, images: List[RSImage]) -> List[Tuple[int, int]]:
    """Find overlapping image pairs by geographic bbox intersection."""
    bboxes = []
    for img in images:
        min_x = img.corner_xys[:, 0].min()
        max_x = img.corner_xys[:, 0].max()
        min_y = img.corner_xys[:, 1].min()
        max_y = img.corner_xys[:, 1].max()
        bboxes.append((min_x, min_y, max_x, max_y))

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            b1 = bboxes[i]
            b2 = bboxes[j]
            
            is_disjoint = (b1[2] < b2[0] or  # b1.maxX < b2.minX
                           b1[0] > b2[2] or  # b1.minX > b2.maxX
                           b1[3] < b2[1] or  # b1.maxY < b2.minY
                           b1[1] > b2[3])   # b1.minY > b2.maxY
            
            if not is_disjoint:
                pairs.append((i, j))
    
    print(f"Found {len(pairs)} overlapping pairs for validation.")
    return pairs

class TqdmLogger:
    """
    Unified logger (active only on rank 0).
    """
    def __init__(self, args, total_iters: int, level: int, local_rank: int):
        self.is_rank_0 = (local_rank == 0)
        self.total_iters = total_iters
        self.level = level
        self.num_levels = args.num_levels
        self.pbar = None
        
        if not self.is_rank_0:
            return

        self.start_time = time.time()
            
    def update(self, iter: int, loss: float, metrics_dict: Dict, patience: int, max_patience: int):
        if not self.is_rank_0:
            return

        if (iter + 1) % 10 == 0:
            elapsed_time_sec = time.time() - self.start_time
            elapsed_time_str = format_time(elapsed_time_sec)
            avg_iter_time = elapsed_time_sec / (iter + 1)
            remaining_iter = self.total_iters - (iter + 1)
            remaining_time_sec = avg_iter_time * remaining_iter
            remaining_time_str = format_time(remaining_time_sec)

            min_metric_log_str = ""
            if 'min_met' in metrics_dict:
                min_metric_log_str = f"min_met:{metrics_dict['min_met']}"

            lr_t_str = f"{metrics_dict.get('lr_t', 0):.2e}"
            lr_r_str = f"{metrics_dict.get('lr_r', 0):.2e}"

            print(f"Lvl:{self.level + 1}/{self.num_levels} iter:{iter+1}/{self.total_iters} \t loss:{loss:.4f} {min_metric_log_str} \t pat:{patience}/{max_patience} \t lr_t:{lr_t_str}  lr_r:{lr_r_str} \t elapsed:{elapsed_time_str}  eta:{remaining_time_str}")

    def close(self):
        return
