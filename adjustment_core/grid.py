import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from typing import List, Dict, Tuple

from rs_image import RSImage
from model.encoder import EncoderDino


def visualize_grid_selection(args, all_candidate_info: List[Dict], selected_diags: List[np.ndarray], ref_image: RSImage, level: int):
    """Visualize grid selection (called on rank 0)."""
    print(f"Rank 0: Generating grid selection visualization for Level {level}...")
    try:
        min_x = ref_image.corner_xys[:, 0].min()
        max_x = ref_image.corner_xys[:, 0].max()
        min_y = ref_image.corner_xys[:, 1].min()
        max_y = ref_image.corner_xys[:, 1].max()

        geo_w = max_x - min_x
        geo_h = max_y - min_y

        if geo_w == 0 or geo_h == 0:
            print("Rank 0: Invalid geographic bounds for visualization.")
            return

        vis_h = 1000
        aspect_ratio = geo_w / geo_h
        vis_w = int(vis_h * aspect_ratio)
        canvas = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255

        def geo_to_canvas(xy: np.ndarray) -> Tuple[int, int]:
            px = int((xy[0] - min_x) / geo_w * (vis_w - 1))
            py = int((max_y - xy[1]) / geo_h * (vis_h - 1))
            return (px, py)

        for info in all_candidate_info:
            diag = info['diag']
            corners_geo = np.array([diag[0], [diag[1, 0], diag[0, 1]], diag[1], [diag[0, 0], diag[1, 1]]])
            canvas_corners = [geo_to_canvas(pt) for pt in corners_geo]
            cv2.polylines(canvas, [np.array(canvas_corners, dtype=np.int32)], isClosed=True, color=(200, 200, 200), thickness=1)

        for diag in selected_diags:
            corners_geo = np.array([diag[0], [diag[1, 0], diag[0, 1]], diag[1], [diag[0, 0], diag[1, 1]]])
            canvas_corners = [geo_to_canvas(pt) for pt in corners_geo]
            cv2.polylines(canvas, [np.array(canvas_corners, dtype=np.int32)], isClosed=True, color=(0, 200, 0), thickness=2)

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
                               encoder: EncoderDino
                               ) -> Tuple[List[np.ndarray], List[Dict]]:
    """Select grids by confidence scoring and spatial NMS."""

    print(f"Rank {local_rank}: Strategy = Confidence-based selection...")

    encoder_assess = encoder.eval()

    transform_assess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    all_valid_grids_info = []

    print(f"Rank {local_rank}: Assessing quality for {len(all_candidate_diags)} candidate grids...")
    resample_size = 1024

    with torch.no_grad():
        for diag in all_candidate_diags:
            total_conf_score = 0.0
            overlapping_img_count = 0

            for img in images:
                try:
                    corners_geo = np.array([
                        diag[0], [diag[1, 0], diag[0, 1]],
                        diag[1], [diag[0, 0], diag[1, 1]]
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

                except Exception:
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
        print(f"Rank {local_rank}: Found {len(all_valid_grids_info)} valid grids. Selecting {num_to_select} using spatial selection...")

        all_valid_grids_sorted = sorted(all_valid_grids_info, key=lambda x: x['score'], reverse=True)
        candidate_grids_for_nms = all_valid_grids_sorted.copy()

        selected_grids_info_nms = []
        suppression_radius = current_window_size * 1.5

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
            num_to_backfill = num_to_select - num_selected_by_nms
            selected_diags_set = {info['diag'].tobytes() for info in selected_grids_info_nms}
            backfill_grids_info = []

            for grid_info in all_valid_grids_sorted:
                if grid_info['diag'].tobytes() not in selected_diags_set:
                    backfill_grids_info.append(grid_info)
                    if len(backfill_grids_info) == num_to_backfill:
                        break

            final_selected_grids_info = selected_grids_info_nms + backfill_grids_info
        else:
            final_selected_grids_info = selected_grids_info_nms

        selected_tasks = [info['diag'] for info in final_selected_grids_info]

    else:
        selected_tasks = [info['diag'] for info in all_valid_grids_info]

    del transform_assess

    return selected_tasks, all_valid_grids_info


def select_grids_uniformly(args,
                           all_candidate_diags: List[np.ndarray],
                           num_to_select: int,
                           local_rank: int) -> Tuple[List[np.ndarray], List[Dict]]:
    print(f"Rank {local_rank}: Strategy = Uniform selection (fast, reproducible).")

    num_candidates = len(all_candidate_diags)
    all_valid_grids_info_for_vis = []

    if num_to_select > 0 and num_candidates > num_to_select:
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
        selected_tasks = list(all_candidate_diags)
        all_valid_grids_info_for_vis = [{'diag': diag, 'center': diag.mean(axis=0), 'score': 1} for diag in selected_tasks]

    return selected_tasks, all_valid_grids_info_for_vis


def subdivide_grids(parent_diags: List[np.ndarray]) -> List[np.ndarray]:
    sub_grids = []
    for diag in parent_diags:
        min_x = np.min(diag[:, 0])
        max_x = np.max(diag[:, 0])
        min_y = np.min(diag[:, 1])
        max_y = np.max(diag[:, 1])

        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0

        sub_grids.append(np.array([[min_x, min_y], [mid_x, mid_y]]))
        sub_grids.append(np.array([[mid_x, min_y], [max_x, mid_y]]))
        sub_grids.append(np.array([[min_x, mid_y], [mid_x, max_y]]))
        sub_grids.append(np.array([[mid_x, mid_y], [max_x, max_y]]))

    return sub_grids
