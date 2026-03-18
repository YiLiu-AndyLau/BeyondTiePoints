import os
import argparse
import random
import itertools
import torch
import numpy as np
import warnings
import json
from typing import List

from torch.nn.parallel import DistributedDataParallel as DDP

from model.encoder import EncoderDino
from utils import find_grids, str2bool

import adjustment_core.ddp as ddp
import adjustment_core.model as adj_model
import adjustment_core.data as adj_data
import adjustment_core.loop as adj_loop
import adjustment_core.grid as adj_grid
import adjustment_core.validation as adj_validation
import adjustment_core.utils as adj_utils

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')

    parser.add_argument('--dino_path', type=str, default='weights',
                        help='folder containing DINO backbone weights')

    parser.add_argument('--adapter_path', type=str, default='weights/pretrain_swt_cnn_r2_0409_large/adapter.pth',
                        help='path to pre-trained adapter weights')

    parser.add_argument('--use_ddp', type=str, default='auto', choices=['auto', 'true', 'false'],
                        help='whether to use distributed data parallel: auto/true/false')

    parser.add_argument('--max_lr', type=float, default=0.0001,
                        help='highest learning rate')

    parser.add_argument('--max_iter', type=int, default=1000)

    parser.add_argument('--conf_threshold', type=float, default=.5)

    parser.add_argument('--kmin_k', type=int, default=16)

    parser.add_argument('--window_size', type=int, default=2000,
                        help='INITIAL window size in meter(m) for the coarsest level.')

    parser.add_argument('--select_imgs', type=str, default='0,1')

    parser.add_argument('--grid_offset_x', type=float, default=0)
    parser.add_argument('--grid_offset_y', type=float, default=0)
    parser.add_argument('--grid_num', type=int, default=1)

    parser.add_argument('--max_grid_num', type=int, default=1000,
                        help='Maximum number of grids per level for level > 0. No cap when <= 0.')

    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping (e.g., 100 iterations)')

    parser.add_argument('--min_loss_threshold', type=float, default=1e-4,
                        help='Minimum improvement threshold for min_loss to reset patience (e.g., 1e-4)')

    parser.add_argument('--num_levels', type=int, default=1,
                        help='Total number of pyramid levels for adjustment (default: 1).')

    parser.add_argument('--vis_resolution', type=float, default=1.0,
                        help='Resolution (in meters) for output orthophotos and checkerboards.')

    parser.add_argument('--select_grid_by_conf', action='store_true',
                        help='If set, use slow confidence-based grid selection. If not set, use fast uniform selection.')

    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Unique ID for the experiment, used for output folder naming.')

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_adapter', type=str2bool, default=True)
    parser.add_argument('--use_conf', type=str2bool, default=True)

    args = parser.parse_args()

    local_rank, world_size, use_ddp = ddp.setup_distributed(args.use_ddp)

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.experiment_id:
        args.debug_output_path = os.path.join(args.root, f'output_{args.experiment_id}')
    else:
        args.debug_output_path = os.path.join(args.root, 'debug_output')

    if local_rank == 0:
        os.makedirs(args.debug_output_path, exist_ok=True)

    images = adj_utils.load_imgs_bundle(args)
    if len(images) < 2:
        if local_rank == 0:
            print("Error: Found less than 2 images. Bundle adjustment requires at least 2.")
        ddp.destroy_distributed()
        exit()

    overlapping_pairs = []

    if local_rank == 0:
        print("Rank 0: Finding overlapping pairs (for final validation)...")
        overlapping_pairs = adj_utils.find_overlapping_pairs(args, images)

        print("\nStarting initial error check on Rank 0...")
        initial_report = adj_validation.calculate_error_report(
            images, overlapping_pairs, verbose=True
        )
        if initial_report['count'] == 0:
            print("No valid tie points found. Initial error check skipped.")

    selected_diags_for_level = []

    for level in range(args.num_levels):

        current_window_size = args.window_size / (2 ** level)
        all_tasks = []

        if local_rank == 0:
            print("\n" + "=" * 50)
            print(f"--- Starting Pyramid Level {level + 1} / {args.num_levels} ---")
            print(f"--- Current Window Size: {current_window_size:.2f} m ---")
            print("=" * 50 + "\n")

        encoder_level = EncoderDino(
            os.path.join(args.dino_path, 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),
            upsample_times=0,
            use_adapter=args.use_adapter,
            use_conf=args.use_conf
        )
        encoder_level.load_adapter(args.adapter_path)
        if torch.cuda.is_available():
            encoder_level.cuda(local_rank)
        encoder_level.eval()

        if local_rank == 0:
            print(f"Encoder Loaded for Level {level + 1}")

        if local_rank == 0:
            if level == 0:
                print("Rank 0: Level 0. Finding, assessing, and selecting initial grids...")

                all_corners = np.stack([img.corner_xys for img in images], axis=0)
                all_common_diags = find_grids(all_corners, current_window_size,
                                              offset_x=args.grid_offset_x,
                                              offset_y=args.grid_offset_y)
                print(f"Rank 0: Found {len(all_common_diags)} total common grids.")

                if args.select_grid_by_conf:
                    all_tasks, all_valid_grids_info = adj_grid.select_grids_by_confidence(
                        args, all_common_diags, args.grid_num, images, current_window_size, local_rank,
                        encoder=encoder_level
                    )
                    adj_grid.visualize_grid_selection(args, all_valid_grids_info, all_tasks, images[0], level)

                else:
                    all_tasks, all_valid_grids_info_for_vis = adj_grid.select_grids_uniformly(
                        args, all_common_diags, args.grid_num, local_rank
                    )
                    adj_grid.visualize_grid_selection(args, all_valid_grids_info_for_vis, all_tasks, images[0], level)

                selected_diags_for_level = all_tasks

            else:
                print(f"Rank 0: Level {level + 1}. Subdividing {len(selected_diags_for_level)} grids from previous level.")

                subdivided_tasks = adj_grid.subdivide_grids(selected_diags_for_level)
                num_subdivided = len(subdivided_tasks)

                if args.max_grid_num > 0 and num_subdivided > args.max_grid_num:
                    print(f"Rank 0: Subdivided into {num_subdivided} grids. Capping at {args.max_grid_num} using selection strategy...")

                    if args.select_grid_by_conf:
                        all_tasks, all_valid_grids_info = adj_grid.select_grids_by_confidence(
                            args, subdivided_tasks, args.max_grid_num, images, current_window_size, local_rank,
                            encoder=encoder_level
                        )
                        adj_grid.visualize_grid_selection(args, all_valid_grids_info, all_tasks, images[0], level)

                    else:
                        all_tasks, all_valid_grids_info_for_vis = adj_grid.select_grids_uniformly(
                            args, subdivided_tasks, args.max_grid_num, local_rank
                        )
                        adj_grid.visualize_grid_selection(args, all_valid_grids_info_for_vis, all_tasks, images[0], level)

                else:
                    all_tasks = subdivided_tasks
                    print(f"Rank 0: Created {len(all_tasks)} new sub-grids (no cap applied).")

                selected_diags_for_level = all_tasks

            random.shuffle(all_tasks)
            print(f"Rank 0: Final task list for level {level + 1} has {len(all_tasks)} grids.")

        tasks_to_broadcast = [all_tasks] if local_rank == 0 else [None]
        ddp.broadcast_object_list(tasks_to_broadcast, src=0)
        all_tasks = tasks_to_broadcast[0]

        pairs_to_broadcast = [overlapping_pairs] if local_rank == 0 else [None]
        ddp.broadcast_object_list(pairs_to_broadcast, src=0)
        overlapping_pairs = pairs_to_broadcast[0]

        my_tasks = all_tasks[local_rank::world_size]

        local_shared_grids: List[adj_data.SharedGrid] = []

        if local_rank == 0:
            print(f"[Rank {local_rank}] Level {level + 1}: Total grids: {len(all_tasks)}, assigned: {len(my_tasks)}.")

        for idx, diag in enumerate(my_tasks):
            global_grid_id = f"L{level}_R{local_rank}_{idx}"
            try:
                grid = adj_data.SharedGrid(args, diag, images, global_grid_id)
                grid.extract_features_sequentially(encoder_level, local_rank)
                local_shared_grids.append(grid)
                print(f"[Rank {local_rank}] Grid {global_grid_id} (with {len(grid.overlapping_image_ids)} images) created and features extracted on cuda:{local_rank}")
            except Exception as e:
                print(f"[Rank {local_rank}] !! FAILED to create grid {global_grid_id}. Error: {e}")

        model = adj_model.BundleAffineModel(len(images)).to(local_rank if torch.cuda.is_available() else 'cpu')
        if use_ddp:
            model_ddp = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
        else:
            model_ddp = adj_loop.SingleDeviceModel(model)

        all_R_params = [m.R for m in model_ddp.module.models]
        all_T_params = [m.T for m in model_ddp.module.models]

        optimizer_r = None
        optimizer_t = None
        scheduler_r = None
        scheduler_t = None

        if all_R_params:
            optimizer_r = torch.optim.Adam(all_R_params, lr=args.max_lr * 1e-5 / (10 ** level))
            scheduler_r = torch.optim.lr_scheduler.OneCycleLR(optimizer_r, max_lr=args.max_lr * 1e-5 / (10 ** level),
                                                               total_steps=args.max_iter, pct_start=20 / args.max_iter)

        if all_T_params:
            optimizer_t = torch.optim.Adam(all_T_params, lr=args.max_lr / (10 ** level))
            scheduler_t = torch.optim.lr_scheduler.OneCycleLR(optimizer_t, max_lr=args.max_lr / (10 ** level),
                                                               total_steps=args.max_iter, pct_start=20 / args.max_iter)

        best_model_state = []

        if optimizer_r is None and optimizer_t is None and local_rank == 0:
            print(f"Warning: No parameters to optimize for level {level + 1}. Skipping optimization.")
        else:
            best_model_state = adj_loop.fit_affine_bundle(
                args,
                local_shared_grids,
                images,
                model_ddp,
                optimizer_r,
                optimizer_t,
                scheduler_r,
                scheduler_t,
                local_rank,
                world_size,
                patience=args.patience,
                current_level=level
            )

        ddp.barrier()

        state_to_broadcast = [best_model_state] if local_rank == 0 else [None]
        ddp.broadcast_object_list(state_to_broadcast, src=0)
        best_model_state = state_to_broadcast[0]

        if local_rank == 0:
            print(f"\n[All Ranks] Applying (best) adjustments from Level {level + 1} to RPC models...")

        if best_model_state:
            with torch.no_grad():
                for i, state in enumerate(best_model_state):
                    if i < len(model_ddp.module.models):
                        model_ddp.module.models[i].R.data.copy_(state['R'])
                        model_ddp.module.models[i].T.data.copy_(state['T'])
        else:
            if local_rank == 0:
                print(f"Warning: No best model state found for Level {level + 1}. Using final iteration state.")

        for i in range(1, len(images)):
            final_A_i_level = model_ddp.module.get_affine(i).detach()

            if local_rank == 0:
                print(f"Level {level + 1} Affine Delta for image {i}: \n {final_A_i_level.cpu().numpy()}")
            images[i].rpc.Update_Adjust(final_A_i_level)

        if local_rank == 0:
            print(f"Rank 0: Level {level + 1} adjustments applied by all ranks.")
            print(f"\n--- Error Report After Level {level + 1} ---")
            adj_validation.calculate_error_report(
                images, overlapping_pairs, verbose=True
            )

        if local_rank == 0:
            print(f"\n[All Ranks] Starting parallel visualization for Level {level + 1} (Res: {args.vis_resolution}m)...")

        vis_resolution = args.vis_resolution

        for grid in local_shared_grids:
            grid_ortho_cache = {}
            grid_vis_path = os.path.join(args.debug_output_path, f"vis_{grid.id}")
            os.makedirs(grid_vis_path, exist_ok=True)

            for img_id in grid.overlapping_image_ids:
                rs_image = images[img_id]
                try:
                    ortho_array, transform = adj_utils.orthorectify_patch_mercator(
                        rs_image,
                        grid.diag,
                        resolution=vis_resolution,
                        output_path=None
                    )
                    grid_ortho_cache[img_id] = (ortho_array, transform)
                except Exception as e:
                    print(f"[Rank {local_rank}] FAILED orthorectification for {grid.id}/img_{img_id}. Error: {e}")

            for (i, j) in itertools.combinations(grid.overlapping_image_ids, 2):
                if i in grid_ortho_cache and j in grid_ortho_cache:
                    ortho_i, transform_i = grid_ortho_cache[i]
                    ortho_j, transform_j = grid_ortho_cache[j]

                    checker_output_path = os.path.join(grid_vis_path, f"checker_{i}_vs_{j}.png")
                    try:
                        adj_utils.create_checkerboard(
                            ortho_i, ortho_j,
                            transform_i,
                            checker_output_path,
                            block_size=50
                        )
                    except Exception as e:
                        print(f"[Rank {local_rank}] FAILED checkerboard for {grid.id}/({i},{j}). Error: {e}")

        ddp.barrier()
        if local_rank == 0:
            print(f"[All Ranks] Visualization for Level {level + 1} complete.")
            print(f"\n[All Ranks] Baking RPC adjustments from Level {level + 1}...")

        for img in images:
            img.rpc.Merge_Adjust()

        if local_rank == 0:
            rpc_save_path = os.path.join(args.debug_output_path, f"rpc_level_{level}")
            os.makedirs(rpc_save_path, exist_ok=True)
            print(f"[Rank 0] Saving baked RPCs to {rpc_save_path}...")

            for img in images:
                save_name = f"image_{img.id}_L{level}_baked.txt"
                output_rpc_path = os.path.join(rpc_save_path, save_name)
                try:
                    img.rpc.save_rpc_to_file(output_rpc_path)
                except Exception as e:
                    print(f"[Rank 0] FAILED to save RPC for image {img.id} to {output_rpc_path}. Error: {e}")

            print(f"[Rank 0] RPCs for Level {level + 1} saved.")

        ddp.barrier()

        del local_shared_grids, model, model_ddp, optimizer_r, optimizer_t, scheduler_r, scheduler_t, encoder_level
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ddp.barrier()

    if local_rank == 0:
        print("\n" + "=" * 50)
        print(f"--- Multi-level Bundle Adjustment Finished ({args.num_levels} levels) ---")
        print("=" * 50 + "\n")

        final_report = adj_validation.calculate_error_report(
            images, overlapping_pairs, verbose=False
        )

        results_dict = {}
        if final_report['count'] > 0:
            results_dict['mean_error'] = final_report['mean']
            results_dict['median_error'] = final_report['median']
            results_dict['max_error'] = final_report['max']
            results_dict['rmse'] = final_report['rmse']
            results_dict['<1m'] = final_report['<1m_percent']
            results_dict['<3m'] = final_report['<3m_percent']
            results_dict['<5m'] = final_report['<5m_percent']
            results_dict['total_tie_points'] = final_report['count']
        else:
            keys = ['mean_error', 'median_error', 'max_error', 'rmse', '<1m', '<3m', '<5m']
            results_dict = {k: np.nan for k in keys}
            results_dict['total_tie_points'] = 0

        json_path = os.path.join(args.debug_output_path, 'final_results.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
            print(f"[Rank 0] Saved results to {json_path}")
        except Exception as e:
            print(f"[!!] [Rank 0] Failed to save results.json to {json_path}. Error: {e}")

    ddp.destroy_distributed()
