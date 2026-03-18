import os
import torch
from torchvision import transforms
import numpy as np
import cv2
from typing import List, Dict

from rs_image import RSImage
from rpc import RPCModelParameterTorch
from model.encoder import EncoderDino
from utils import vis_conf, downsample_average

import adjustment_core.ddp as ddp

from adjustment_core.loop import warp_local, feature_sampling

class Window():
    def __init__(self,img:np.ndarray,local:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch):
        self.img = img 
        self.local = torch.from_numpy(local)
        self.dem = torch.from_numpy(dem)
        self.rpc = rpc
        self.feature = None
        self.conf = None
        
    
    def to_gpu(self):
        self.local = self.local.cuda()
        self.dem = self.dem.cuda()
        self.rpc.to_gpu()
        if self.feature is not None:
            self.feature = self.feature.cuda()
        if self.conf is not None:
            self.conf = self.conf.cuda()


class SharedGrid():
    def __init__(self, args, diag: np.ndarray, all_rs_images: List[RSImage], grid_id: str):
        """
        A shared geographic grid that stores all overlapping image windows.
        """
        self.args = args
        self.diag = diag
        self.id = grid_id
        self.resample_size = 1024

        self.windows: Dict[int, Window] = {}
        
        self.overlapping_image_ids: List[int] = []

        corners_geo = np.array([
            diag[0],
            [diag[1,0], diag[0,1]],
            diag[1],
            [diag[0,0], diag[1,1]]
        ])
        
        for img in all_rs_images:
            try:
                corners_sampline_i = img.xy_to_sampline(corners_geo) # [cite: rs_image_1022.py, line 85]
                
                if (corners_sampline_i.min() < 0 or 
                    corners_sampline_i[:, 0].max() > img.W or 
                    corners_sampline_i[:, 1].max() > img.H):
                    continue

                img_raw, local_i = img.resample_image_by_sampline(corners_sampline_i, (self.resample_size, self.resample_size), need_local=True)
                dem_i = img.resample_dem_by_sampline(corners_sampline_i, (self.resample_size, self.resample_size))

                self.windows[img.id] = Window(img_raw, local_i, dem_i, img.rpc)
                self.overlapping_image_ids.append(img.id)
                
            except Exception as e:
                print(f"[Grid {self.id}] Warning: Failed to create window for image {img.id}. Error: {e}")

        if len(self.overlapping_image_ids) < 2:
            raise ValueError(f"Grid {self.id} has {len(self.overlapping_image_ids)} overlapping images. Need at least 2.")
        
        if ddp.get_rank() == 0:
            self.debug_output_path = os.path.join(args.debug_output_path, f'grid_{self.id}')
            os.makedirs(self.debug_output_path, exist_ok=True)
            for img_id in self.overlapping_image_ids:
                cv2.imwrite(os.path.join(self.debug_output_path, f'img_raw_{img_id}.png'), self.windows[img_id].img)


    @torch.no_grad()
    def extract_features_sequentially(self, encoder: EncoderDino, local_rank: int):
        """
        Extract features for all overlapping images in this grid.
        """
        encoder_eval = encoder.eval() 
        
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                    ])
        
        if ddp.get_rank() == 0:
            os.makedirs(self.debug_output_path, exist_ok=True)
        
        for img_id in self.overlapping_image_ids:
            window = self.windows[img_id]
            
            img_tensor = transform(window.img)[None].cuda(local_rank)
            feature, conf = encoder_eval(img_tensor)
            
            h, w = feature.shape[-2:]
            window.feature = feature[0].permute(1,2,0).flatten(0,1)
            window.conf = conf.squeeze().flatten(0,1)
            window.local = downsample_average(window.local, encoder.SAMPLE_FACTOR).flatten(0,1)
            window.dem = downsample_average(window.dem, encoder.SAMPLE_FACTOR).flatten(0,1)

            if ddp.get_rank() == 0:
                original_img_for_vis = window.img.copy()
            
            del window.img

            window.to_gpu() 

            if ddp.get_rank() == 0:
                feat_vis = window.feature.cpu().numpy().reshape(h,w,-1)
                conf_vis = window.conf.cpu().numpy().reshape(h,w)
                
                feat_pca = (feat_vis.reshape(-1, feat_vis.shape[-1]) @ (torch.pca_lowrank(window.feature, q=3)[2]).cpu().numpy())
                feat_pca = feat_pca.reshape(h, w, 3)
                feat_pca = (feat_pca - feat_pca.min(axis=(0,1))) / (feat_pca.max(axis=(0,1)) - feat_pca.min(axis=(0,1)) + 1e-6)
                cv2.imwrite(os.path.join(self.debug_output_path, f'feat_vis_{img_id}.png'), (feat_pca * 255).astype(np.uint8))
                
                conf_cont, conf_div = vis_conf(conf_vis, original_img_for_vis, encoder.SAMPLE_FACTOR, div=self.args.conf_threshold)
                cv2.imwrite(os.path.join(self.debug_output_path, f'conf_cont_{img_id}.png'), cv2.cvtColor(conf_cont,cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(self.debug_output_path, f'conf_div_{img_id}.png'), cv2.cvtColor(conf_div,cv2.COLOR_RGB2BGR))

    def calculate_all_pairs_loss(self, model_ddp, images: List[RSImage], local_rank: int) -> torch.Tensor:
        """
         Compute pairwise asymmetric losses for all image pairs in this grid.
        """
        import itertools
        
        grid_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_pairs_in_grid = 0
        
        for (i, j) in itertools.combinations(self.overlapping_image_ids, 2):
            
            A_i = model_ddp.module.get_affine(i)
            A_j = model_ddp.module.get_affine(j)
            
            window_i = self.windows[i]
            window_j = self.windows[j]
            
            rpc_i = images[i].rpc
            rpc_j = images[j].rpc
            
            warp_j_to_i = warp_local(window_j.local.float(), window_j.dem, rpc_j, rpc_i, A_j)
            feat_j_in_i, conf_j_in_i, valid_j = feature_sampling(window_i.feature.float(), window_i.conf.float(), window_i.local.float(), warp_j_to_i, k = self.args.kmin_k)

            loss_a = torch.tensor(0.0, device=local_rank)
            if feat_j_in_i is not None:
                feat_j_orig = window_j.feature[valid_j].float()
                weight_a = (window_j.conf[valid_j].float() + conf_j_in_i) * .5
                loss_a = (torch.norm(feat_j_orig - feat_j_in_i, dim=-1) * weight_a).mean() * 10000.

            pair_loss = loss_a
            
            if not torch.isnan(pair_loss) and not torch.isinf(pair_loss) and pair_loss > 0:
                grid_total_loss = grid_total_loss + pair_loss
                num_valid_pairs_in_grid += 1
            else:
                if pair_loss > 0:
                    print(f"[Rank{local_rank}]: Detect invalid loss:{pair_loss.item()} in Grid {self.id} for pair ({i}, {j})")


        if num_valid_pairs_in_grid > 0:
            return grid_total_loss / num_valid_pairs_in_grid 
        else:
            return torch.tensor(0.0, device=local_rank)
