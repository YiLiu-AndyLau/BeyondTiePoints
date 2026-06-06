import torch.nn as nn 
import torch 
import torch.nn.functional as F
import numpy as np
import math
from rpc import RPCModelParameterTorch
from utils import project_mercator,mercator2lonlat
import time
from typing import List

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        diff = torch.norm(input - target,dim=1)

        loss = torch.where(diff < self.beta,
                           0.5 * diff ** 2 / self.beta,
                           diff - 0.5 * self.beta)

        return loss

def tanh_clamp(loss,r,max):
    w = np.sqrt(1 - r**2)
    t = w * max + 1
    return t * torch.tanh(loss / t)

def get_dis_matrix(points_a:torch.Tensor,points_b:torch.Tensor):
    return torch.cdist(points_a,points_b,p=2)

@torch.no_grad()
def get_near_points(points1,points2,batch_size,threshold):
    point_num = len(points1)
    min_dis = torch.full((point_num,),1e9,device=points1.device,dtype=points1.dtype)
    min_dis_idx = torch.full((point_num,),-1,device=points1.device,dtype=int)
    batch_num = int(np.ceil(point_num / batch_size))
    for b1 in range(batch_num):
        for b2 in range(batch_num):
            dis = torch.cdist(points1[b1 * batch_size : (b1 + 1) * batch_size],points2[b2 * batch_size : (b2 + 1) * batch_size])
            min_dis_batch,min_dis_idx_batch = torch.min(dis,dim=1)

            update_mask = min_dis_batch < min_dis[b1 * batch_size : (b1 + 1) * batch_size]

            min_dis_idx[b1 * batch_size : (b1 + 1) * batch_size][update_mask] = min_dis_idx_batch[update_mask] + b2 * batch_size
            min_dis[b1 * batch_size : (b1 + 1) * batch_size][update_mask] = min_dis_batch[update_mask]

    valid_mask = min_dis < threshold
    anchor_points = torch.arange(point_num,device=points1.device,dtype=int)[valid_mask]
    positive_points = min_dis_idx[valid_mask]
    return anchor_points,positive_points


@torch.no_grad()
def get_far_points(anchor_points,candidates,threshold=100):
    anchor_points_num = len(anchor_points)
    sample_idx = torch.randperm(len(candidates),device=candidates.device)[:anchor_points_num]
    while len(sample_idx) < anchor_points_num:
        sample_idx = torch.cat([sample_idx,torch.randperm(len(candidates),device=candidates.device)[:anchor_points_num - len(sample_idx)]])
    dis = torch.norm(anchor_points - candidates[sample_idx],dim=1)
    invalid_idx = torch.where(dis<threshold)[0]
    count = 0
    while(len(invalid_idx) > 0 and count < 3):
        append_idx = torch.randperm(len(candidates),device=candidates.device)[:len(invalid_idx)]
        while len(append_idx) < len(invalid_idx):
            append_idx = torch.cat([append_idx,torch.randperm(len(candidates),device=candidates.device)[:len(invalid_idx) - len(append_idx)]])
        sample_idx[invalid_idx] = append_idx
        dis = torch.norm(anchor_points - candidates[sample_idx],dim=1)
        invalid_idx = torch.where(dis<threshold)[0]
        count += 1
    return sample_idx


def affine_loss(local:torch.Tensor,pred:torch.Tensor,conf:torch.Tensor):
    def affine_trans(ori, dst):
        ones = torch.ones(ori.shape[0], 1, device=ori.device)
        ori_homogeneous = torch.cat([ori, ones], dim=1)  # (N, 3)
        X = ori_homogeneous  # (N, 3)
        B = dst  # (N, 2)

        XTX = torch.matmul(X.transpose(0, 1), X)  # (3, 3)
        XTX_inv = torch.inverse(XTX)  # (3, 3)
        X_pinv = torch.matmul(XTX_inv, X.transpose(0, 1))  # (3, N)

        transform_matrix = torch.matmul(X_pinv, B)  # (3, 2)
        
        ori_trans = torch.matmul(ori_homogeneous, transform_matrix)  # (N, 2)
        
        return ori_trans
    
    dis = torch.norm(local + (torch.mean(pred,dim=0)[None] - torch.mean(local,dim=0)[None]) - pred,dim=-1)
    valid_idx = (conf > .5) & (dis < dis.mean() + dis.std())
    local = local[valid_idx]
    pred = pred[valid_idx]
    
    reg = affine_trans(local,pred)
    loss_affine = torch.norm(reg - local,dim=-1).mean()
    return loss_affine

def conf_norm(conf):
    conf = conf.clone().detach()
    conf_norm = conf - conf.mean() + 1.
    return torch.clip(conf_norm,min=0.)

def residual2conf(residual,t = 6.):
    conf = torch.full(residual.shape,.5,device=residual.device,dtype=residual.dtype)
    conf[residual > t] = .1
    conf[(residual < t) & (residual >= 0)] = .9
    conf[residual < 0] = .1
    return conf

def detect_nan(items:list):
    for item in items:
        if torch.isnan(item).any():
            return True
    return False




class CriterionFinetune(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self,epoch,max_epoch,
                feat1_PD,feat2_PD,
                pred1_P3,pred2_P3,
                conf1_P,conf2_P,
                obj1_P3,obj2_P3,
                residual1_P,residual2_P,
                res_mid,
                only_decoder:bool,
                H,W
                ):
        
        residual1_P,residual2_P = residual1_P.to(torch.float32),residual2_P.to(torch.float32)

        P = H*W
        residual_threshold = res_mid
        conf1_gt_P = residual2conf(residual1_P,residual_threshold)
        conf2_gt_P = residual2conf(residual2_P,residual_threshold)
        
        conf_valid1 = residual1_P >= 0
        conf_valid2 = residual2_P >= 0
        weights1_P = conf_norm(conf1_gt_P)
        weights2_P = conf_norm(conf2_gt_P)


        pairs = np.random.randint(0,len(pred1_P3)-1,size=(1000,2))
        invalid_pair_mask = pairs[:, 0] == pairs[:, 1]
        pairs[invalid_pair_mask,1] = (pairs[invalid_pair_mask, 1] + 1) % len(pred1_P3)

        loss_obj = .5 * (torch.norm(pred1_P3[:,:2] - obj1_P3[:,:2],dim=-1) * weights1_P).mean() + .5 * (torch.norm(pred2_P3[:,:2] - obj2_P3[:,:2],dim=-1) * weights2_P).mean()
        loss_height = .5 * (torch.abs(pred1_P3[:,2] - obj1_P3[:,2]) * weights1_P).mean() + .5 * (torch.abs(pred2_P3[:,2] - obj2_P3[:,2]) * weights2_P).mean()

        offset_gt_pair_1 = obj1_P3[pairs[:,0]] - obj1_P3[pairs[:,1]]
        offset_gt_pair_2 = obj2_P3[pairs[:,0]] - obj2_P3[pairs[:,1]]
        offset_pred_pair_1 = pred1_P3[pairs[:,0]] - pred1_P3[pairs[:,1]]
        offset_pred_pair_2 = pred2_P3[pairs[:,0]] - pred2_P3[pairs[:,1]]
        offset_dis_1 = torch.norm(torch.abs(offset_pred_pair_1 - offset_gt_pair_1),dim=-1)
        offset_dis_2 = torch.norm(torch.abs(offset_pred_pair_2 - offset_gt_pair_2),dim=-1)
        loss_relative = .5 * offset_dis_1.mean() + .5 * offset_dis_2.mean()

        loss_conf = .5 * self.bce(conf1_P[conf_valid1],conf1_gt_P[conf_valid1]).mean() + .5 * self.bce(conf2_P[conf_valid2],conf2_gt_P[conf_valid2]).mean()
        loss_conf = loss_conf * 1000 * min(1.,epoch / 3.)


        shift_amount1 = torch.randint(low=1,high = P // 2,size=(1,))[0].item()
        shift_amount2 = torch.randint(low=-P // 2,high = -1,size=(1,))[0].item()
        feat1_negative = torch.roll(feat1_PD,shift_amount1,dims=0)
        feat2_negative = torch.roll(feat2_PD,shift_amount2,dims=0)
        simi_positive = torch.concatenate([torch.sum(feat1_PD * feat2_PD,dim=1),
                                           torch.sum(feat2_PD * feat1_PD,dim=1)])
        simi_negative = torch.concatenate([torch.sum(feat1_PD * feat1_negative,dim=1),
                                           torch.sum(feat2_PD * feat2_negative,dim=1)])


        loss_feat = torch.clip(simi_negative - simi_positive + .5,min=0.).mean() * 10000 
        loss_feat_weight = 1.


        if only_decoder:
            loss = loss_obj + loss_height + loss_conf * 0. + loss_feat * 0. 
        else:
            loss = loss_obj + loss_height + loss_conf + loss_feat * loss_feat_weight
        return loss,loss_obj,loss_height,loss_relative,loss_conf,loss_feat,residual_threshold,simi_positive.mean(),simi_negative.mean()
        
class CriterionFinetuneDis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                pred1_P3,pred2_P3,
                residual1_P,residual2_P,
                residual_threshold,
                ):
        pred1_P3,pred2_P3,residual1_P,residual2_P = \
           pred1_P3.to(torch.float32),pred2_P3.to(torch.float32),residual1_P.to(torch.float32),residual2_P.to(torch.float32)
        robust_mask = (residual1_P <= residual_threshold) & (residual2_P <= residual_threshold)

        dis_obj = torch.norm(pred1_P3[robust_mask,:2] - pred2_P3[robust_mask,:2],dim=-1)
        dis_height = torch.abs(pred1_P3[robust_mask,2] - pred2_P3[robust_mask,2]) * 100


        dis_obj = dis_obj.mean()
        dis_height = dis_height.mean()

        loss_dis = dis_obj + dis_height
        return loss_dis,dis_obj,dis_height

        
    
class CriterionTrainOneImg(nn.Module):

    def __init__(self):
        super().__init__()
        self.height_tolerance = 5
        self.clamp_max = 1000
        self.loss_height_weight = 10
        self.conf_clamp_rate = 2      

    def forward(self,epoch,max_epoch,pred,conf,photo_gt,dem_gt,rpc:RPCModelParameterTorch):
        
        pred,conf,photo_gt,dem_gt = [i.to(torch.float64) for i in [pred,conf,photo_gt,dem_gt]]


        progress = 1. * epoch / max_epoch

        valid_idx = conf > .5
        conf[conf > .5] = .5 + progress * .4
        conf[conf < .5] = .5 - progress * .4
        conf = torch.clip(conf - conf.mean() + 1.,min=0.)

        obj_proj_pred,dem_pred = pred[:,:2],pred[:,2]

        photo_gt[:,[0,1]] = photo_gt[:,[1,0]]

        obj_gt = torch.stack(rpc.RPC_PHOTO2OBJ(photo_gt[:,0],photo_gt[:,1],dem_gt),dim=-1)
        obj_proj_gt = project_mercator(obj_gt)

        loss_obj = torch.norm(obj_proj_pred - obj_proj_gt,dim=1) * conf
        
        height_dis = torch.abs(dem_pred - dem_gt)
        loss_height = tanh_clamp(height_dis,progress,self.clamp_max)  * conf

        height_valid_mask = height_dis < self.height_tolerance
        dem_pred[~height_valid_mask] = dem_gt[~height_valid_mask]
        
        obj_pred = mercator2lonlat(obj_proj_pred)
        photo_pred = torch.stack(rpc.RPC_OBJ2PHOTO(obj_pred[:,0],obj_pred[:,1],dem_pred),dim=1)
        photo_dis = torch.norm(photo_pred - photo_gt,dim=1)
        loss_photo = tanh_clamp(photo_dis,progress,self.clamp_max) * conf
        real_photo_loss = tanh_clamp(photo_dis,progress,self.clamp_max)[valid_idx].mean()

        bias = tanh_clamp(photo_pred - photo_gt,progress,self.clamp_max) 
        loss_bias =((bias[:,0] * conf).mean() ** 2 + (bias[:,1] * conf).mean() ** 2) ** 0.5

        loss_photo_regularized = affine_loss(photo_gt,photo_pred,conf)
        
        loss = loss_obj.mean() + loss_height.mean() * self.loss_height_weight + loss_photo.mean() + loss_bias + loss_photo_regularized

        return loss, loss_obj.mean() ,loss_height.mean() ,loss_photo.mean(), real_photo_loss,loss_bias.item(),loss_photo_regularized.item()
   
class CriterionTrainElement(nn.Module):

    def __init__(self):
        super().__init__()
        self.height_tolerance = 5
        self.clamp_max = 1000
        self.loss_height_weight = 10
        self.conf_clamp_rate = 2      

    def forward(self,epoch,max_epoch,xyh_pred,conf,linesamp_gt,xyh_gt,rpc:RPCModelParameterTorch):
        
        xyh_pred,conf,linesamp_gt,xyh_gt = [i.to(torch.float64) for i in [xyh_pred,conf,linesamp_gt,xyh_gt]]


        progress = 1. * epoch / max_epoch

        valid_idx = conf > .5
        conf[conf > .5] = .5 + progress * .4
        conf[conf < .5] = .5 - progress * .4
        conf = torch.clip(conf - conf.mean() + 1.,min=0.)

        xy_pred,h_pred = xyh_pred[:,:2],xyh_pred[:,2]
        xy_gt,h_gt = xyh_gt[:,:2],xyh_gt[:,2]

        latlon_pred = mercator2lonlat(xy_pred[:,[1,0]])
        linesamp_pred = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred[:,0],latlon_pred[:,1],h_pred),dim=1)[:,[1,0]]
        
        bias = tanh_clamp(linesamp_pred - linesamp_gt,progress,self.clamp_max)

        loss_obj = torch.norm(xy_pred - xy_gt,dim=1) * conf
        loss_height = torch.abs(h_pred - h_gt) * conf
        loss_photo = tanh_clamp(torch.norm(linesamp_pred - linesamp_gt,dim=1),progress,self.clamp_max) * conf
        loss_bias = ((bias[:,0] * conf).mean() ** 2 + (bias[:,1] * conf).mean() ** 2) ** .5
        loss_reg = affine_loss(linesamp_gt,linesamp_pred,conf)
        
        loss = loss_obj.mean() + loss_height.mean() * self.loss_height_weight + loss_photo.mean() + loss_bias + loss_reg

        return loss, loss_obj.mean() ,loss_height.mean() ,loss_photo.mean(),loss_bias.item(),loss_reg.item()
    
class CriterionTrainGrid(nn.Module):

    def __init__(self):
        super().__init__()
        self.height_tolerance = 5
        self.clamp_max = 1000
        self.loss_height_weight = 10
        self.conf_clamp_rate = 2      

    def forward(self,epoch,max_epoch,feature_dis,mu_xyh,log_sigma_xyh,conf,linesamp_gt,xyh_gt,rpc:RPCModelParameterTorch):
        
        mu_xyh_sample,mu_xyh_anchor = mu_xyh
        log_sigma_xyh_sample,log_sigma_xyh_anchor = log_sigma_xyh
        conf_sample,conf_anchor = conf
        linesamp_gt_sample,linesamp_gt_anchor = linesamp_gt
        xyh_gt_sample,xyh_gt_anchor = xyh_gt

        mu_xyh = torch.cat([mu_xyh_sample,mu_xyh_anchor],dim=0)
        log_sigma_xyh = torch.cat([log_sigma_xyh_sample,log_sigma_xyh_anchor],dim=0)
        conf = torch.cat([conf_sample,conf_anchor],dim=0)
        linesamp_gt = torch.cat([linesamp_gt_sample,linesamp_gt_anchor],dim=0)
        xyh_gt = torch.cat([xyh_gt_sample,xyh_gt_anchor],dim=0)

        progress = 1. * epoch / max_epoch

        conf[conf > .5] = .5 + progress * .4
        conf[conf < .5] = .5 - progress * .4
        conf = torch.clip(conf - conf.mean() + 1.,min=0.)

        xy_pred,h_pred = mu_xyh[:,:2],mu_xyh[:,2]
        xy_gt,h_gt = xyh_gt[:,:2],xyh_gt[:,2]

        sigma_xyh = torch.exp(log_sigma_xyh)
        sigma_xy = sigma_xyh[:,:2]
        sigma_h = sigma_xyh[:,2:]

        loss_distribution_xy = (torch.sum(((xy_gt - xy_pred) ** 2) / (2 * sigma_xy**2 + 1e-8),dim=-1) + torch.sum(log_sigma_xyh[:,:2],dim=-1)) * conf
        loss_distribution_h = (torch.sum(((h_gt - h_pred) ** 2) / (2 * sigma_h**2 + 1e-8),dim=-1) + log_sigma_xyh[:,2:]) * conf
        loss_distribution = loss_distribution_xy + loss_distribution_h

        sigma_avg = torch.norm(sigma_xy,dim=-1).mean()
        

        latlon_pred = mercator2lonlat(xy_pred[:,[1,0]])
        linesamp_pred = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred[:,0],latlon_pred[:,1],h_pred),dim=1)[:,[1,0]]
        

        loss_obj = torch.norm(xy_pred - xy_gt,dim=1) * conf
        loss_height = torch.abs(h_pred - h_gt) * conf

        loss_photo = torch.norm(linesamp_pred - linesamp_gt,dim=1) * conf

        dis_pred = torch.norm(mu_xyh_sample - mu_xyh_anchor,dim=1)
        dis_gt = torch.norm(xyh_gt_sample - xyh_gt_anchor,dim=1)

        loss_dis = torch.clip(dis_pred - dis_gt,min=0.) * .1

        pairs = np.random.randint(0,len(xy_pred)-1,size=(1000,2))
        invalid_pair_mask = pairs[:, 0] == pairs[:, 1]
        pairs[invalid_pair_mask,1] = (pairs[invalid_pair_mask, 1] + 1) % len(xy_pred)
        offset_gt_pair = xy_gt[pairs[:,0]] - xy_gt[pairs[:,1]]
        offset_pred_pair = xy_pred[pairs[:,0]] - xy_pred[pairs[:,1]]
        loss_relative = torch.norm(torch.abs(offset_pred_pair - offset_gt_pair),dim=-1)

        loss = loss_distribution.mean() + loss_obj.mean() + loss_height.mean() * self.loss_height_weight + loss_photo.mean() + progress * loss_dis.mean() + loss_relative.mean()

        return loss, loss_distribution.mean(),loss_obj.mean() ,loss_height.mean() ,loss_photo.mean(),loss_dis.mean(),loss_relative.mean(),sigma_avg.item()