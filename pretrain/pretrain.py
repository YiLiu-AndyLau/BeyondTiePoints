import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from criterion import CriterionFinetune
from model.encoder import EncoderDino
from model.decoder import DecoderFinetune
import h5py
import datetime
import time
from tqdm import trange
from dataloader import PretrainDataset,ImageSampler
from utils import str2bool,vis_conf,vis_feat_pca,visualize_feature_correspondences,visualize_obj_error
import warnings
warnings.filterwarnings("ignore")
from scheduler import MultiStageOneCycleLR

from functools import partial
import cv2

def print_on_main(msg, rank):
    if rank == 0:
        print(msg)

def apply_polynomial(x, coefs):
    y = torch.zeros_like(x)
    for i, c in enumerate(coefs):
        y = y + c * (x ** (len(coefs) - 1 - i))
    return y

def sample_features(features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    H, W = features.shape[-2:]
    
    y_coords = coords[:, :, 0]
    x_coords = coords[:, :, 1]

    x_normalized = 2.0 * x_coords / W - 1.0
    y_normalized = 2.0 * y_coords / H - 1.0
    
    normalized_grid = torch.stack([x_normalized, y_normalized], dim=2)

    grid_for_sampling = normalized_grid.unsqueeze(1)

    sampled_features = F.grid_sample(
        features,
        grid_for_sampling,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    final_output = sampled_features.squeeze(2)
    return final_output

def warp_by_poly(raw,coefs):
    x = (raw[:,0] + 1.) * .5 * (coefs['x'][1] - coefs['x'][0]) + coefs['x'][0]
    y = (raw[:,1] + 1.) * .5 * (coefs['y'][1] - coefs['y'][0]) + coefs['y'][0]
    h = apply_polynomial(raw[:,2],coefs['h'])
    warped = torch.stack([x,y,h],dim=-1)
    return warped

def distibute_model(model:nn.Module,local_rank):
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)
    return model

def output_img(imgs_raw:torch.Tensor,output_path:str,name:str):
    os.makedirs(output_path,exist_ok=True)
    for idx,img in enumerate(imgs_raw):
        img = img.permute(1,2,0).cpu().numpy()[:,:,0]
        img = 255 * (img - img.min()) / (img.max() - img.min())
        cv2.imwrite(f'{output_path}/{name}_{idx}.png',img.astype(np.uint8))

@torch.no_grad()
def vis(encoder:EncoderDino,vis_img:np.ndarray):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))
                ])
    input = transform(vis_img).unsqueeze(0).to(encoder.device)
    feat,conf = encoder(input)
    h,w,c = feat.shape[-2],feat.shape[-1],feat.shape[1]
    feat = feat.permute(0,2,3,1).reshape(h,w,c).cpu().numpy()
    conf = conf.reshape(h,w).cpu().numpy()
    feat = vis_feat_pca(feat)
    conf_cont,conf_div = vis_conf(conf,vis_img,encoder.module.SAMPLE_FACTOR)
    return feat,conf_cont,conf_div


def compute_loss(args,epoch,data,encoder:EncoderDino,decoder:DecoderFinetune,criterion:nn.Module,only_decoder:bool = False):
    img1 = data['img1'].squeeze(0).to(args.device)
    img2 = data['img2'].squeeze(0).to(args.device)
    obj1 = data['obj1'].squeeze(0).to(args.device)
    obj2 = data['obj2'].squeeze(0).to(args.device)
    residual1 = data['residual1'].squeeze(0).to(args.device)
    residual2 = data['residual2'].squeeze(0).to(args.device)
    overlap1 = data['overlap1'].squeeze(0).to(args.device)
    overlap2 = data['overlap2'].squeeze(0).to(args.device)
    obj_map_coef = data['obj_map_coef']
    res_mid = data['res_mid']
    B,H,W = obj1.shape[:3]

    feat1,conf1 = encoder(img1)
    feat2,conf2 = encoder(img2)

    size = feat1.shape[2]


    feat1_sample = sample_features(feat1,overlap1).unsqueeze(-1)
    feat2_sample = sample_features(feat2,overlap2).unsqueeze(-1)
    obj1_sample_P3 = sample_features(obj1.permute(0,3,1,2),overlap1).permute(0,2,1).flatten(0,1)
    obj2_sample_P3 = sample_features(obj2.permute(0,3,1,2),overlap2).permute(0,2,1).flatten(0,1)


    feat_input1 = feat1
    feat_input2 = feat2


    output1_B3hw = decoder(feat_input1)
    output2_B3hw = decoder(feat_input2)
    output1_P3 = output1_B3hw.permute(0,2,3,1).flatten(0,2)
    output2_P3 = output2_B3hw.permute(0,2,3,1).flatten(0,2)
    pred1_P3 = warp_by_poly(output1_P3,obj_map_coef)
    pred2_P3 = warp_by_poly(output2_P3,obj_map_coef)

    output1_sample_B3hw = decoder(feat1_sample)
    output2_sample_B3hw = decoder(feat2_sample)
    output1_sample_P3 = output1_sample_B3hw.permute(0,2,3,1).flatten(0,2)
    output2_sample_P3 = output2_sample_B3hw.permute(0,2,3,1).flatten(0,2)
    pred1_sample_P3 = warp_by_poly(output1_sample_P3,obj_map_coef)
    pred2_sample_P3 = warp_by_poly(output2_sample_P3,obj_map_coef)

    feat_vis = [feat1[0].permute(1,2,0).detach().cpu().numpy(),feat2[0].permute(1,2,0).detach().cpu().numpy()]

    feat1_PD = feat1_sample.permute(0,2,3,1).flatten(0,2)
    feat2_PD = feat2_sample.permute(0,2,3,1).flatten(0,2)
    conf1_P = conf1.permute(0,2,3,1).reshape(-1)
    conf2_P = conf2.permute(0,2,3,1).reshape(-1)
    obj1_P3 = obj1.flatten(0,2)
    obj2_P3 = obj2.flatten(0,2)
    residual1_P = residual1.reshape(-1).detach()
    residual2_P = residual2.reshape(-1).detach()
    conf_mean = .5 * conf1_P.clone().detach().mean() + .5 * conf2_P.clone().detach().mean()

    loss,loss_obj,loss_height,loss_relative,loss_conf,loss_feat,k,sp,sn = criterion(epoch,args.max_epoch,
                                                                feat1_PD,feat2_PD,
                                                                pred1_P3,pred2_P3,
                                                                conf1_P,conf2_P,
                                                                obj1_P3,obj2_P3,
                                                                residual1_P,residual2_P,
                                                                res_mid,
                                                                only_decoder,
                                                                H,W)
    feat_dis = torch.norm(feat1_sample.permute(0,2,3,1).flatten(0,2) - feat2_sample.permute(0,2,3,1).flatten(0,2),dim=1).mean().detach() * 100.

    loss_dis = torch.norm(pred1_sample_P3 - pred2_sample_P3,dim=-1).mean()
    loss_obj_sample = .5 * torch.norm(pred1_sample_P3 - obj1_sample_P3,dim=-1).mean() + .5 * torch.norm(pred2_sample_P3 - obj2_sample_P3,dim=-1).mean()
    if not only_decoder:
        loss = loss + loss_dis + loss_obj_sample            
    else:
        loss = loss + loss_dis * 0.  + loss_obj_sample * 0.
    
    if epoch % 5 == 0:
        obj_vis = visualize_obj_error(obj1_P3[:size * size,:2].detach().cpu().numpy(),pred1_P3[:size * size,:2].detach().cpu().numpy(),sample_k=1e9)
        obj1_sample_vis = visualize_obj_error(obj1_sample_P3[:1000,:2].detach().cpu().numpy(),pred1_sample_P3[:1000,:2].detach().cpu().numpy(),sample_k=1e9)
        obj2_sample_vis = visualize_obj_error(obj2_sample_P3[:1000,:2].detach().cpu().numpy(),pred2_sample_P3[:1000,:2].detach().cpu().numpy(),sample_k=1e9)
        dis_vis = visualize_obj_error(pred1_sample_P3[:1000,:2].detach().cpu().numpy(),pred2_sample_P3[:1000,:2].detach().cpu().numpy(),sample_k=1e9,ranges=[[-1000,1000],[-1000,1000]])

        vis_data = {
            'feat_vis':feat_vis,
            'obj_vis':obj_vis,
            'obj1_sample_vis':obj1_sample_vis,
            'obj2_sample_vis':obj2_sample_vis,
            'dis_vis':dis_vis
        }
    else:
        vis_data = {}

    return loss,loss_obj,loss_height,loss_relative,loss_conf,loss_feat,loss_dis,loss_obj_sample,k,conf_mean,feat_dis,sp,sn,vis_data

def pretrain(args):
    os.makedirs('./log',exist_ok=True)
    os.makedirs(args.encoder_output_path,exist_ok=True)
    os.makedirs(args.checkpoints_path,exist_ok=True)
    pprint = partial(print_on_main, rank=dist.get_rank())
    num_gpus = dist.get_world_size()
    pprint(f"Using {num_gpus} GPUS")
    pprint("Loading Dataset")
    rank = dist.get_rank()
    
    if args.resume_training:
        training_configs = torch.load(os.path.join(args.checkpoints_path,'training_configs.pth'))
        min_loss = training_configs['min_loss']
        last_loss = training_configs['last_loss']
        epoch = training_configs['epoch']
        dataset_indices = training_configs['dataset_indices'].to(args.device)
        obj_map_coefs = training_configs['obj_map_coefs']
        obj_map_coefs = [{k:v.numpy() for k,v in i.items()} for i in obj_map_coefs]
        log_name = training_configs['log_name']
        if rank == 0:
            logger = SummaryWriter(log_dir=os.path.join('./log',f'{log_name}_tensorboard'))
        else:
            logger = None
    else:
        training_configs = None
        min_loss = args.min_loss
        last_loss = None
        epoch = 0
        obj_map_coefs = None
        log_name = args.log_prefix
        if rank == 0:
            logger = SummaryWriter(log_dir=os.path.join('./log',f'{args.log_prefix}_tensorboard'))
        else:
            logger = None

    if not args.resume_training:
        if not args.decoder_path is None:
            dataset_indices = torch.from_numpy(np.load(os.path.join(args.decoder_path,'dataset_indices.npy'))).to(dtype=torch.long,device=args.device)
        else:

            if not args.dataset_select is None:
                args.dataset_num = len(args.dataset_select.split(','))
            dataset_indices = torch.empty(args.dataset_num,dtype=torch.long,device=args.device)
            if rank == 0:
                with h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r') as f:
                    total_num = len(f.keys())
                if args.dataset_select is None:
                    dataset_indices = torch.randperm(total_num)[:args.dataset_num].to(args.device)
                else:
                    dataset_indices = torch.tensor([int(i) for i in args.dataset_select.split(',')],dtype=int,device=args.device)
                indices_str = [str(idx) for idx in dataset_indices.cpu().numpy()]
                indices_str = " ".join(indices_str)
                with open(os.path.join('./log',f'{args.log_prefix}_dataset_idxs_log.txt'),'a') as f:
                    f.write(f"{indices_str}\n")

    dist.barrier()
    dist.broadcast(dataset_indices,src=0)

    

    dataset_indices = dataset_indices.cpu().numpy()

    dataset = PretrainDataset(root = args.dataset_path,
                              dataset_idxs=dataset_indices,
                              batch_size = args.batch_size,
                              downsample = 4,
                              input_size = 1024,
                              obj_map_coefs = obj_map_coefs,
                              norm_coefs={
                                    'mean':(0.485, 0.456, 0.406),
                                    'std':(0.229, 0.224, 0.225)
                                },
                              use_clahe = False,
                              mode='train')
    sampler = ImageSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset,sampler=sampler,batch_size=1,num_workers=4,drop_last=False,pin_memory=False,shuffle=False)
    dataset_num = dataset.dataset_num
    train_images = dataset.get_train_images()
    if dist.get_rank() == 0:
        for i,img in enumerate(train_images):
            tag = f'train_imgs/{i}'
            logger.add_image(tag,img,0,dataformats='HWC')


    pprint("Building Encoder")

    
    only_decoder_epoch = int(args.max_epoch * args.only_decoder_ratio)
    
    encoder = EncoderDino(dino_weight_path=args.dino_weight_path,adapter_pos_embed = args.pos_embed,unitize=True)
    adapter_optimizer = optim.AdamW(params=encoder.adapter.parameters(),lr = args.lr_encoder_max)

    adapter_scheduler = MultiStageOneCycleLR(optimizer=adapter_optimizer,
                                             total_steps=(args.max_epoch - only_decoder_epoch) * dataset_num,
                                             warmup_ratio=min(5. / args.max_epoch,.1),
                                             cooldown_ratio=.9)


    args.output_channels = encoder.output_channels
    
    
    if args.resume_training:
        encoder.load_adapter(os.path.join(args.checkpoints_path,'adapter.pth'))
        adapter_optimizer.load_state_dict(torch.load(os.path.join(args.checkpoints_path,'adapter_optimizer.pth'),map_location='cpu'))
        adapter_scheduler.load_state_dict(torch.load(os.path.join(args.checkpoints_path,'adapter_scheduler.pth'),map_location='cpu'))
        
    elif not args.encoder_path is None:
        encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
        pprint('Encoder Loaded')

    encoder = encoder.to(args.device)
    for state in adapter_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)


    encoder_op = encoder
    if num_gpus > 1:
        encoder = distibute_model(encoder,args.local_rank)
        encoder_op = encoder.module
    

    pprint("Building Decoders")     

    decoders = []
    optimizers = []
    schedulers = []
    for dataset_idx in trange(dataset_num):
        decoder = DecoderFinetune(in_channels=args.output_channels,block_num=args.decoder_block_num,use_bn=False)
        optimizer = optim.AdamW(params=decoder.parameters(),lr = args.lr_decoder_max)
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                        total_steps=args.max_epoch,
                                        warmup_ratio=min(5. / args.max_epoch,.1),
                                        cooldown_ratio=.9)
        
        if args.resume_training:
            decoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.checkpoints_path,f'decoder_{dataset_idx}.pth'),map_location='cpu').items()})
            optimizer.load_state_dict(torch.load(os.path.join(args.checkpoints_path,f'decoder_optimizer_{dataset_idx}.pth'),map_location='cpu'))
            scheduler.load_state_dict(torch.load(os.path.join(args.checkpoints_path,f'decoder_scheduler_{dataset_idx}.pth'),map_location='cpu'))
        elif not args.decoder_path is None:
            decoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.decoder_path,f'decoder_{dataset_idx}.pth'),map_location='cpu').items()})
                

        decoder = decoder.to(args.device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        if num_gpus > 1:
            decoder = distibute_model(decoder,args.local_rank)

        decoder.train()
        
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        decoders.append(decoder)

    
    start_time = time.perf_counter()
    criterion = CriterionFinetune()
    step_count = 0
    for epoch in range(epoch,args.max_epoch):
        pprint(f'\nEpoch:{epoch}')
        sampler.set_epoch(epoch)
        total_loss = 0
        total_loss_obj = 0
        total_loss_dis = 0
        total_loss_obj_sample = 0
        total_loss_relative = 0
        total_loss_height = 0
        total_loss_conf = 0
        total_loss_feat = 0
        total_sp = 0
        total_sn = 0
        count = 0
        encoder.train()

        for iter_idx,data in enumerate(dataloader):
            img1,img2,obj1,obj2,residual1,residual2,overlap1,overlap2,dataset_idx = data
            dataset_idx = dataset_idx.item()

            decoder = decoders[dataset_idx]
            decoder_optimizer = optimizers[dataset_idx]
            decoder.train()
            adapter_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            compose_data = {
                "img1":img1,
                'img2':img2,
                "obj1":obj1,
                "obj2":obj2,
                "residual1":residual1,
                "residual2":residual2,
                "overlap1":overlap1,
                "overlap2":overlap2,
                "obj_map_coef":dataset.obj_map_coefs[dataset_idx],
                "res_mid":dataset.red_mids[dataset_idx]
            }

            loss,loss_obj,loss_height,loss_relative,loss_conf,loss_feat,loss_dis,loss_obj_sample,k,conf_mean,feat_dis,sp,sn,vis_data = compute_loss(args,epoch,compose_data,encoder,decoder,criterion,epoch < only_decoder_epoch)

            

            loss_is_nan = not torch.isfinite(loss).all()

            loss_status_tensor = torch.tensor([loss_is_nan], dtype=torch.float32, device=rank)

            dist.all_reduce(loss_status_tensor, op=dist.ReduceOp.SUM)

            if loss_status_tensor.item() > 0:
                pprint(f"---  NaN！Epoch {epoch}, iter {iter_idx+1}, dataset {dataset_idx}. 。---")
                del loss,loss_obj,loss_height,loss_conf,loss_feat,loss_dis,conf_mean
                adapter_scheduler.step()
                continue 
            
            loss.backward()

            decoder_optimizer.step()
            adapter_optimizer.step()
            
            loss_rec = loss.clone().detach()
            loss_obj_rec = loss_obj.clone().detach()
            loss_dis_rec = loss_dis.clone().detach()
            loss_obj_sample_rec = loss_obj_sample.clone().detach()
            loss_height_rec = loss_height.clone().detach()
            loss_conf_rec = loss_conf.clone().detach()
            loss_feat_rec = loss_feat.clone().detach()
            loss_relative_rec = loss_relative.clone().detach()
            sp_rec = sp.clone().detach()
            sn_rec = sn.clone().detach()

            total_loss += loss_rec
            total_loss_obj += loss_obj_rec
            total_loss_dis += loss_dis_rec
            total_loss_obj_sample += loss_obj_sample
            total_loss_relative += loss_relative_rec
            total_loss_height += loss_height_rec
            total_loss_conf += loss_conf_rec
            total_loss_feat += loss_feat_rec
            total_sp += sp_rec
            total_sn += sn_rec
            count += 1
            step_count += 1


            dist.all_reduce(loss_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_obj_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_dis_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_obj_sample_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_height_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_relative_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_conf_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_feat_rec,dist.ReduceOp.AVG)
            dist.all_reduce(conf_mean,dist.ReduceOp.AVG)
            dist.all_reduce(feat_dis,dist.ReduceOp.AVG)
            dist.all_reduce(sp_rec,dist.ReduceOp.AVG)
            dist.all_reduce(sn_rec,dist.ReduceOp.AVG)
            dist.barrier()

            if dist.get_rank() == 0:
                curtime = time.perf_counter()
                curstep = step_count
                remain_step = (args.max_epoch - epoch)  * dataset_num - count
                cost_time = curtime - start_time
                remain_time = remain_step * cost_time / curstep

                print(f"epoch:{epoch} iter:{iter_idx+1}/{dataset_num}\t l_obj:{loss_obj_rec.item():.2f} \t l_obj_s:{loss_obj_sample_rec.item():.2f} \t l_dis:{loss_dis_rec.item():.2f} \t l_h:{loss_height_rec.item():.2f} \t l_r:{loss_relative_rec.item():.2f} \t l_conf:{loss_conf_rec.item():.2f} \t cm:{conf_mean.item():.2f} \t fd:{feat_dis.item():.2f} \t k:{k:.2f} \t l_f:{loss_feat_rec.item():.2f} \t sp:{sp_rec.item():.2f} \t sn:{sn_rec.item():.2f} \t en_lr:{adapter_optimizer.param_groups[0]['lr']:.2e}  de_lr:{optimizers[0].param_groups[0]['lr']:.2e} \t time:{str(datetime.timedelta(seconds=round(cost_time)))}  ETA:{str(datetime.timedelta(seconds=round(remain_time)))}")


            if epoch >= only_decoder_epoch:
                adapter_scheduler.step()

        for scheduler in schedulers:
            scheduler.step()            

        total_loss /= count
        total_loss_obj /= count
        total_loss_dis /= count
        total_loss_obj_sample /= count
        total_loss_relative /= count
        total_loss_height /= count
        total_loss_conf /= count
        total_loss_feat /= count
        total_sp /= count
        total_sn /= count

        dist.all_reduce(total_loss,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_obj,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_dis,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_obj_sample,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_relative,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_height,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_conf,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_feat,dist.ReduceOp.AVG)
        dist.all_reduce(total_sp,dist.ReduceOp.AVG)
        dist.all_reduce(total_sn,dist.ReduceOp.AVG)

        total_loss = total_loss.item()
        total_loss_obj = total_loss_obj.item()
        total_loss_dis = total_loss_dis.item()
        total_loss_obj_sample = total_loss_obj_sample.item()
        total_loss_relative = total_loss_relative.item()
        total_loss_height = total_loss_height.item()
        total_loss_conf = total_loss_conf.item()
        total_loss_feat = total_loss_feat.item()
        total_sp = total_sp.item()
        total_sn = total_sn.item()


        if dist.get_rank() == 0:
            if last_loss is None:
                print(f'total_loss:{total_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t obj_s:{total_loss_obj_sample:.2f}  \t dis:{total_loss_dis:.2f} \t rela:{total_loss_relative:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f} \t sp:{total_sp:.2f} \t sn:{total_sn:.2f}')
            else:
                print(f"total_loss:{total_loss} \t diff:{'+' if total_loss - last_loss > 0 else ''}{total_loss - last_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t obj_s:{total_loss_obj_sample:.2f}  \t dis:{total_loss_dis:.2f} \t rela:{total_loss_relative:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f} \t sp:{total_sp:.2f} \t sn:{total_sn:.2f}")
            last_loss = total_loss

            
            if total_loss_obj < min_loss and epoch >= only_decoder_epoch + 50:
                min_loss = total_loss_obj
                encoder_op.save_adapter(os.path.join(args.encoder_output_path,'adapter.pth'))
                print('best updated')
            
            if epoch % 5 == 0:
                path = args.checkpoints_path
                adapter_optimizer_state_dict = adapter_optimizer.state_dict()
                adapter_scheduler_state_dict = adapter_scheduler.state_dict()
                encoder_op.save_adapter(os.path.join(path,'adapter.pth'))
                torch.save(adapter_optimizer_state_dict,os.path.join(path,'adapter_optimizer.pth'))
                torch.save(adapter_scheduler_state_dict,os.path.join(path,'adapter_scheduler.pth'))
                for i in range(dataset_num):
                    decoder_state_dict = {k:v.detach().cpu() for k,v in decoders[i].state_dict().items()}
                    decoder_optimizer_state_dict = optimizers[i].state_dict()
                    decoder_scheduler_state_dict = schedulers[i].state_dict()
                    torch.save(decoder_state_dict,os.path.join(path,f'decoder_{i}.pth'))
                    torch.save(decoder_optimizer_state_dict,os.path.join(path,f'decoder_optimizer_{i}.pth'))
                    torch.save(decoder_scheduler_state_dict,os.path.join(path,f'decoder_scheduler_{i}.pth'))
                obj_map_coefs_save = [{k:torch.from_numpy(v) for k,v in i.items()} for i in dataset.obj_map_coefs]
                training_configs = {
                    'dataset_indices':torch.from_numpy(dataset_indices),
                    'obj_map_coefs':obj_map_coefs_save,
                    'epoch':torch.tensor(epoch),
                    'min_loss':torch.tensor(min_loss),
                    'last_loss':torch.tensor(last_loss),
                    'log_name':log_name
                }
                torch.save(training_configs,os.path.join(path,'training_configs.pth'))

                vis_img_raw = cv2.imread(args.vis_img_path)
                vis_img = np.zeros(vis_img_raw.shape,dtype=np.uint8)
                cv2.normalize(vis_img_raw,vis_img,0,255,cv2.NORM_MINMAX)
                feat,conf_cont,conf_div = vis(encoder,vis_img)

                vis_cor_idx = torch.randperm(len(overlap1[0][0]))[:10]
                cor_idx1, cor_idx2 = overlap1[0][0][vis_cor_idx].detach().cpu().numpy()[:,[1,0]],overlap2[0][0][vis_cor_idx].detach().cpu().numpy()[:,[1,0]]
                feat_cor = visualize_feature_correspondences(vis_data['feat_vis'][0],vis_data['feat_vis'][1],cor_idx1,cor_idx2)

                train_img_1,train_img_2 = img1[0][0].permute(1,2,0).detach().cpu().numpy(),img2[0][0].permute(1,2,0).detach().cpu().numpy()
                train_img_1 = 255. * (train_img_1 - train_img_1.min()) / (train_img_1.max() - train_img_1.min())
                train_img_2 = 255. * (train_img_2 - train_img_2.min()) / (train_img_2.max() - train_img_2.min())


                logger.add_image('vis/feat',feat,epoch,dataformats='HWC')
                logger.add_image('vis/conf_cont',conf_cont,epoch,dataformats='HWC')
                logger.add_image('vis/conf_div',conf_div,epoch,dataformats='HWC')
                logger.add_image('vis/feat_cor',feat_cor,epoch,dataformats='HWC')
                logger.add_image('vis/train_img_1',train_img_1.astype(np.uint8),epoch,dataformats='HWC')
                logger.add_image('vis/train_img_2',train_img_2.astype(np.uint8),epoch,dataformats='HWC')
                logger.add_image('vis/obj_vis_scatter',vis_data['obj_vis']['scatter'],epoch,dataformats='HWC')
                logger.add_image('vis/obj1_sample_vis_scatter',vis_data['obj1_sample_vis']['scatter'],epoch,dataformats='HWC')
                logger.add_image('vis/obj2_sample_vis_scatter',vis_data['obj2_sample_vis']['scatter'],epoch,dataformats='HWC')
                logger.add_image('vis/dis_vis_scatter',vis_data['dis_vis']['scatter'],epoch,dataformats='HWC')
            

            logger.add_scalar('loss/total_loss', total_loss, epoch)
            logger.add_scalar('loss/loss_obj', total_loss_obj, epoch)
            logger.add_scalar('loss/loss_dis', total_loss_dis, epoch)
            logger.add_scalar('loss/loss_relative',total_loss_relative,epoch)
            logger.add_scalar('loss/loss_height', total_loss_height, epoch)
            logger.add_scalar('loss/loss_conf', total_loss_conf, epoch)
            logger.add_scalar('loss/loss_feat', total_loss_feat, epoch)
            logger.add_scalar('train/lr_encoder',adapter_optimizer.param_groups[0]['lr'], epoch)
            logger.add_scalar('train/lr_decoder',optimizers[0].param_groups[0]['lr'], epoch)
            logger.add_scalar('metrics/sp', total_sp, epoch)
            logger.add_scalar('metrics/sn', total_sn, epoch)

        dist.barrier()
    if dist.get_rank() == 0:
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=None)
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--encoder_output_path',type=str,default='./weights/encoder_finetune.pth')
    parser.add_argument('--checkpoints_path',type=str,default=None)
    parser.add_argument('--vis_img_path',type=str,default=None)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--decoder_block_num',type=int,default=1)
    parser.add_argument('--decoder_path',type=str,default=None)
    parser.add_argument('--resume_training',type=str2bool,default=False)
    parser.add_argument('--pos_embed',type=str2bool,default=False)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--lr_encoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_encoder_max',type=float,default=5e-4)
    parser.add_argument('--lr_decoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_decoder_max',type=float,default=1e-3)
    parser.add_argument('--min_loss',type=float,default=1e8)
    parser.add_argument('--log_prefix',type=str,default='')
    parser.add_argument('--dataset_select',type=str,default=None)
    parser.add_argument('--only_decoder_ratio',type=float,default=0.)
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        args.device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    pprint = partial(print_on_main, rank=dist.get_rank())

    torch.manual_seed(42)
    np.random.seed(42)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pprint("==============================configs==============================")
    for k,v in vars(args).items():
        pprint(f"{k}:{v}")
    pprint("===================================================================")
    pretrain(args)