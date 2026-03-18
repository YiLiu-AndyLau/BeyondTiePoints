import warnings

warnings.filterwarnings('ignore')
import argparse
import torch
import numpy as np
import os
import cv2
from utils import project_mercator,mercator2lonlat,bilinear_interpolate,resample_from_quad

from rpc import RPCModelParameterTorch
from tqdm import tqdm,trange
import rasterio
from typing import Tuple

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class RSImage():
    def __init__(self,options,root:str,id:int,size_limit = 0):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        """
        self.options = options
        self.root = root
        self.id = id
        
        self.image = cv2.imread(os.path.join(root,'image.png'),cv2.IMREAD_GRAYSCALE)
        self.image = np.stack([self.image] * 3,axis=-1)
        self.dem = np.load(os.path.join(root,'dem.npy'),mmap_mode='r')
        if os.path.exists(os.path.join(root,'tie_points.txt')):
            self.tie_points = self.__load_tie_points__(os.path.join(root,'tie_points.txt'))
            self.tie_points_height = self.dem[self.tie_points[:,0],self.tie_points[:,1]]
        else:
            self.tie_points = None
            self.tie_points_height = None

        if size_limit > 0:
            self.image = self.image[:size_limit,:size_limit]
            self.dem = self.dem[:size_limit,:size_limit]

        self.H,self.W = self.image.shape[:2]
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(root,'rpc.txt'))
        self.rpc.to_gpu()
        
        self.corner_xys = self.__get_corner_xys__() #[tl,tr,bl,br] [x,y]
        self.overlap_grids = []
        self.R = torch.tensor([[1.0,0.0],
                                [0.0,1.0]])
        self.T = torch.tensor([0.,0.])

    def __load_image__(self,path) -> np.ndarray:
        print("Loading Image")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            for band in range(data.shape[0]):
                data[band] = (255. * data[band] / data[band].max())
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0)).squeeze()
        return data[:10000,:10000]

    def __load_tie_points__(self,path) -> np.ndarray:
        tie_points = np.loadtxt(path,dtype=int)
        if tie_points.ndim == 1:
            tie_points = tie_points.reshape(1,-1)
        elif tie_points.shape[1] != 2:
            print("tie points format error")
            return None
        return tie_points
    
    @torch.no_grad()
    def __get_corner_xys__(self):
        """
        return: [tl,tr,bl,br] [x,y] np.ndarray
        """
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ([0.,self.W-1.,0.,self.W-1.],[0.,0.,self.H - 1.,self.H - 1.],[self.dem[0,0],self.dem[0,-1],self.dem[-1,0],self.dem[-1,-1]]),dim=-1)
        xys = project_mercator(latlons)
        return xys.cpu().numpy()[:,[1,0]] # y,x -> x,y

    
    @torch.no_grad()
    def dem_interp(self,sampline:np.ndarray):
        if sampline.ndim == 1:
            sampline = sampline[None]
        return bilinear_interpolate(self.dem,sampline)
    
    @torch.no_grad()
    def xy_to_sampline(self,xy:np.ndarray,max_iter = 100):
        if xy.ndim == 1:
            xy = xy[None]
        latlon = mercator2lonlat(xy[:,[1,0]])
        sampline = np.array([self.W,self.H],dtype=np.float32) * (xy - self.corner_xys[0]) / (self.corner_xys[3] - self.corner_xys[0])
        dem = self.dem_interp(sampline)
        invalid_mask = np.full(dem.shape,True,dtype=bool)
        for iter in range(max_iter):
            sampline_new = np.stack(self.rpc.RPC_OBJ2PHOTO(latlon[invalid_mask,0],latlon[invalid_mask,1],dem[invalid_mask],'numpy'),axis=-1)
            dis = np.linalg.norm(sampline_new - sampline[invalid_mask],axis=-1)
            sampline[invalid_mask] = sampline_new
            invalid_mask[invalid_mask] = dis > 1.
            if invalid_mask.sum() == 0:
                break
        return sampline.squeeze()

    @torch.no_grad()
    def get_image_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
        tl_sampline = np.array(tl_sampline)
        br_sampline = np.array(br_sampline)
        H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
        samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
        tl_sampline = np.array([samp_start,line_start],dtype=int)
        br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        return self.image[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]
    
    @torch.no_grad()
    def get_dem_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
        tl_sampline = np.array(tl_sampline)
        br_sampline = np.array(br_sampline)
        H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
        samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
        tl_sampline = np.array([samp_start,line_start],dtype=int)
        br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        return self.dem[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]

    @torch.no_grad()
    def get_image_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
        """
        return: crop_img,tl_sampline,br_sampline
        """
        tlxy = np.array(tlxy)
        brxy = np.array(brxy)
        tl_sampline = self.xy_to_sampline(tlxy)
        br_sampline = self.xy_to_sampline(brxy)
        return self.get_image_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    @torch.no_grad()
    def get_dem_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
        """
        return: crop_dem,tl_sampline,br_sampline
        """
        tlxy = np.array(tlxy)
        brxy = np.array(brxy)
        tl_sampline = self.xy_to_sampline(tlxy)
        br_sampline = self.xy_to_sampline(brxy)
        return self.get_dem_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    @torch.no_grad()
    def resample_image_by_sampline(self,corner_samplines:np.ndarray,target_shape:Tuple[int,int],need_local:bool = False):
        img_resampled,local_hw2 = resample_from_quad(self.image,corner_samplines[:,[1,0]],target_shape)
        if need_local:
            return img_resampled,local_hw2
        else:
            return img_resampled
    
    @torch.no_grad()
    def resample_dem_by_sampline(self,corner_samplines:np.ndarray,target_shape:Tuple[int,int],need_local:bool = False):
        dem_resampled,local_hw2 = resample_from_quad(self.dem,corner_samplines[:,[1,0]],target_shape)
        if need_local:
            return dem_resampled,local_hw2
        else:
            return dem_resampled
        
    def vis_grid(self,diags:list[np.ndarray],output_path:str = None):
        vis_img = self.image.copy()
        for diag in diags:
            min_x,min_y,max_x,max_y = diag[:,0].min(),diag[:,1].min(),diag[:,0].max(),diag[:,1].max()
            corners = [self.xy_to_sampline(np.array([min_x,min_y])),
                       self.xy_to_sampline(np.array([max_x,min_y])),
                       self.xy_to_sampline(np.array([max_x,max_y])),
                       self.xy_to_sampline(np.array([min_x,max_y])),
                       self.xy_to_sampline(np.array([min_x,min_y]))]
            for i in range(len(corners) - 1):
                cv2.line(vis_img,(int(corners[i][0]),int(corners[i][1])),(int(corners[i+1][0]),int(corners[i+1][1])),(0,0,255),5)
        
        if not output_path is None:
            cv2.imwrite(output_path,vis_img)
        
        return vis_img