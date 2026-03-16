import numpy as np
import os
import torch
import cv2


class RPCModelParameterTorch:
    def __init__(self, data=torch.zeros(170,dtype=torch.double)):
        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        self.LATNUM = data[90:110]
        self.LATDEM = data[110:130]
        self.LONNUM = data[130:150]
        self.LONDEM = data[150:170]

        self.Clear_Adjust()

        self.device = self.LINE_OFF.device

    """Read orginal RPC File"""
    def load_from_file(self, filepath):
        """
        Here, we define:
            direct rpc: from object (lat, lon, hei) to photo (sample, line)
            inverse rpc: from photo (sample, line, hei) to object (lat, lon)
        Function: Read direct rpc from file and then calculate the inverse rpc
        """
        if os.path.exists(filepath) is False:
            print("Error#001: cann't find " + filepath + " in the file system!")
            return

        with open(filepath, 'r') as f:
            all_the_text = f.read().splitlines()
            rfm_line = -1
            for line,text in enumerate(all_the_text):
                if "RFM_CORRECTION_PARAMETERS" in text:
                    rfm_line = line
                    break

        data = [np.float64(text.split()[1]) for text in (all_the_text[:rfm_line-1] if rfm_line > 0 else all_the_text)]
        data = torch.from_numpy(np.array(data, dtype=np.float64)).to(torch.double)

        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]
        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]
        
        if data.shape[0] >= 170:
            self.LATNUM = data[90:110]
            self.LATDEM = data[110:130]
            self.LONNUM = data[130:150]
            self.LONDEM = data[150:170]
        else:
            self.Calculate_Inverse_RPC()
        
    
        # if data.shape[0] == 96:
        #     self.raw_adjust_params = data[90:96]
        # else:
        #     self.raw_adjust_params = None
        
        if rfm_line > 0:
            self.raw_adjust_params = [np.float64(text.split()[1]) for text in all_the_text[rfm_line + 1:]]
            self.Calculate_Adjust()
        else:
            self.raw_adjust_params = None

    def Create_Virtual_3D_Grid(self, xy_sample=30, z_sample=20):
        """
        Create_Virtual 3D control grid in the object space
        :return: grid (N, 5)
        """
        lat_max = self.LAT_OFF + self.LAT_SCALE
        lat_min = self.LAT_OFF - self.LAT_SCALE
        lon_max = self.LONG_OFF + self.LONG_SCALE
        lon_min = self.LONG_OFF - self.LONG_SCALE
        hei_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hei_min = self.HEIGHT_OFF - self.HEIGHT_SCALE
        samp_max = self.SAMP_OFF + self.SAMP_SCALE
        samp_min = self.SAMP_OFF - self.SAMP_SCALE
        line_max = self.LINE_OFF + self.LINE_SCALE
        line_min = self.LINE_OFF - self.LINE_SCALE

        lat = torch.linspace(lat_min, lat_max, xy_sample).to(self.device,dtype=torch.double) #np.linspace(lat_min, lat_max, xy_sample)
        lon = torch.linspace(lon_min, lon_max, xy_sample).to(self.device,dtype=torch.double)
        hei = torch.linspace(hei_min, hei_max, z_sample).to(self.device,dtype=torch.double)

        lat, lon, hei = torch.meshgrid(lat, lon, hei, indexing='ij') # 保持与旧版torch/numpy一致的行为

        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp, line = self.RPC_OBJ2PHOTO(lat, lon, hei)
        grid = torch.stack((samp, line, lat, lon, hei), dim=-1).to(self.device,dtype=torch.double)

        selected_grid = []
        for g in grid:
            flag = [g[0] < samp_min, g[0] > samp_max, g[1] < line_min, g[1] > line_max]
            if True in flag:
                continue
            else:
                selected_grid.append(g)

        if len(selected_grid) > 0:
            grid = torch.stack(selected_grid,dim=0).to(self.device,dtype=torch.double)
        else:
            grid = torch.empty(0, 5, device=self.device, dtype=torch.double)

        return grid

    def _solve_lstsq(self, ma, lv, x=None, k=1):
        """
        【私有】最小二乘解算器
        :param lv:
        :param ma: the Normal matrix
        :param x: init value
        :param k:
        :return:
        """
        assert ma.shape[0] == ma.shape[1], "ma with shape {} is not a square matrix.".format(ma.shape[0], ma.shape[1])
        
        if x is None:
            x = torch.zeros(ma.shape[0], dtype=torch.double, device=self.device)

        n = ma.shape[0]
        mak = ma.clone() #np.copy(ma)
        mak += k * torch.eye(n).to(self.device,dtype=torch.double)
        lk = lv.clone() #np.copy(lv)

        finish_time = 0

        for times in range(1000):
            try:
                x1 = torch.linalg.solve(mak,lk) #np.linalg.solve(mak, lk)
            except torch.linalg.LinAlgError:
                k *= 10
                mak = ma.clone() + k * torch.eye(n).to(self.device, dtype=torch.double)
                continue
                
            dif = torch.abs(x1 - x)#np.fabs(x1 - x)
            maxdif = torch.max(dif)
            x = x1
            lk = lv + k * x

            finish_time = times + 1
            # print(finish_time, maxdif)
            if maxdif < 1.0e-10:
                break
        return x, finish_time

    def Solve_Inverse_RPC(self, grid):
        samp, line, lat, lon, hei = torch.hsplit(grid,5) #np.hsplit(grid.copy(), 5)

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        # 归一化
        samp = samp - self.SAMP_OFF
        samp = samp / self.SAMP_SCALE
        line = line - self.LINE_OFF
        line = line / self.LINE_SCALE

        lat = lat - self.LAT_OFF
        lat = lat / self.LAT_SCALE
        lon = lon - self.LONG_OFF
        lon = lon / self.LONG_SCALE
        hei = hei - self.HEIGHT_OFF
        hei = hei / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp, line, hei)

        n_num = coef.shape[0]
        A = torch.zeros((n_num * 2, 78)).to(self.device,dtype=torch.double)
        A[0: n_num, 0:20] = - coef
        A[0: n_num, 20:39] = lat.reshape(-1, 1) * coef[:, 1:]
        A[n_num:, 39:59] = - coef
        A[n_num:, 59:78] = lon.reshape(-1, 1) * coef[:, 1:]

        l = torch.cat((lat, lon), -1) #np.concatenate((lat, lon), -1)
        l = -l

        ATA = torch.matmul(A.T, A)

        ATl = torch.matmul(A.T, l)

        # 使用重构后的类方法
        x, times = self._solve_lstsq(ATA, ATl)

        self.LATNUM = x[0:20]
        self.LATDEM[0] = 1.0
        self.LATDEM[1:20] = x[20:39]
        self.LONNUM = x[39:59]
        self.LONDEM[0] = 1.0
        self.LONDEM[1:20] = x[59:]

        return times

    def Calculate_Inverse_RPC(self):
        
        grid = self.Create_Virtual_3D_Grid()
        if grid.shape[0] == 0:
            return -1
        times = self.Solve_Inverse_RPC(grid)
        return times

    def Inverse_Adjust(self):
        R = self.adjust_params[:, :2]
        t = self.adjust_params[:, 2]   
        try:
            R_inv = torch.inverse(R) 
            t_new = -(R_inv @ t) 
            self.adjust_params_inv = torch.cat([R_inv, t_new.unsqueeze(1)], dim=1).to(torch.double)
        except torch.linalg.LinAlgError:
            print("警告: 仿射矩阵R不可逆。使用伪逆。")
            R_inv = torch.linalg.pinv(R)
            t_new = -(R_inv @ t)
            self.adjust_params_inv = torch.cat([R_inv, t_new.unsqueeze(1)], dim=1).to(torch.double)

    
    def Clear_Adjust(self):
        self.adjust_params = torch.tensor([
            [1.,0.,0.],
            [0.,1.,0.]
        ],dtype=torch.double,device=self.LNUM.device)
        self.Inverse_Adjust()

    def Update_Adjust(self,new_adjust_params:torch.Tensor):
        new_adjust_params = new_adjust_params.to(self.adjust_params.device).to(torch.double)
        def merge_adjust(A:torch.Tensor,B:torch.Tensor) -> torch.Tensor:
            device = A.device
            dtype = A.dtype

            # 创建用于扩展的行
            bottom_row = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)

            # 扩展到齐次坐标
            A_h = torch.cat([A, bottom_row], dim=0)
            B_h = torch.cat([B, bottom_row], dim=0)

            # 矩阵乘法，注意顺序 B @ A
            C_h = B_h @ A_h

            # 返回 (2, 3) 形式
            return C_h[:2, :]
        self.adjust_params = merge_adjust(self.adjust_params,new_adjust_params).to(self.adjust_params.device).to(torch.double)
        self.Inverse_Adjust()
    
    def Calculate_Adjust(self):
        corners = np.array([[0.,0.],[100.,0.],[0.,100.]],dtype=np.float32) #line samp
        offset_line = self.raw_adjust_params[0] + self.raw_adjust_params[1] * corners[:,1] + self.raw_adjust_params[2] * corners[:,0]
        offset_samp = self.raw_adjust_params[3] + self.raw_adjust_params[4] * corners[:,1] + self.raw_adjust_params[5] * corners[:,0]
        
        # 注意：这里的 offset_corners 计算逻辑是 (原始 - 偏移)
        # 这意味着 af_trans 是 (调整后 -> 原始) 的变换
        offset_corners = corners - np.stack([offset_line,offset_samp],axis=1) 
        
        af_trans = cv2.getAffineTransform(corners,offset_corners)
        self.Update_Adjust(torch.from_numpy(af_trans))

    def Merge_Adjust(self):
        """
        将当前的仿射变换参数 self.adjust_params "烘焙"（Bake）到
        RPC的主系数（LNUM, LDEM, SNUM, SDEM）中。
        这将导致正向和反向系数被重新计算，
        然后 adjust_params 将被重置为单位矩阵。
        """
        
        # 1. 检查是否需要合并
        identity_adjust = torch.tensor([
            [1.,0.,0.],
            [0.,1.,0.]
        ], dtype=torch.double, device=self.device)
        
        if torch.allclose(self.adjust_params, identity_adjust, atol=1e-8):
            return


        # 2. 生成 3D 虚拟格网 (物方)
        #    Create_Virtual_3D_Grid 内部会调用 RPC_OBJ2PHOTO
        #    RPC_OBJ2PHOTO 内部会 *自动应用* 当前的 self.adjust_params_inv
        #    因此，grid_obj 包含的就是我们需要的 (line_final, samp_final) 真值
        grid_obj = self.Create_Virtual_3D_Grid(xy_sample=50, z_sample=30) # 使用更密集的格网保证拟合精度

        if grid_obj.shape[0] == 0:
            return

        # 3. 提取坐标
        samp_target = grid_obj[:, 0]
        line_target = grid_obj[:, 1]
        lat = grid_obj[:, 2]
        lon = grid_obj[:, 3]
        hei = grid_obj[:, 4]

        # 4. 归一化坐标
        # 物方坐标归一化 (作为多项式输入)
        P = (lat - self.LAT_OFF) / self.LAT_SCALE
        L = (lon - self.LONG_OFF) / self.LONG_SCALE
        H = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        
        # 像方坐标归一化 (作为最小二乘的目标)
        # 我们重用 *旧* 的 offset/scale 来归一化，
        # 使得新系数的数值范围与旧系数保持一致，提高稳定性。
        line_target_norm = (line_target - self.LINE_OFF) / self.LINE_SCALE
        samp_target_norm = (samp_target - self.SAMP_OFF) / self.SAMP_SCALE

        # 5. 构建最小二乘系统 (求解新的正向系数)
        coef = self.RPC_PLH_COEF(P, L, H)
        n_num = coef.shape[0]
        
        # 求解 LNUM_new 和 LDEM_new[1:] (共39个未知数)
        # A_L * x_L = l_L
        # [coef, -line_norm * coef[:, 1:]] * [LNUM_new, LDEM_new[1:]] = line_norm * (LDEM_new[0]=1)
        A_L = torch.zeros((n_num, 39), dtype=torch.double, device=self.device)
        A_L[:, 0:20] = coef
        A_L[:, 20:39] = -line_target_norm.unsqueeze(-1) * coef[:, 1:]
        l_L = line_target_norm * coef[:, 0] # coef[:, 0] is 1.0
        
        ATA_L = A_L.T @ A_L
        ATl_L = A_L.T @ l_L
        
        # 求解 SNUM_new 和 SDEM_new[1:] (共39个未知数)
        # A_S * x_S = l_S
        A_S = torch.zeros((n_num, 39), dtype=torch.double, device=self.device)
        A_S[:, 0:20] = coef
        A_S[:, 20:39] = -samp_target_norm.unsqueeze(-1) * coef[:, 1:]
        l_S = samp_target_norm * coef[:, 0] # coef[:, 0] is 1.0
        
        ATA_S = A_S.T @ A_S
        ATl_S = A_S.T @ l_S

        # 6. 解算
        x_L, _ = self._solve_lstsq(ATA_L, ATl_L)
        x_S, _ = self._solve_lstsq(ATA_S, ATl_S)

        # 7. 更新 正向 RPC 系数
        self.LNUM = x_L[0:20].clone()
        self.LDEM[0] = 1.0
        self.LDEM[1:20] = x_L[20:39].clone()
        self.SNUM = x_S[0:20].clone()
        self.SDEM[0] = 1.0
        self.SDEM[1:20] = x_S[20:39].clone()

        # 8. 重置 adjust_params
        #    必须在 Calculate_Inverse_RPC 之前调用！
        #    因为 Calculate_Inverse_RPC 会调用 Create_Virtual_3D_Grid，
        #    那时必须使用新的正向模型 和 *零* 仿射变换。
        self.Clear_Adjust()

        # 9. 重新计算 反向 RPC 系数
        #    (因为正向模型已经改变，反向模型必须重新拟合)
        times = self.Calculate_Inverse_RPC()

    def RPC_PLH_COEF(self, P, L, H):
        n_num = P.shape[0]
        coef = torch.zeros((n_num, 20),dtype=torch.double,device=P.device)
        coef[:, 0] = 1.0
        coef[:, 1] = L
        coef[:, 2] = P
        coef[:, 3] = H
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        coef[:, 10] = P * L * H
        coef[:, 11] = L * L * L
        coef[:, 12] = L * P * P
        coef[:, 13] = L * H * H
        coef[:, 14] = L * L * P
        coef[:, 15] = P * P * P
        coef[:, 16] = P * H * H
        coef[:, 17] = L * L * H
        coef[:, 18] = P * P * H
        coef[:, 19] = H * H * H

        return coef
    
    def convert_tensor(self,arr,device):
        if isinstance(arr,torch.Tensor):
            return arr.to(dtype=torch.double,device=device)
        else:
            return torch.as_tensor(arr,dtype=torch.double,device=device)

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei, output_type='tensor'):
        """
        From (lat, lon, hei) to (samp, line) using the direct rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        lat = self.convert_tensor(inlat,self.device)
        lon = self.convert_tensor(inlon,self.device)
        hei = self.convert_tensor(inhei,self.device)
        
        # 确保输入是1D
        is_batched = lat.dim() > 0
        if not is_batched:
            lat = lat.unsqueeze(0)
            lon = lon.unsqueeze(0)
            hei = hei.unsqueeze(0)


        lat_norm = (lat - self.LAT_OFF) / self.LAT_SCALE
        lon_norm = (lon - self.LONG_OFF) / self.LONG_SCALE
        hei_norm = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(lat_norm, lon_norm, hei_norm)

        # # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp_norm = torch.sum(coef * self.SNUM,dim=-1) / torch.sum(coef * self.SDEM,dim=-1)
        line_norm = torch.sum(coef * self.LNUM,dim=-1) / torch.sum(coef * self.LDEM,dim=-1)

        samp = samp_norm * self.SAMP_SCALE + self.SAMP_OFF
        line = line_norm * self.LINE_SCALE + self.LINE_OFF

        transformed_points = torch.stack([line,samp],dim=-1) @ self.adjust_params_inv[:,:2].T + self.adjust_params_inv[:,2]
        line_final = transformed_points[:,0]
        samp_final = transformed_points[:,1]

        if not is_batched:
            line_final = line_final.squeeze(0)
            samp_final = samp_final.squeeze(0)

        if output_type == 'numpy':
            samp_final = samp_final.cpu().numpy()
            line_final = line_final.cpu().numpy()

        return samp_final, line_final

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei, output_type='tensor'):
        """
        From (samp, line, hei) to (lat, lon) using the inverse rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        hei = self.convert_tensor(inhei,self.device)
        samp = self.convert_tensor(insamp,self.device)
        line = self.convert_tensor(inline,self.device)
 
        # 确保输入是1D
        is_batched = samp.dim() > 0
        if not is_batched:
            samp = samp.unsqueeze(0)
            line = line.unsqueeze(0)
            hei = hei.unsqueeze(0)

        transformed_points = torch.stack([line,samp],dim=-1) @ self.adjust_params[:,:2].T + self.adjust_params[:,2]
        line_orig = transformed_points[:,0]
        samp_orig = transformed_points[:,1]

        samp_norm = (samp_orig - self.SAMP_OFF) / self.SAMP_SCALE
        line_norm = (line_orig - self.LINE_OFF) / self.LINE_SCALE
        hei_norm = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE


        coef = self.RPC_PLH_COEF(samp_norm, line_norm, hei_norm)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        lat_norm = torch.sum(coef * self.LATNUM, dim=-1) / torch.sum(coef * self.LATDEM, dim=-1)
        lon_norm = torch.sum(coef * self.LONNUM, dim=-1) / torch.sum(coef * self.LONDEM, dim=-1)

        lat = lat_norm * self.LAT_SCALE + self.LAT_OFF
        lon = lon_norm * self.LONG_SCALE + self.LONG_OFF
        
        if not is_batched:
            lat = lat.squeeze(0)
            lon = lon.squeeze(0)

        if output_type == 'numpy':
            lon = lon.cpu().numpy()
            lat = lat.cpu().numpy()

        return lat, lon
    
    def latlon2yx(self,latlon:torch.Tensor):
        """
        (lat,lon) -> (y,x) N,2
        """
        r = 6378137.
        lon_rad = latlon[:,1] * torch.pi / 180.
        lat_rad = latlon[:,0] * torch.pi / 180.
        x = r * lon_rad
        y = r * torch.log(torch.tan(torch.pi / 4. + lat_rad / 2.))
        return torch.stack([y,x],dim=-1)

    def yx2latlon(self,yx:torch.Tensor):
        """
        (y,x) -> (lat,lon) N,2
        """
        yx = self.convert_tensor(yx,self.device)
        r = 6378137.
        lon = (180. * yx[:,1]) / (torch.pi * r)
        lat = (2 * torch.atan(torch.exp(yx[:,0] / r)) - torch.pi * 0.5) * 180. / torch.pi
        return torch.stack([lat,lon],dim=-1)

    def RPC_XY2LINESAMP(self,x_in, y_in, h_in, output_type='tensor'):
        x = self.convert_tensor(x_in,self.device)
        y = self.convert_tensor(y_in,self.device)
        h = self.convert_tensor(h_in,self.device)
        
        is_batched = x.dim() > 0
        if not is_batched:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            h = h.unsqueeze(0)
            
        latlon = self.yx2latlon(torch.stack([y,x],dim=-1))
        samp,line = self.RPC_OBJ2PHOTO(latlon[:,0],latlon[:,1],h)
        
        if not is_batched:
            line = line.squeeze(0)
            samp = samp.squeeze(0)

        if output_type == 'numpy':
            line = line.cpu().numpy()
            samp = samp.cpu().numpy()
        
        return line,samp
    
    def RPC_LINESAMP2XY(self,line_in, samp_in, h_in, output_type='tensor'):
        line = self.convert_tensor(line_in,self.device)
        samp = self.convert_tensor(samp_in,self.device)
        h = self.convert_tensor(h_in,self.device)

        is_batched = line.dim() > 0
        if not is_batched:
            line = line.unsqueeze(0)
            samp = samp.unsqueeze(0)
            h = h.unsqueeze(0)

        lat,lon = self.RPC_PHOTO2OBJ(samp,line,h)
        yx = self.latlon2yx(torch.stack([lat,lon],dim=-1))
        y,x = yx[:,0],yx[:,1]
        
        if not is_batched:
            x = x.squeeze(0)
            y = y.squeeze(0)

        if output_type == 'numpy':
            x = x.cpu().numpy()
            y = y.cpu().numpy()

        return x,y
    
    def _project_xyh_to_linesamp_for_jacobian(self, xyh_tensor: torch.Tensor) -> torch.Tensor:
        """
        A helper function that takes a single (N, 3) tensor for xyh
        and returns a (N, 2) tensor for line/samp.
        This format is required by torch.autograd.functional.jacobian.
        """
        x, y, h = xyh_tensor[..., 0], xyh_tensor[..., 1], xyh_tensor[..., 2]
        
        # Chain the transformations
        latlon = self.yx2latlon(torch.stack([y, x], dim=-1))
        samp, line = self.RPC_OBJ2PHOTO(latlon[:, 0], latlon[:, 1], h)
        
        return torch.stack([line, samp], dim=-1)

    def _vjp_projection_core(self, mu_xyh: torch.Tensor, sigma_xyh: torch.Tensor):
        """
        【私有核心函数】执行VJP投影计算。
        假定输入的张量块能完全载入显存。
        """
        # 确保 mu_xyh 可以追踪梯度，这是 autograd.grad 的要求
        mu_xyh.requires_grad_(True)
        if mu_xyh.grad is not None:
            mu_xyh.grad.zero_()

        # --- 均值传播 ---
        line, samp = self.RPC_XY2LINESAMP(mu_xyh[:, 0], mu_xyh[:, 1], mu_xyh[:, 2])
        mu_linesamp = torch.stack([line, samp], dim=-1)

        # --- 方差传播 (VJP方法) ---
        grad_line, = torch.autograd.grad(
            outputs=line, inputs=mu_xyh,
            grad_outputs=torch.ones_like(line),
            create_graph=True, retain_graph=True,
        )
        grad_samp, = torch.autograd.grad(
            outputs=samp, inputs=mu_xyh,
            grad_outputs=torch.ones_like(samp),
            create_graph=False, retain_graph=False, # 最后一次调用可以关闭图保留
        )
        
        var_xyh = sigma_xyh.pow(2)
        var_line = (grad_line.pow(2) * var_xyh).sum(dim=-1)
        var_samp = (grad_samp.pow(2) * var_xyh).sum(dim=-1)
        var_linesamp = torch.stack([var_line, var_samp], dim=-1)
            
        return mu_linesamp, var_linesamp

    def xy_distribution_to_linesamp(self, mu_xyh: torch.Tensor, sigma_xyh: torch.Tensor, chunk_size: int = 524288):
        is_batched = mu_xyh.dim() == 2
        if not is_batched:
            mu_xyh = mu_xyh.unsqueeze(0)
            sigma_xyh = sigma_xyh.unsqueeze(0)
        
        num_points = mu_xyh.shape[0]

        if num_points <= chunk_size:
            mu_linesamp, var_linesamp = self._vjp_projection_core(mu_xyh, sigma_xyh)
        else:
            mu_results = []
            var_results = []
            
            mu_chunks = torch.split(mu_xyh, chunk_size)
            sigma_chunks = torch.split(sigma_xyh, chunk_size)

            for mu_chunk, sigma_chunk in zip(mu_chunks, sigma_chunks):
                mu_chunk_out, var_chunk_out = self._vjp_projection_core(mu_chunk, sigma_chunk)
                
                mu_results.append(mu_chunk_out.detach())
                var_results.append(var_chunk_out.detach())

            mu_linesamp = torch.cat(mu_results, dim=0)
            var_linesamp = torch.cat(var_results, dim=0)
        
        sigma_linesamp = torch.sqrt(var_linesamp)
        if not is_batched:
            mu_linesamp = mu_linesamp.squeeze(0)
            sigma_linesamp = sigma_linesamp.squeeze(0)

        return mu_linesamp, sigma_linesamp
    

    def to_gpu(self,device = None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                print("CUDA not available, using CPU.")
                device = 'cpu'
        
        self.device = torch.device(device)
        
        self.LINE_OFF = self.LINE_OFF.to(self.device)
        self.SAMP_OFF = self.SAMP_OFF.to(self.device)
        self.LAT_OFF = self.LAT_OFF.to(self.device)
        self.LONG_OFF = self.LONG_OFF.to(self.device)
        self.HEIGHT_OFF = self.HEIGHT_OFF.to(self.device)
        self.LINE_SCALE = self.LINE_SCALE.to(self.device)
        self.SAMP_SCALE = self.SAMP_SCALE.to(self.device)
        self.LAT_SCALE = self.LAT_SCALE.to(self.device)
        self.LONG_SCALE = self.LONG_SCALE.to(self.device)
        self.HEIGHT_SCALE = self.HEIGHT_SCALE.to(self.device)

        self.LNUM = self.LNUM.to(self.device)
        self.LDEM = self.LDEM.to(self.device)
        self.SNUM = self.SNUM.to(self.device)
        self.SDEM = self.SDEM.to(self.device)

        self.LATNUM = self.LATNUM.to(self.device)
        self.LATDEM = self.LATDEM.to(self.device)
        self.LONNUM = self.LONNUM.to(self.device)
        self.LONDEM = self.LONDEM.to(self.device)

        self.adjust_params = self.adjust_params.to(self.device)
        self.adjust_params_inv = self.adjust_params_inv.to(self.device)

    def save_rpc_to_file(self, filepath):
        """
        Save the direct and inverse rpc to a file
        :param filepath: where to store the file
        :return:
        """
        # 确保所有参数都在CPU上以便转换为 .item()
        original_device = self.device
        if self.device.type != 'cpu':
            self.to_gpu('cpu')

        addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                     'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
                     'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:', 'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:',
                     'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:', 'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:',
                     'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:', 'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:',
                     'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:', 'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:',
                     'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:', 'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:',
                     'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:', 'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:',
                     'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:', 'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:',
                     'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:', 'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:',
                     'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:', 'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:',
                     'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:', 'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:',
                     'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:', 'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:',
                     'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:', 'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:',
                     'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:', 'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:',
                     'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:', 'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:',
                     'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:', 'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:',
                     'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:', 'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:',
                     'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:', 'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:',
                     'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:', 'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:',
                     'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:', 'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:',
                     'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:', 'LAT_NUM_COEFF_1:', 'LAT_NUM_COEFF_2:',
                     'LAT_NUM_COEFF_3:', 'LAT_NUM_COEFF_4:', 'LAT_NUM_COEFF_5:', 'LAT_NUM_COEFF_6:',
                     'LAT_NUM_COEFF_7:', 'LAT_NUM_COEFF_8:', 'LAT_NUM_COEFF_9:', 'LAT_NUM_COEFF_10:',
                     'LAT_NUM_COEFF_11:', 'LAT_NUM_COEFF_12:', 'LAT_NUM_COEFF_13:', 'LAT_NUM_COEFF_14:',
                     'LAT_NUM_COEFF_15:', 'LAT_NUM_COEFF_16:', 'LAT_NUM_COEFF_17:', 'LAT_NUM_COEFF_18:',
                     'LAT_NUM_COEFF_19:', 'LAT_NUM_COEFF_20:', 'LAT_DEN_COEFF_1:', 'LAT_DEN_COEFF_2:',
                     'LAT_DEN_COEFF_3:', 'LAT_DEN_COEFF_4:', 'LAT_DEN_COEFF_5:', 'LAT_DEN_COEFF_6:',
                     'LAT_DEN_COEFF_7:', 'LAT_DEN_COEFF_8:', 'LAT_DEN_COEFF_9:', 'LAT_DEN_COEFF_10:',
                     'LAT_DEN_COEFF_11:', 'LAT_DEN_COEFF_12:', 'LAT_DEN_COEFF_13:', 'LAT_DEN_COEFF_14:',
                     'LAT_DEN_COEFF_15:', 'LAT_DEN_COEFF_16:', 'LAT_DEN_COEFF_17:', 'LAT_DEN_COEFF_18:',
                     'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:', 'LONG_NUM_COEFF_1:', 'LONG_NUM_COEFF_2:',
                     'LONG_NUM_COEFF_3:', 'LONG_NUM_COEFF_4:', 'LONG_NUM_COEFF_5:', 'LONG_NUM_COEFF_6:',
                     'LONG_NUM_COEFF_7:', 'LONG_NUM_COEFF_8:', 'LONG_NUM_COEFF_9:', 'LONG_NUM_COEFF_10:',
                     'LONG_NUM_COEFF_11:', 'LONG_NUM_COEFF_12:', 'LONG_NUM_COEFF_13:', 'LONG_NUM_COEFF_14:',
                     'LONG_NUM_COEFF_15:', 'LONG_NUM_COEFF_16:', 'LONG_NUM_COEFF_17:', 'LONG_NUM_COEFF_18:',
                     'LONG_NUM_COEFF_19:', 'LONG_NUM_COEFF_20:', 'LONG_DEN_COEFF_1:', 'LONG_DEN_COEFF_2:',
                     'LONG_DEN_COEFF_3:', 'LONG_DEN_COEFF_4:', 'LONG_DEN_COEFF_5:', 'LONG_DEN_COEFF_6:',
                     'LONG_DEN_COEFF_7:', 'LONG_DEN_COEFF_8:', 'LONG_DEN_COEFF_9:', 'LONG_DEN_COEFF_10:',
                     'LONG_DEN_COEFF_11:', 'LONG_DEN_COEFF_12:', 'LONG_DEN_COEFF_13:', 'LONG_DEN_COEFF_14:',
                     'LONG_DEN_COEFF_15:', 'LONG_DEN_COEFF_16:', 'LONG_DEN_COEFF_17:', 'LONG_DEN_COEFF_18:',
                     'LONG_DEN_COEFF_19:', 'LONG_DEN_COEFF_20:']
        addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                     'meters']
        
        # addition2 = ['CL0:','CLS:','CLL:','CS0:','CSS:','CSL:']

        # corection_params = [self.adjust_params[0,2],self.adjust_params[0,1],self.adjust_params[0,0] - 1.,self.adjust_params[1,2],self.adjust_params[1,1] - 1.,self.adjust_params[1,0]]

        text = ""

        text += addition0[0] + " " + str(self.LINE_OFF.item()) + " " + addition1[0] + "\n"
        text += addition0[1] + " " + str(self.SAMP_OFF.item()) + " " + addition1[1] + "\n"
        text += addition0[2] + " " + str(self.LAT_OFF.item()) + " " + addition1[2] + "\n"
        text += addition0[3] + " " + str(self.LONG_OFF.item()) + " " + addition1[3] + "\n"
        text += addition0[4] + " " + str(self.HEIGHT_OFF.item()) + " " + addition1[4] + "\n"
        text += addition0[5] + " " + str(self.LINE_SCALE.item()) + " " + addition1[5] + "\n"
        text += addition0[6] + " " + str(self.SAMP_SCALE.item()) + " " + addition1[6] + "\n"
        text += addition0[7] + " " + str(self.LAT_SCALE.item()) + " " + addition1[7] + "\n"
        text += addition0[8] + " " + str(self.LONG_SCALE.item()) + " " + addition1[8] + "\n"
        text += addition0[9] + " " + str(self.HEIGHT_SCALE.item()) + " " + addition1[9] + "\n"

        for i in range(10, 30):
            text += addition0[i] + " " + str(self.LNUM[i - 10].item()) + "\n"
        for i in range(30, 50):
            text += addition0[i] + " " + str(self.LDEM[i - 30].item()) + "\n"
        for i in range(50, 70):
            text += addition0[i] + " " + str(self.SNUM[i - 50].item()) + "\n"
        for i in range(70, 90):
            text += addition0[i] + " " + str(self.SDEM[i - 70].item()) + "\n"
        for i in range(90, 110):
            text += addition0[i] + " " + str(self.LATNUM[i - 90].item()) + "\n"
        for i in range(110, 130):
            text += addition0[i] + " " + str(self.LATDEM[i - 110].item()) + "\n"
        for i in range(130, 150):
            text += addition0[i] + " " + str(self.LONNUM[i - 130].item()) + "\n"
        for i in range(150, 170):
            text += addition0[i] + " " + str(self.LONDEM[i - 150].item()) + "\n"
        
        # 重要：保存时不再写出 RFM 块，因为它们已经被合并了。
        # (或者，如果需要，可以写出一个全为0的 RFM 块)
        # 这里选择不写出。

        f = open(filepath, "w")
        f.write(text)
        f.close()
        
        # 恢复设备
        if original_device.type != 'cpu':
            self.to_gpu(original_device)


def load_rpc(rpc_path:str,to_gpu=False) -> RPCModelParameterTorch:
    rpc = RPCModelParameterTorch()
    rpc.load_from_file(rpc_path)
    if to_gpu:
        rpc.to_gpu()
    return rpc

def project_linesamp(rpc1:RPCModelParameterTorch,rpc2:RPCModelParameterTorch,lines,samps,heights,output_type='tensor'):
    lat,lon = rpc1.RPC_PHOTO2OBJ(samps,lines,heights)
    samps,lines = rpc2.RPC_OBJ2PHOTO(lat,lon,heights,output_type)
    return lines,samps
