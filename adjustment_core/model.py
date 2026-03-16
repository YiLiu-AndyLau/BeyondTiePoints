import torch
import torch.nn as nn

# DDP Step 2: 将需要优化的参数封装为 nn.Module
class AffineModel(nn.Module):
    """
    将仿射变换参数R和T封装成一个PyTorch模块，以便DDP管理。
    """
    def __init__(self):
        super().__init__()
        self.R = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.T = nn.Parameter(torch.tensor([0., 0.], dtype=torch.float32))

    def forward(self):
        return torch.concatenate([self.R, self.T.unsqueeze(-1)], dim=-1)

class BundleAffineModel(nn.Module):
    """
    管理所有N-1个可学习仿射变换的模型。
    img_0 被假定为锚点(anchor)，其变换固定为单位矩阵。
    """
    def __init__(self, num_images):
        super().__init__()
        self.num_images = num_images
        
        # 我们有 N 张影像, 但只为 img_1 ... img_N-1 创建可学习模型
        self.models = nn.ModuleList()
        for _ in range(num_images - 1):
            self.models.append(AffineModel())
            
    def get_affine(self, index: int) -> torch.Tensor:
        """
        获取第 index 张影像的仿射变换矩阵.
        index 0 (锚点) 返回固定的单位矩阵.
        index > 0   返回其对应的可学习矩阵.
        """
        if index == 0:
            # 返回一个固定的、float32的单位仿射矩阵
            # 它需要和 model 在同一个 device 上 (通过第一个模型获取)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if len(self.models) > 0:
                device = self.models[0].R.device
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
                                dtype=torch.float32, device=device)
        else:
            # 返回第 (index - 1) 个子模型的仿射矩阵
            # 调用子模型的 forward()
            return self.models[index - 1]()
