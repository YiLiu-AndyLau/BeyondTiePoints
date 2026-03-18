import torch
import torch.nn as nn

class AffineModel(nn.Module):
    """
    Wrap affine parameters R and T as a PyTorch module.
    """
    def __init__(self):
        super().__init__()
        self.R = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.T = nn.Parameter(torch.tensor([0., 0.], dtype=torch.float32))

    def forward(self):
        return torch.concatenate([self.R, self.T.unsqueeze(-1)], dim=-1)

class BundleAffineModel(nn.Module):
    """
    Manage all learnable affine transforms for N images.
    Image 0 is treated as the fixed anchor.
    """
    def __init__(self, num_images):
        super().__init__()
        self.num_images = num_images
        
        self.models = nn.ModuleList()
        for _ in range(num_images - 1):
            self.models.append(AffineModel())
            
    def get_affine(self, index: int) -> torch.Tensor:
        """
        Return affine matrix for image index.
        index == 0 returns identity.
        index > 0 returns its learnable affine matrix.
        """
        if index == 0:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if len(self.models) > 0:
                device = self.models[0].R.device
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
                                dtype=torch.float32, device=device)
        else:
            return self.models[index - 1]()
