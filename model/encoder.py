import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PositionalEncoding(nn.Module):
    """二维正弦位置编码，为特征图注入空间位置信息"""
    def __init__(self, dim: int, max_shape: tuple = (256, 256)):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(dim, *max_shape)
        device = pe.device
        y_position = torch.arange(0, max_shape[0], dtype=torch.float32, device=device).unsqueeze(1)
        x_position = torch.arange(0, max_shape[1], dtype=torch.float32, device=device).unsqueeze(0)

        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(torch.log(torch.tensor(10000.0)) / dim))
        
        pe[0::2, :, :] = torch.sin(y_position * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::2, :, :] = torch.cos(y_position * div_term.unsqueeze(1).unsqueeze(2))
        
        pe[0::2, :, :] += torch.sin(x_position * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::2, :, :] += torch.cos(x_position * div_term.unsqueeze(1).unsqueeze(2))

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, D, H, W) 的输入特征图
        Returns:
            torch.Tensor: 添加了位置编码的特征图
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


# --- 核心模块 1: 增强型Encoder ---
class AttentionBlock(nn.Module):
    """一个通用的注意力模块，可用于自注意力或交叉注意力"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, key, value (torch.Tensor): (B, N, D) shape, N = H * W
        """
        B, N, D = query.shape
        
        # 线性投影并切分为多头
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_scores, dim=-1)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # 输出线性投影
        return self.out_proj(x)

class Adapter(nn.Module):
    def __init__(self,input_channels = 512,output_channels = 512,pos_embed = False,unitize = True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels,self.input_channels // 4,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 4,self.output_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,1,1,0),
        )

        self.pos_embed = pos_embed
        self.pos_encoder = PositionalEncoding(dim=output_channels)
        
        self.unitize = unitize

        self.self_attention_block = AttentionBlock(dim=output_channels, num_heads=8)
        self.norm = nn.LayerNorm(output_channels)

        self.conf_head = nn.Sequential(
            nn.Conv2d(self.input_channels,self.input_channels // 4,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 4, self.input_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 16, 1 ,1,1,0),
            nn.Sigmoid()
        )
    def forward(self,x):
        raw_feat = self.cnn(x)
        B,D,H,W = raw_feat.shape
        if self.pos_embed:
            feat_with_pos = self.pos_encoder(raw_feat)
            feat_seq = feat_with_pos.flatten(2).transpose(1,2)
        else:
            feat_seq = raw_feat.flatten(2).transpose(1,2)
        attn_output = self.self_attention_block(feat_seq, feat_seq, feat_seq)
        attended_sequence = self.norm(feat_seq + attn_output)
        feat = attended_sequence.transpose(1, 2).view(B, D, H, W)
        if self.unitize:
            feat = F.normalize(feat,dim=1)
        conf = self.conf_head(x)
        # feat = F.normalize(feat,dim=1)
        return feat,conf
    
    

class EncoderDino(nn.Module):

    def __init__(self,dino_weight_path,output_channels=512,verbose = 1,layers = [5,11,17,23],adapter_pos_embed = False,unitize = True,upsample_times = 2,use_adapter = True,use_conf = True):
        super().__init__()
        self.verbose = verbose
        self.layers = layers
        self.SAMPLE_FACTOR = 16 // (2 ** upsample_times)
        self.upsample_times = upsample_times
        self.input_channels = 3
        self.output_channels = output_channels

        self.backbone = torch.hub.load('./dinov3','dinov3_vitl16',source='local',weights=dino_weight_path)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.adapter = Adapter(input_channels=1024 * len(layers),output_channels=output_channels,pos_embed=adapter_pos_embed,unitize=unitize)

        self.use_adapter = use_adapter
        self.use_conf = use_conf


    def forward(self, x):
        B = x.shape[0]
        H,W = x.shape[-2:]

        feat_multilayers = self.backbone.get_intermediate_layers(x = x, n = self.layers)
        feat_backbone = torch.cat(feat_multilayers,dim=-1)
        feat_backbone = feat_backbone.reshape(B,H // 16,W // 16,-1).permute(0,3,1,2)
        feat,conf = self.adapter(feat_backbone)
        if not self.use_conf:
            conf = torch.full(conf.shape,0.5,device=conf.device,dtype=conf.dtype)
        if not self.use_adapter:
            feat = feat_multilayers[-1].reshape(B,H // 16,W // 16,-1).permute(0,3,1,2)            

        for i in range(self.upsample_times):
            feat = F.interpolate(feat,scale_factor=2,mode='bilinear')
            conf = F.interpolate(conf,scale_factor=2,mode='bilinear')
        
        return feat,conf
    
    def unfreeze_backbone(self,layers:List[int] = []):
        parameters = []
        for layer in layers:
            block = self.backbone.blocks[layer]
            block.requires_grad_(True)
            parameters.extend(list(block.parameters()))
            if self.verbose > 0:
                print(f"Unfreeze backbone layer {layer}")
        return parameters
    
    def load_adapter(self,adapter_path:str):
        self.adapter.load_state_dict({k.replace("module.",""):v for k,v in torch.load(adapter_path,map_location='cpu').items()},strict=True)
    
    def save_adapter(self,output_path:str):
        state_dict = {k:v.detach().cpu() for k,v in self.adapter.state_dict().items()}
        torch.save(state_dict,output_path)

    def save_backbone(self,output_path:str):
        state_dict = {k:v.detach().cpu() for k,v in self.backbone.state_dict().items()}
        torch.save(state_dict,output_path)