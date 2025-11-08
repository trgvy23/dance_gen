import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from src.backbone import MotionBERTBackbone
from src.rotary_embedding import RotaryEmbedding

class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        d_pose: int,
        d_video: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_out: int = 256,
        p_drop: float = 0.1,
        use_rotary: bool = False,
    ):
        super().__init__()
        self.pose_proj  = nn.Linear(d_pose,  d_model)
        self.video_proj = nn.Linear(d_video, d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=p_drop, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * 2, d_out),
        )

        self.use_rotary = use_rotary
        self.rotary = RotaryEmbedding(dim=d_model) if use_rotary else None

    def forward(self, pose_seq, video_seq):
        # pose_seq:  [B, T_p, d_pose]
        # video_seq: [B, T_v, d_video]
        q = self.pose_proj(pose_seq)     # [B, T_p, d_model]
        k = self.video_proj(video_seq)   # [B, T_v, d_model]
        v = k

        q = self.norm_q(q)

        if self.use_rotary:
            q = self.rotary.rotate_queries_or_keys(q)
            k = self.rotary.rotate_queries_or_keys(k)

        # pose attends to video
        attn_out, _ = self.cross_attn(q, k, v)   # [B, T_p, d_model]
        x = self.norm_o(q + attn_out)            # residual

        # global pooling + MLP to final embedding
        x = x.mean(dim=1)                        # [B, d_model]
        x = self.mlp(x)                          # [B, d_out]
        return x



class MeanPoolMLP(nn.Module):
    def __init__(
        self, d_in: int, d_hidden: int = 512, d_out: int = 256, p_drop: float = 0.1
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):  # x: [B, F, J, D]
        x = x.mean(dim=(1, 2))  # [B, D]   (temporal + joint mean)
        x = self.mlp(x)  # [B, E]
        return F.normalize(x, dim=-1)


class VideoMeanPoolMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        # x: [B, T, D]
        x = x.mean(dim=1)  # [B, D]  (here D = d_in = 768)
        x = self.mlp(x)  # [B, d_out]  (256)
        return x


class UserEmbeddingNet(nn.Module):
    def __init__(self, motionbert: MotionBERTBackbone):
        super(UserEmbeddingNet, self).__init__()
        self.motionbert = motionbert

        # for p in self.motionbert.parameters():
        #     p.requires_grad = False
        # for p in self.video_prism.parameters():
        #     p.requires_grad = False
        # self.motionbert.eval()
        # self.video_prism.eval()

        # self.mean_pool_mlp = MeanPoolMLP(d_in=512, d_hidden=512, d_out=256, p_drop=0.1)
        # self.video_mean_pool_mlp = VideoMeanPoolMLP(
        #     d_in=768, d_hidden=512, d_out=256, p_drop=0.1
        # )
        
        self.fusion = CrossAttentionFusion(
            d_pose=512,
            d_video=768,   # VideoPrism dim
            d_model=512,
            n_heads=8,
            d_out=256,
            p_drop=0.1,
            use_rotary=True,
        )

    def forward(self, video_feat, pose_est):
        """
        x: [B, T, input_dim]
        return: [B, embed_dim]
        """
        with torch.no_grad():
            pose_feat = self.motionbert(pose_est)  # [B, 243, 17, 512]
            
        #TODO: projection layer for each input
            
        # pose_feat = self.mean_pool_mlp(pose_feat)
        # video_feat = self.video_mean_pool_mlp(video_feat)
        
        pose_feat  = pose_feat.mean(dim=2)        # [B, 243, 512]
        # video_feat: [B, 16 * 16 * F (243), 768] -> [B, 62208, 768]
        
        embeddings = self.fusion(pose_feat, video_feat)  # [B, 256]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
