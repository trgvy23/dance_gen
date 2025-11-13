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
        self.pose_proj = nn.Linear(d_pose, d_model)
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
        q = self.pose_proj(pose_seq)  # [B, T_p, d_model]
        k = self.video_proj(video_seq)  # [B, T_v, d_model]
        v = k

        q = self.norm_q(q)

        if self.use_rotary:
            q = self.rotary.rotate_queries_or_keys(q)
            k = self.rotary.rotate_queries_or_keys(k)

        # pose attends to video
        attn_out, _ = self.cross_attn(q, k, v)  # [B, T_p, d_model]
        x = self.norm_o(q + attn_out)  # residual

        # global pooling + MLP to final embedding
        x = x.mean(dim=1)  # [B, d_model]
        x = self.mlp(x)  # [B, d_out]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class UserEmbeddingNet(nn.Module):
    def __init__(
        self, motionbert: MotionBERTBackbone, num_dancer_class, num_genre_class
    ):
        super(UserEmbeddingNet, self).__init__()
        self.motionbert = motionbert

        self.dancer_predictor = MLP(256, 512, num_dancer_class, 3)
        self.genre_predictor = MLP(256, 512, num_genre_class, 3)

        self.fusion = CrossAttentionFusion(
            d_pose=512,
            d_video=768,  # VideoPrism dim
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

        pose_feat = pose_feat.mean(dim=2)  # [B, 243, 512]
        # video_feat: [B, 16 * 16 * F (243), 768] -> [B, 62208, 768]

        embeddings = self.fusion(pose_feat, video_feat)  # [B, 256]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        dancer_labels = self.dancer_predictor(embeddings)  # [B, ]
        genre_labels = self.genre_predictor(embeddings)  # [B, ]

        return embeddings, dancer_labels, genre_labels
