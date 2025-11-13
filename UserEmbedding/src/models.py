from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from src.backbone import MotionBERTBackbone
from src.rotary_embedding import RotaryEmbedding

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TemporalSelfAttention(nn.Module):
    """
    Stacks several TransformerEncoderLayer blocks.
    Used for per-modality temporal encoding.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int = 2,
        p_drop: float = 0.1,
        use_rotary: bool = False,
    ):
        super().__init__()
        rotary = RotaryEmbedding(dim=d_model) if use_rotary else None

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=p_drop,
                batch_first=True,
                norm_first=True,
                rotary=rotary,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: Tensor,                       # [B, T, D]
        pad_mask: Optional[Tensor] = None,  # [B, T], True = padding
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        return x
    
class CrossAttentionBlock(nn.Module):
    """
    One cross-attention block: Q attends to KV, with LayerNorm + residual.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        p_drop: float = 0.1,
        use_rotary: bool = False,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=p_drop, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

        self.use_rotary = use_rotary
        self.rotary = RotaryEmbedding(dim=d_model) if use_rotary else None

    def forward(
        self,
        q: Tensor,                  # [B, T_q, D]
        k: Tensor,                  # [B, T_k, D]
        v: Tensor,                  # [B, T_k, D]
        key_padding_mask: Optional[Tensor] = None,  # [B, T_k], True = pad
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q_norm = self.norm_q(q)

        if self.use_rotary:
            q_rot = self.rotary.rotate_queries_or_keys(q_norm)
            k_rot = self.rotary.rotate_queries_or_keys(k)
        else:
            q_rot, k_rot = q_norm, k

        attn_out, _ = self.mha(
            q_rot, k_rot, v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = q_norm + self.dropout(attn_out)
        x = self.norm_o(x)
        return x              # [B, T_q, D]
    
class TemporalAttentionPoolingHead(nn.Module):
    """
    Pools over time with attention, then MLP.
    """
    def __init__(self, d_model: int, d_out: int, p_drop: float = 0.1):
        super().__init__()
        self.att_pool = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * 2, d_out),
        )

    def forward(
        self,
        x: Tensor,                      # [B, T, D] (pose timeline)
        pad_mask: Optional[Tensor] = None,   # [B, T], True = padding
    ) -> Tensor:
        scores = self.att_pool(x).squeeze(-1)   # [B, T]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        z = (weights * x).sum(dim=1)                           # [B, D]
        z = self.mlp(z)                                        # [B, d_out]
        return z

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MaskBackbone(nn.Module):
    """
    Takes [B, T, H, W] probability masks and returns [B, T, D].
    """
    def __init__(self, d_out: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),  # [B, 1, H, W] -> [B, 16, H/2, W/2]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    # -> [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, d_out)

    def forward(self, masks: Tensor) -> Tensor:
        """
        masks: [B, T, H, W] float in [0,1]
        returns: [B, T, d_out]
        """
        B, T, H, W = masks.shape
        x = masks.view(B * T, 1, H, W)          # [B*T, 1, H, W]
        x = self.conv(x)                        # [B*T, 64, 1, 1]
        x = x.view(B * T, -1)                   # [B*T, 64]
        x = self.fc(x)                          # [B*T, d_out]
        x = x.view(B, T, -1)                    # [B, T, d_out]
        return x


class UserEmbeddingNet(nn.Module):
    def __init__(self, motionbert: MotionBERTBackbone,
                 num_dancer_class: int,
                 num_gerne_class: int):
        super().__init__()

        self.motionbert = motionbert

        d_pose_raw  = 512    # per-joint dim from MotionBERT
        d_video_in  = 768    # VideoPrism
        d_model     = 256
        d_embed     = 256
        n_heads     = 8
        p_drop      = 0.1

        # 1) joints → per-frame pose feature
        self.joint_flat_proj = nn.Linear(17 * 512, d_pose_raw)   # if J=17

        # 2) project to common model dim
        self.pose_proj  = nn.Linear(d_pose_raw, d_model)
        self.video_proj = nn.Linear(d_video_in, d_model)
        
        self.mask_backbone = MaskBackbone(d_out=d_model)

        # 3) temporal encoders
        self.pose_encoder = TemporalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            p_drop=p_drop,
            use_rotary=True,
        )
        self.video_encoder = TemporalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            p_drop=p_drop,
            use_rotary=True,
        )
        self.mask_encoder = TemporalSelfAttention(
            d_model=d_model, n_heads=n_heads, n_layers=2,
            p_drop=p_drop, use_rotary=True,
        )

        # 4) cross-attention: pose (Q) ← video (K,V)
        self.cross_attn_video = CrossAttentionBlock(
            d_model=d_model, n_heads=n_heads, p_drop=p_drop, use_rotary=True,
        )
        self.cross_attn_mask = CrossAttentionBlock(
            d_model=d_model, n_heads=n_heads, p_drop=p_drop, use_rotary=True,
        )

        # 5) temporal attention pooling over pose timeline + MLP
        self.pool_head = TemporalAttentionPoolingHead(
            d_model=d_model,
            d_out=d_embed,
            p_drop=p_drop,
        )

        # 6) classification heads
        self.dancer_predictor = MLP(d_embed, 512, num_dancer_class, 3)
        self.gerne_predictor  = MLP(d_embed, 512, num_gerne_class, 3)

    def forward(
        self,
        video_feat: Tensor,   # [B, T_v, 768]
        video_mask: Tensor,   
        pose_est:   Tensor,   # whatever MotionBERT expects
        pose_pad_mask:  Optional[Tensor] = None,  # [B, T_p]
        video_pad_mask: Optional[Tensor] = None,  # [B, T_v]
    ):
        # ---- Pose branch ----
        with torch.no_grad():
            pose_feat = self.motionbert(pose_est)    # [B, T_p, J, 512]

        B, T, J, D = pose_feat.shape
        pose_feat = pose_feat.view(B, T, J * D)           # [B, T, 17*512]
        pose_feat = self.joint_flat_proj(pose_feat)       # [B, T, 512]
        pose_feat = self.pose_proj(pose_feat)        # [B, T_p, D]
        pose_feat = self.pose_encoder(
            pose_feat, pad_mask=pose_pad_mask
        )                                            # [B, T_p, D]

        # ---- Video branch ----
        video_feat = self.video_proj(video_feat)     # [B, T_v, D]
        video_feat = self.video_encoder(
            video_feat, pad_mask=video_pad_mask
        )                                            # [B, T_v, D]
        
        # ---- Mask branch ----
        mask_feat = self.mask_backbone(video_mask)     # [B, T_v, D]
        mask_feat = self.mask_encoder(mask_feat)

        # ---- Cross-attention: pose (Q) attends to video (V) ----
        fused_from_video = self.cross_attn_video(
            q=pose_feat,
            k=video_feat,
            v=video_feat,
        )                                             # [B, T_p, D]
        
        # ---- Cross-attention: pose (Q) attends to mask (M) ----
        fused_from_mask = self.cross_attn_mask(
            q=pose_feat,
            k=mask_feat,
            v=mask_feat,
        )
        
        fused_pose = fused_from_video + fused_from_mask
        
        # --- Final fused pose ---
        # fused_pose = fused_pose + fused_pose_mask

        # ---- Temporal attention pooling over pose timeline ----
        embeddings = self.pool_head(
            fused_pose,
            pad_mask=pose_pad_mask,
        )                                            # [B, 256]

        embeddings = F.normalize(embeddings, p=2, dim=1)

        # ---- Classification heads ----
        dancer_logits = self.dancer_predictor(embeddings)
        gerne_logits  = self.gerne_predictor(embeddings)

        return embeddings, dancer_logits, gerne_logits