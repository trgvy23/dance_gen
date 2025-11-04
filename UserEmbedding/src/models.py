import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from src.backbone import MotionBERTBackbone, VideoPrismBackbone

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, attn_weights = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, attn_weights = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)

        return tgt, attn_weights

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


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


class UserEmbeddingNet(nn.Module):
    def __init__(self, motionbert: MotionBERTBackbone, video_prism: VideoPrismBackbone):
        super(UserEmbeddingNet, self).__init__()
        self.motionbert = motionbert
        self.video_prism = video_prism

            # for p in self.motionbert.parameters():
            #     p.requires_grad = False
            # for p in self.video_prism.parameters():
            #     p.requires_grad = False
            # self.motionbert.eval()
            # self.video_prism.eval()
        
        self.mean_pool_mlp = MeanPoolMLP(d_in=512, d_hidden=512, d_out=256, p_drop=0.1)

        # TODO: Úm ba la xì bùa

    def forward(self, video, pose_est):
        """
        x: [B, T, input_dim]
        return: [B, embed_dim]
        """
        # with torch.no_grad():
        #     # video_feat = self.video_prism(video)
        pose_feat = self.motionbert(pose_est)
        video_feat = self.video_prism(video)
        pose_feat = self.mean_pool_mlp(pose_feat)
        
        print(f'Pose feature shape: {pose_feat.shape}')
        print(f'Video feature shape: {video_feat.shape}')
        
        embeddings = torch.cat([pose_feat, video_feat], dim=-1)
        
        return embeddings
