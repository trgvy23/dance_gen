from functools import partial
import jax
import jax.numpy as jnp
from videoprism import models as vp
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import torch
from src.MotionBert.DSTformer import DSTformer


class MotionBERTBackbone(nn.Module):
    def __init__(
        self,
    ):
        super(MotionBERTBackbone, self).__init__()
        # TODO: hardcode args for MotionBERT for now: https://github.com/Walter0807/MotionBERT/blob/main/configs/pose3d/MB_ft_h36m_global_lite.yaml

        print("Initializing MotionBERTBackbone...")
        self.dstformer = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=243,
            num_joints=17,
        )

        # TODO: load pretrained MotionBERT weights
        if torch.cuda.is_available():
            self.dstformer = nn.DataParallel(self.dstformer)
            self.dstformer = self.dstformer.cuda()

        print(
            "Loading checkpoint",
            "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/checkpoint/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        )
        checkpoint = torch.load(
            "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/checkpoint/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
            map_location=lambda storage, loc: storage,
        )
        self.dstformer.load_state_dict(checkpoint["model_pos"], strict=True)

        self.dstformer.eval()

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, T, embed_dim]
        """
        out = self.dstformer(x)

        assert out.dim() == 4  # [B, F, S, D]

        # TODO: do we need an MLP here?
        # out = out.mean(dim=1)  # [B, D]
        # # out = F.normalize(out, p=2, dim=1)s
        # print('MotionBERTBackbone output shape:', out.shape)

        return out
