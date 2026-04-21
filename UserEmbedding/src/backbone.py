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

        print(
            "Loading checkpoint",
            "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/checkpoint/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        )
        checkpoint = torch.load(
            "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/checkpoint/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
            map_location=lambda storage, loc: storage,
        )
        
        
        state = checkpoint.get("model_pos", None)
        if state is None:
            # fallbacks in case the container key differs
            for k in ("state_dict", "model", "ema_state_dict"):
                if isinstance(checkpoint.get(k, None), dict):
                    state = checkpoint[k]
                    print(f"Using sub-dict: {k}")
                    break
        if state is None:
            # As a last resort, try interpreting the whole checkpoint as a state dict
            state = {k: v for k, v in checkpoint.items() if hasattr(v, "size")}
            print("Using raw checkpoint as state dict")

        # If the checkpoint was saved under DataParallel, strip 'module.' prefix
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        # Try strict=False once to inspect mismatches (don't train like this if core blocks are missing)
        missing, unexpected = self.dstformer.load_state_dict(state, strict=False)
        print(f"Missing keys: {len(missing)} -> {missing[:20]}")
        print(f"Unexpected keys: {len(unexpected)} -> {unexpected[:20]}")

        
        
        if torch.cuda.is_available():
            self.dstformer = torch.nn.DataParallel(self.dstformer).cuda()


        self.dstformer.eval()

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, T, embed_dim]
        """
        out = self.dstformer(x, return_rep=True)

        assert out.dim() == 4  # [B, F, S, D]

        # TODO: do we need an MLP here?
        # out = out.mean(dim=1)  # [B, D]
        # # out = F.normalize(out, p=2, dim=1)s
        # print("MotionBERTBackbone output shape:", out.shape)

        return out
