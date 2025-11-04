from functools import partial
import jax
import jax.numpy as jnp
from videoprism import models as vp
import torch, torch.nn as nn, torch.nn.functional as F

from src.MotionBert.DSTformer import DSTformer

# import os

# # Force JAX to use CPU only, for every process that imports this file
# os.environ["JAX_PLATFORMS"] = "cpu"        # new-style flag
# os.environ["JAX_PLATFORM_NAME"] = "cpu"    # legacy flag
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# import jax
# jax.config.update("jax_platform_name", "cpu")

class MotionBERTBackbone(nn.Module):
    def __init__(self,):
        super(MotionBERTBackbone, self).__init__()
        # TODO: hardcode args for MotionBERT for now: https://github.com/Walter0807/MotionBERT/blob/main/configs/pose3d/MB_ft_h36m_global_lite.yaml
        
        print('Initializing MotionBERTBackbone...')
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
        
        #TODO: load pretrained MotionBERT weights
        # if torch.cuda.is_available():
        #     self.motionbert_backbone = nn.DataParallel(self.motionbert_backbone)
        #     self.motionbert_backbone = self.motionbert_backbone.cuda()

        # print('Loading checkpoint', args.motionbert_checkpoint)
        # checkpoint = torch.load(args.motionbert_checkpoint, map_location=lambda storage, loc: storage)
        # self.motionbert_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
        self.dstformer.requires_grad_(False)
        

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, T, embed_dim]
        """
        out = self.dstformer(x)
        
        assert out.dim() == 4  # [B, F, S, D]
        
        # TODO: do we need an MLP here?
        # out = out.mean(dim=1)  # [B, D]
        # out = F.normalize(out, p=2, dim=1)
        
        print('MotionBERTBackbone output shape:', out.shape)
        
        return out
    
class VideoPrismBackbone(nn.Module):
    def __init__(self, model_name = 'videoprism_public_v1_base', use_bfloat16 = False):
        super(VideoPrismBackbone, self).__init__()
        
        self.fprop_dtype = jnp.bfloat16 if use_bfloat16 else None
        self.flax_model = vp.get_model(model_name, fprop_dtype=self.fprop_dtype)
        self.loaded_state = vp.load_pretrained_weights(model_name)
        
    def forward(self, x, train=False):
        """
        x: [B, T, input_dim]
        return: [B, T, embed_dim]
        """
        if self.fprop_dtype is not None:
            x = x.astype(jnp.bfloat16)
            
        print(f'Input shape: {x.shape} [type: {x.dtype}]')
        
        embeddings, _ = self.flax_model.apply(self.loaded_state, x, train=train)
        print(f'Encoded embedding shape: {embeddings.shape} [type: {embeddings.dtype}]')

        return embeddings