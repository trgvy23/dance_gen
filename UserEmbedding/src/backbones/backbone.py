from functools import partial
import torch, torch.nn as nn, torch.nn.functional as F

from src.backbones import DSTformer

def freeze_module(m: nn.Module, eval_mode: bool = True):
    for p in m.parameters():
        p.requires_grad = False
    if eval_mode:
        m.eval()
    return m

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
        
        # if torch.cuda.is_available():
        #     self.motionbert_backbone = nn.DataParallel(self.motionbert_backbone)
        #     self.motionbert_backbone = self.motionbert_backbone.cuda()

        # print('Loading checkpoint', args.motionbert_checkpoint)
        # checkpoint = torch.load(args.motionbert_checkpoint, map_location=lambda storage, loc: storage)
        # self.motionbert_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
        self.dstformers = freeze_module(self.dstformer, eval_mode=True)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, T, embed_dim]
        """
        out = self.dstformer(x)
        
        print("Output shape: ", out.shape)
        
        assert out.dim() == 3  # [B, T, D]
        
        #TODO: do we need an MLP here?
        out = out.mean(dim=1)  # [B, D]
        out = F.normalize(out, p=2, dim=1)
        
        print('MotionBERTBackbone output shape:', out.shape)
        
        return out
    
class VideoPrismBackbone(nn.Module):
    def __init__(self,):
        super(VideoPrismBackbone, self).__init__()
        
    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, T, embed_dim]
        """
        pass